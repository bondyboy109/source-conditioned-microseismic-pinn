"""Independent random-collocation audit of an eikonal checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eikonal_traveltime import load_eikonal_checkpoint
from source_conditioned_pinn import SourceConditionedConfig, sample_source_condition
from train_source_conditioned_pinn import GridVelocity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260924)
    model, _ = load_eikonal_checkpoint(args.checkpoint, device)
    velocity = GridVelocity(args.velocity, device)
    cfg = SourceConditionedConfig()
    source = sample_source_condition(args.samples, device, cfg)[:, :3]
    xyz = torch.rand(args.samples, 3, device=device, requires_grad=True)
    travel = model(xyz, source)
    gradient = torch.autograd.grad(travel.sum(), xyz)[0]
    physical = gradient / xyz.new_tensor(model.domain_m)
    predicted = torch.linalg.vector_norm(physical, dim=-1)
    target = 1.0 / velocity(xyz).squeeze(-1)
    relative = ((predicted - target) / target).detach().cpu().numpy()
    absolute = np.abs(relative)
    report = {
        "sample_count": args.samples,
        "relative_residual_rmse": float(np.sqrt(np.mean(relative ** 2))),
        "relative_residual_mae": float(np.mean(absolute)),
        "relative_residual_p95": float(np.quantile(absolute, 0.95)),
        "relative_residual_max": float(np.max(absolute)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
