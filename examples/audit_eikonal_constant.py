"""Compare a trained eikonal network with constant-velocity analytic times."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--velocity", type=float, default=3500.0)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_eikonal_checkpoint(args.checkpoint, device)
    generator = torch.Generator(device=device).manual_seed(2026)
    xyz = torch.rand((args.samples, 3), generator=generator, device=device)
    source = torch.rand((args.samples, 3), generator=generator, device=device)
    lower = torch.tensor([2000 / 16000, 2000 / 16000, 100 / 3720], device=device)
    upper = torch.tensor([14000 / 16000, 14000 / 16000, 1500 / 3720], device=device)
    source = lower + source * (upper - lower)
    lengths = torch.tensor(model.domain_m, device=device)
    distance = torch.linalg.vector_norm((xyz - source) * lengths, dim=-1)
    mask = distance >= 100.0
    xyz, source, distance = xyz[mask], source[mask], distance[mask]
    with torch.inference_mode():
        prediction = model(xyz, source).squeeze(-1)
    reference = distance / args.velocity
    error = prediction - reference
    report = {
        "sample_count": int(len(reference)),
        "relative_l2": float(torch.linalg.vector_norm(error)
                             / torch.linalg.vector_norm(reference)),
        "mean_absolute_error_s": float(error.abs().mean()),
        "p95_absolute_error_s": float(torch.quantile(error.abs(), 0.95)),
        "max_absolute_error_s": float(error.abs().max()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
