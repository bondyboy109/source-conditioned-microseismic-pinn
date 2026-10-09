"""Inspect the wave equation at and around a known source for diagnostics."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source_conditioned_pinn import load_conditioned_checkpoint, source_term


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--velocity", type=Path,
                        default=ROOT / "data" / "velocity" / "vp_model.npy")
    parser.add_argument("--source", type=float, nargs=4,
                        default=[7000.0, 8000.0, 500.0, 0.2])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(args.checkpoint, device)
    from source_conditioned_pinn import SourceConditionedConfig
    config_keys = {field.name for field in fields(SourceConditionedConfig)}
    cfg = SourceConditionedConfig(**{
        key: value for key, value in metadata["config"].items()
        if key in config_keys
    })
    lengths = np.array([cfg.x_length_m, cfg.y_length_m,
                        cfg.z_length_m, cfg.time_length_s], dtype=np.float32)
    source = np.array(args.source, dtype=np.float32)
    velocity = np.load(args.velocity, mmap_mode="r")
    idx = np.rint(source[:3] / lengths[:3] * (np.array(velocity.shape) - 1)).astype(int)
    speed = float(velocity[tuple(idx)])
    query = np.stack((source, source + [100, 0, 0, 0],
                      source + [0, 0, 0, 0.05])).astype(np.float32)
    space_time = torch.tensor(query / lengths, device=device, requires_grad=True)
    src = torch.tensor(np.broadcast_to(source / lengths, query.shape).copy(), device=device)
    u = model(space_time, src)
    grad = torch.autograd.grad(u.sum(), space_time, create_graph=True)[0]
    d2 = [torch.autograd.grad(grad[:, i].sum(), space_time,
                              create_graph=False, retain_graph=True)[0][:, i]
          for i in range(4)]
    lhs = (d2[3] / cfg.time_length_s**2 - speed**2 *
           sum(d2[i] / length**2 for i, length in enumerate(lengths[:3])))
    rhs = source_term(space_time, src, metadata["frequency_hz"], cfg).squeeze(-1)
    for name, output, left, right in zip(
            ("source at origin time", "100 m from source", "source +0.05 s"),
            u.detach().cpu().squeeze(-1).tolist(),
            lhs.detach().cpu().tolist(), rhs.detach().cpu().tolist()):
        print(f"{name}: u={output:.5g}, PDE lhs={left:.5g}, forcing={right:.5g}, "
              f"residual={left-right:.5g}")


if __name__ == "__main__":
    main()
