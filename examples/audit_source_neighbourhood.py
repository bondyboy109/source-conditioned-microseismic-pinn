"""Quantify acoustic PDE residual around a registered source."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source_conditioned_pinn import (
    SourceConditionedConfig,
    load_conditioned_checkpoint,
    source_term,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--source", type=float, nargs=4,
                        default=[3500.0, 4000.0, 500.0, 0.12])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(args.checkpoint, device)
    if metadata.get("architecture", {}).get("ray_travel_time", False):
        model.set_velocity_grid(torch.as_tensor(
            np.array(np.load(args.velocity, mmap_mode="r"), dtype=np.float32),
            device=device))
    config_keys = {field.name for field in fields(SourceConditionedConfig)}
    cfg = SourceConditionedConfig(**{
        key: value for key, value in metadata["config"].items()
        if key in config_keys
    })
    lengths = np.asarray([cfg.x_length_m, cfg.y_length_m,
                          cfg.z_length_m, cfg.time_length_s], dtype=np.float32)
    source = np.asarray(args.source, dtype=np.float32)
    distances = np.asarray([0, 25, 50, 75, 100, 150, 200], dtype=np.float32)
    time_offsets = np.asarray([-0.10, -0.05, 0.0, 0.05, 0.10], dtype=np.float32)
    samples = []
    for distance in distances:
        directions = [(0, 0, 0)] if distance == 0 else (
            (distance, 0, 0), (-distance, 0, 0),
            (0, distance, 0), (0, -distance, 0),
            (0, 0, distance), (0, 0, -distance),
        )
        for direction in directions:
            for time_offset in time_offsets:
                samples.append([source[0] + direction[0],
                                source[1] + direction[1],
                                source[2] + direction[2],
                                source[3] + time_offset])
    query = np.asarray(samples, dtype=np.float32)
    if np.any(query < 0) or np.any(query > lengths):
        raise ValueError("Source-neighbourhood query lies outside the PINN domain")

    velocity = np.load(args.velocity, mmap_mode="r")
    indices = np.rint(query[:, :3] / lengths[:3]
                      * (np.asarray(velocity.shape) - 1)).astype(int)
    speeds = np.asarray([velocity[tuple(index)] for index in indices],
                        dtype=np.float32)
    space_time = torch.tensor(query / lengths, device=device, requires_grad=True)
    source_batch = torch.tensor(
        np.broadcast_to(source / lengths, query.shape).copy(), device=device)
    speed = torch.tensor(speeds, device=device)
    prediction = model(space_time, source_batch)
    gradient = torch.autograd.grad(prediction.sum(), space_time,
                                   create_graph=True)[0]
    second = [torch.autograd.grad(gradient[:, axis].sum(), space_time,
                                  retain_graph=True)[0][:, axis]
              for axis in range(4)]
    lhs = (second[3] / cfg.time_length_s ** 2
           - speed.square() * sum(second[axis] / lengths[axis] ** 2
                                  for axis in range(3)))
    rhs = source_term(space_time, source_batch,
                      metadata["frequency_hz"], cfg).squeeze(-1)
    residual = lhs - rhs
    lhs_np = lhs.detach().cpu().numpy()
    rhs_np = rhs.detach().cpu().numpy()
    residual_np = residual.detach().cpu().numpy()
    active = np.abs(rhs_np) >= 0.05
    source_peak = np.all(np.isclose(query, source, atol=1e-6), axis=1)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source": source.tolist(),
        "distances_m": distances.tolist(),
        "time_offsets_s": time_offsets.tolist(),
        "sample_count": int(len(query)),
        "active_forcing_sample_count": int(active.sum()),
        "residual_rmse_all": float(np.sqrt(np.mean(residual_np ** 2))),
        "residual_rmse_active_forcing": float(np.sqrt(np.mean(residual_np[active] ** 2))),
        "relative_residual_rmse_active_forcing": float(
            np.sqrt(np.mean(residual_np[active] ** 2)
                    / np.mean(rhs_np[active] ** 2))),
        "source_peak_lhs": float(lhs_np[source_peak][0]),
        "source_peak_forcing": float(rhs_np[source_peak][0]),
        "source_peak_residual": float(residual_np[source_peak][0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
