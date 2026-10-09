"""Compare traveltime interfaces with FD first-arrival picks."""

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
from examples.compare_ray_shifted_template import ray_travel_times
from source_conditioned_pinn import SourceConditionedConfig


def first_arrival(trace: np.ndarray, times: np.ndarray,
                  threshold_fraction: float = 0.05,
                  consecutive: int = 3) -> float:
    peak = float(np.max(np.abs(trace)))
    if peak <= 0:
        return float("nan")
    active = np.abs(trace) >= threshold_fraction * peak
    run = np.convolve(active.astype(np.int8),
                      np.ones(consecutive, dtype=np.int8), mode="valid")
    indices = np.flatnonzero(run >= consecutive)
    return float(times[indices[0]]) if len(indices) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eikonal-checkpoint", type=Path, required=True)
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--observation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-velocity", type=float, default=3500.0)
    parser.add_argument("--threshold-fraction", type=float, default=0.05)
    parser.add_argument("--ray-samples", type=int, default=64)
    args = parser.parse_args()
    if not 0 < args.threshold_fraction < 1 or args.ray_samples < 2:
        raise ValueError("Invalid first-arrival parameters")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eikonal, payload = load_eikonal_checkpoint(args.eikonal_checkpoint, device)
    model = np.load(args.velocity, mmap_mode="r")
    cfg = SourceConditionedConfig()
    scale = np.asarray([cfg.x_length_m, cfg.y_length_m,
                        cfg.z_length_m, cfg.time_length_s], dtype=np.float32)
    records = []
    for path in args.observation:
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float64)
            times = np.asarray(archive["times"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
            source = np.asarray(archive["source_truth"], dtype=np.float64)
        observed = np.asarray([first_arrival(trace, times,
                                              args.threshold_fraction, 3)
                              for trace in data])
        with torch.inference_mode():
            xyz = torch.as_tensor(receivers / scale[:3], device=device,
                                  dtype=torch.float32)
            src = torch.as_tensor(np.broadcast_to(source[:3] / scale[:3],
                                                  (len(receivers), 3)),
                                  device=device, dtype=torch.float32)
            eikonal_travel = eikonal(xyz, src).squeeze(-1).cpu().numpy()
        constant_travel = (np.linalg.norm(receivers - source[:3], axis=1)
                           / args.reference_velocity)
        integrated_travel = ray_travel_times(model, source, receivers,
                                             samples=args.ray_samples)
        predictions = {
            "fd_pick": observed,
            "constant_3500": source[3] + constant_travel,
            "straight_ray_integrated": source[3] + integrated_travel,
            "eikonal": source[3] + eikonal_travel,
        }
        event = {"observation": str(path), "source": source.tolist(),
                 "receiver_count": len(receivers),
                 "valid_fd_picks": int(np.isfinite(observed).sum()),
                 "interfaces": {}}
        valid = np.isfinite(observed)
        for name, prediction in predictions.items():
            if name == "fd_pick":
                continue
            residual = prediction[valid] - observed[valid]
            event["interfaces"][name] = {
                "mean_error_s": float(np.mean(residual)),
                "mae_s": float(np.mean(np.abs(residual))),
                "rmse_s": float(np.sqrt(np.mean(residual ** 2))),
                "p95_abs_error_s": float(np.quantile(np.abs(residual), 0.95)),
                "predicted_s": prediction.tolist(),
            }
        event["fd_pick_s"] = observed.tolist()
        records.append(event)
    errors = {name: [] for name in ("constant_3500", "straight_ray_integrated", "eikonal")}
    for event in records:
        for name in errors:
            errors[name].append(event["interfaces"][name]["rmse_s"])
    report = {
        "definition": {
            "threshold_fraction_of_trace_peak": args.threshold_fraction,
            "consecutive_samples": 3,
            "dt_s": 0.002,
            "eikonal_checkpoint": str(args.eikonal_checkpoint.resolve()),
            "eikonal_final_loss": payload["history"][-1],
        },
        "event_count": len(records),
        "events": records,
        "event_rmse_summary_s": {
            name: {"mean": float(np.mean(values)), "median": float(np.median(values)),
                   "max": float(np.max(values))}
            for name, values in errors.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["event_rmse_summary_s"], indent=2))


if __name__ == "__main__":
    main()
