"""Compare scikit-fmm heterogeneous first-arrival times with FD picks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import skfmm


def first_arrival(trace, times, threshold=0.05, consecutive=3):
    peak = float(np.max(np.abs(trace)))
    if peak <= 0:
        return np.nan
    active = np.abs(trace) >= threshold * peak
    run = np.convolve(active.astype(np.int8), np.ones(consecutive, dtype=np.int8), "valid")
    ids = np.flatnonzero(run >= consecutive)
    return float(times[ids[0]]) if len(ids) else np.nan


def trilinear(field: np.ndarray, points: np.ndarray, spacing: float) -> np.ndarray:
    position = points / spacing
    lower = np.floor(position).astype(int)
    lower = np.clip(lower, 0, np.asarray(field.shape) - 2)
    frac = position - lower
    value = np.zeros(len(points), dtype=np.float64)
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                weight = np.prod(np.where((ox, oy, oz), frac, 1.0 - frac), axis=1)
                value += weight * field[lower[:, 0] + ox, lower[:, 1] + oy, lower[:, 2] + oz]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--observation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=40.0)
    args = parser.parse_args()
    velocity = np.asarray(np.load(args.velocity), dtype=np.float64)
    records = []
    for path in args.observation:
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float64)
            times = np.asarray(archive["times"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
            source = np.asarray(archive["source_truth"], dtype=np.float64)
        index = np.rint(source[:3] / args.spacing).astype(int)
        index = np.clip(index, 0, np.asarray(velocity.shape) - 1)
        phi = np.ones(velocity.shape, dtype=np.float64)
        phi[tuple(index)] = -1.0
        travel = np.asarray(skfmm.travel_time(phi, speed=velocity, dx=args.spacing), dtype=np.float64)
        predicted = source[3] + trilinear(travel, receivers, args.spacing)
        observed = np.asarray([first_arrival(trace, times) for trace in data])
        valid = np.isfinite(observed)
        residual = predicted[valid] - observed[valid]
        records.append({
            "observation": str(path),
            "source": source.tolist(),
            "valid_fd_picks": int(valid.sum()),
            "mae_s": float(np.mean(np.abs(residual))),
            "rmse_s": float(np.sqrt(np.mean(residual ** 2))),
            "p95_abs_error_s": float(np.quantile(np.abs(residual), 0.95)),
        })
    summary = {key: float(np.mean([row[key] for row in records]))
               for key in ("mae_s", "rmse_s", "p95_abs_error_s")}
    result = {"method": "scikit_fmm_first_arrival", "spacing_m": args.spacing,
              "events": records, "mean": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
