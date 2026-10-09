"""Coarse source localization using numerical FMM first-arrival times."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import skfmm

from audit_fmm_traveltime import first_arrival, trilinear


def fmm_receiver_times(velocity, source_xyz, receivers, spacing):
    index = np.rint(source_xyz / spacing).astype(int)
    index = np.clip(index, 0, np.asarray(velocity.shape) - 1)
    phi = np.ones(velocity.shape, dtype=np.float64)
    phi[tuple(index)] = -1.0
    field = np.asarray(skfmm.travel_time(phi, speed=velocity, dx=spacing))
    return trilinear(field, receivers, spacing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--observation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4,
                        help="Velocity-grid decimation for the coarse baseline")
    parser.add_argument("--grid", type=int, nargs=3, default=[8, 8, 5])
    parser.add_argument("--refine-grid", type=int, default=3,
                        help="Odd local refinement count per spatial axis")
    parser.add_argument("--ts-bounds", type=float, nargs=2, default=[0.05, 0.50])
    args = parser.parse_args()
    if args.stride < 1 or any(value < 2 for value in args.grid):
        raise ValueError("Invalid stride or grid")
    full = np.asarray(np.load(args.velocity), dtype=np.float64)
    velocity = full[::args.stride, ::args.stride, ::args.stride]
    spacing = 40.0 * args.stride
    extent = (np.asarray(full.shape) - 1) * 40.0
    axes = [np.linspace(2000.0, 14000.0, args.grid[0]),
            np.linspace(2000.0, 14000.0, args.grid[1]),
            np.linspace(100.0, 1500.0, args.grid[2])]
    candidates = np.asarray(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    coarse_steps = np.array([(axis[1] - axis[0]) for axis in axes])
    records = []
    for path in args.observation:
        started = time.perf_counter()
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float64)
            times = np.asarray(archive["times"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        observed = np.asarray([first_arrival(trace, times) for trace in data])
        valid = np.isfinite(observed)
        best = None

        def evaluate(candidate, current):
            travel = fmm_receiver_times(velocity, candidate, receivers, spacing)
            origin = float(np.median(observed[valid] - travel[valid]))
            origin = float(np.clip(origin, *args.ts_bounds))
            residual = origin + travel[valid] - observed[valid]
            loss = float(np.mean(residual ** 2))
            if current is None or loss < current[0]:
                return (loss, candidate.copy(), origin, float(np.sqrt(loss)))
            return current

        for candidate in candidates:
            best = evaluate(candidate, best)
        coarse_best = best
        if args.refine_grid >= 3:
            half = args.refine_grid // 2
            local_axes = [
                np.linspace(coarse_best[1][axis] - coarse_steps[axis],
                            coarse_best[1][axis] + coarse_steps[axis],
                            args.refine_grid)
                for axis in range(3)]
            local_candidates = np.asarray(
                np.meshgrid(*local_axes, indexing="ij")).reshape(3, -1).T
            local_candidates = np.clip(local_candidates,
                                       [2000.0, 2000.0, 100.0],
                                       [14000.0, 14000.0, 1500.0])
            for candidate in local_candidates:
                best = evaluate(candidate, best)
        estimate = np.r_[best[1], best[2]]
        delta = estimate - truth
        records.append({
            "observation": str(path), "source_truth": truth.tolist(),
            "estimated_source": estimate.tolist(),
            "component_error": delta.tolist(),
            "location_error_m": float(np.linalg.norm(delta[:3])),
            "origin_time_error_s": float(abs(delta[3])),
            "arrival_rmse_s": best[3], "valid_picks": int(valid.sum()),
            "candidate_count": int(len(candidates)),
            "refinement_count": int(args.refine_grid ** 3 if args.refine_grid >= 3 else 0),
            "runtime_seconds": time.perf_counter() - started,
        })
    result = {"method": "fmm_first_arrival_coarse_grid", "stride": args.stride,
              "grid": args.grid, "events": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
