"""FMM baseline: solve one reciprocal traveltime volume per fixed receiver."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import skfmm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_fmm_traveltime import first_arrival


SOURCE_BOUNDS = np.array([[2000., 14000.], [2000., 14000.],
                          [100., 1500.]], dtype=np.float64)


class InsufficientPicks(ValueError):
    def __init__(self, count: int):
        self.count = count
        super().__init__(f"Only {count} usable first-arrival picks")


def receiver_fields(velocity: np.ndarray, receivers: np.ndarray,
                    spacing_m: float) -> np.ndarray:
    """Compute receiver-centred Eikonal fields once, then reuse via reciprocity."""
    result = np.empty((len(receivers), *velocity.shape), dtype=np.float32)
    for index, receiver in enumerate(receivers):
        cell = np.rint(receiver / spacing_m).astype(int)
        if np.any(cell < 0) or np.any(cell >= velocity.shape):
            raise ValueError(f"Receiver outside FMM grid: {receiver}")
        level_set = np.ones(velocity.shape, dtype=np.float64)
        level_set[tuple(cell)] = -1.
        result[index] = np.asarray(
            skfmm.travel_time(level_set, speed=velocity, dx=spacing_m),
            dtype=np.float32)
        print(f"FMM field {index+1}/{len(receivers)}", flush=True)
    return result


def interpolate_fields(fields: np.ndarray, points: np.ndarray,
                       spacing_m: float) -> np.ndarray:
    """Return [candidate, receiver] times for multiple candidate sources."""
    shape = np.asarray(fields.shape[1:])
    position = np.asarray(points, dtype=np.float64) / spacing_m
    if (position.ndim != 2 or position.shape[1] != 3
            or np.any(position < 0) or np.any(position > shape - 1)):
        raise ValueError("All candidate sources must lie within the traveltime grid")
    lower = np.minimum(np.floor(position).astype(int), shape - 2)
    frac = position - lower
    value = np.zeros((fields.shape[0], len(points)), dtype=np.float64)
    for ox, oy, oz in itertools.product((0, 1), repeat=3):
        weight = np.prod(np.where((ox, oy, oz), frac, 1 - frac), axis=1)
        value += fields[:, lower[:, 0] + ox, lower[:, 1] + oy,
                        lower[:, 2] + oz] * weight[None, :]
    return value.T


def score_candidates(fields: np.ndarray, candidates: np.ndarray, picks: np.ndarray,
                     spacing_m: float, time_bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(picks)
    arrivals = interpolate_fields(fields[valid], candidates, spacing_m)
    origins = np.clip(np.median(picks[valid] - arrivals, axis=1), *time_bounds)
    residual = arrivals + origins[:, None] - picks[valid]
    return np.mean(residual**2, axis=1), origins


def spatial_grid(bounds: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(lo, hi, num) for (lo, hi), num in zip(bounds, shape)]
    return np.asarray(list(itertools.product(*axes)), dtype=np.float64)


def invert_one(fields: np.ndarray, receivers: np.ndarray,
               data: np.ndarray, times: np.ndarray,
               spacing_m: float, coarse_grid: tuple[int, int, int],
               refinement_grid: int, refinement_levels: int,
               time_bounds: tuple[float, float], pick_threshold: float,
               min_valid_picks: int, min_peak_fraction: float = .01,
               end_margin_s: float = .15) -> dict:
    """No source truth enters the picker, candidate search or optimizer."""
    picks = np.asarray([first_arrival(trace, times, threshold=pick_threshold)
                        for trace in data], dtype=np.float64)
    peaks = np.max(np.abs(data), axis=1)
    valid = (np.isfinite(picks) & (peaks >= min_peak_fraction * peaks.max())
             & (picks <= times[-1] - end_margin_s))
    if valid.sum() < min_valid_picks:
        raise InsufficientPicks(int(valid.sum()))
    picks[~valid] = np.nan
    bounds = SOURCE_BOUNDS.copy()
    candidates = spatial_grid(bounds, coarse_grid)
    losses, origins = score_candidates(fields, candidates, picks, spacing_m, time_bounds)
    best_idx = int(np.argmin(losses))
    best = (float(losses[best_idx]), candidates[best_idx].copy(), float(origins[best_idx]))
    steps = (bounds[:, 1] - bounds[:, 0]) / (np.asarray(coarse_grid) - 1)
    assessed = len(candidates)
    for _ in range(refinement_levels):
        low = np.maximum(best[1] - steps, bounds[:, 0])
        high = np.minimum(best[1] + steps, bounds[:, 1])
        candidates = spatial_grid(np.stack((low, high), axis=1), (refinement_grid,) * 3)
        losses, origins = score_candidates(fields, candidates, picks, spacing_m, time_bounds)
        idx = int(np.argmin(losses))
        if losses[idx] < best[0]:
            best = (float(losses[idx]), candidates[idx].copy(), float(origins[idx]))
        steps = steps * (2. / (refinement_grid - 1))
        assessed += len(candidates)
    return {"estimated_source_m_s": np.r_[best[1], best[2]].tolist(),
            "arrival_rmse_s": float(np.sqrt(best[0])),
            "valid_fd_picks": int(valid.sum()),
            "picked_arrivals_s": [float(value) if np.isfinite(value) else None for value in picks],
            "candidate_count": assessed,
            "source_truth_used_in_search": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--observation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--coarse-grid", type=int, nargs=3, default=(11, 11, 8))
    parser.add_argument("--refinement-grid", type=int, default=7)
    parser.add_argument("--refinement-levels", type=int, default=4)
    parser.add_argument("--pick-threshold", type=float, default=.05)
    parser.add_argument("--min-peak-fraction", type=float, default=.01)
    parser.add_argument("--pick-end-margin-s", type=float, default=.15)
    parser.add_argument("--min-valid-picks", type=int, default=6)
    parser.add_argument("--origin-bounds", type=float, nargs=2, default=(.05, .50))
    args = parser.parse_args()
    if (args.stride < 1 or min(args.coarse_grid) < 2 or args.refinement_grid < 3
            or args.refinement_grid % 2 != 1 or args.refinement_levels < 0
            or not 0 < args.pick_threshold < 1 or args.min_valid_picks < 1
            or not 0 < args.min_peak_fraction < 1 or args.pick_end_margin_s < 0
            or args.origin_bounds[0] >= args.origin_bounds[1]):
        parser.error("Invalid grid, threshold or time settings")
    with np.load(args.observation[0]) as archive:
        receiver_coords = np.asarray(archive["receiver_coords"], dtype=np.float64)
    original = np.load(args.velocity, mmap_mode="r")
    if original.shape != (401, 401, 94):
        raise ValueError("Velocity grid must match the registered Overthrust geometry")
    spacing = 40. * args.stride
    velocity = np.ascontiguousarray(original[::args.stride, ::args.stride, ::args.stride],
                                    dtype=np.float32)
    if np.min(velocity) <= 0 or not np.isfinite(velocity).all():
        raise ValueError("Velocity field must be positive and finite")
    setup_started = time.perf_counter()
    fields = receiver_fields(velocity, receiver_coords, spacing)
    setup_seconds = time.perf_counter() - setup_started
    print(f"FMM setup: {setup_seconds:.2f}s; cached receiver fields={fields.nbytes/1048576:.1f} MiB",
          flush=True)
    records = []
    for path in args.observation:
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float64)
            times = np.asarray(archive["times"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if (data.shape != (len(receivers), len(times))
                or not np.array_equal(receivers, receiver_coords)
                or np.any(~np.isfinite(data))):
            raise ValueError(f"FD receiver observations changed: {path}")
        started = time.perf_counter()
        try:
            inversion = invert_one(fields, receivers, data, times, spacing,
                                   tuple(args.coarse_grid), args.refinement_grid,
                                   args.refinement_levels, tuple(args.origin_bounds),
                                   args.pick_threshold, args.min_valid_picks,
                                   args.min_peak_fraction, args.pick_end_margin_s)
        except InsufficientPicks as exc:
            inversion = {"status": "insufficient_valid_picks", "reason": str(exc),
                         "estimated_source_m_s": None, "valid_fd_picks": exc.count,
                         "location_error_m": None, "origin_time_error_s": None,
                         "source_truth_used_in_search": False}
        elapsed = time.perf_counter() - started
        if inversion["estimated_source_m_s"] is not None:
            estimate = np.asarray(inversion["estimated_source_m_s"])
            error = estimate - truth
            inversion.update({"status": "ok", "component_error_m_s": error.tolist(),
                              "location_error_m": float(np.linalg.norm(error[:3])),
                              "origin_time_error_s": float(abs(error[3]))})
        inversion.update({"observation": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "source_truth_m_s_post_search": truth.tolist(),
                          "search_seconds": elapsed})
        records.append(inversion)
        if inversion["status"] == "ok":
            print(f"{path.name}: location error={inversion['location_error_m']:.1f}m, "
                  f"time error={inversion['origin_time_error_s']:.4f}s, "
                  f"search={elapsed:.3f}s", flush=True)
        else:
            print(f"{path.name}: {inversion['reason']}", flush=True)
    output = {"method": "receiver_reciprocal_scikit_fmm_first_arrival",
              "velocity": str(args.velocity.resolve()), "grid_stride": args.stride,
              "grid_spacing_m": spacing, "coarse_grid_xyz": args.coarse_grid,
              "refinement_grid_xyz": args.refinement_grid,
              "refinement_levels": args.refinement_levels,
              "pick_threshold_fraction": args.pick_threshold,
              "minimum_station_peak_fraction": args.min_peak_fraction,
              "pick_end_margin_s": args.pick_end_margin_s,
              "pick_consecutive_samples": 3, "origin_bounds_s": args.origin_bounds,
              "setup_seconds": setup_seconds,
              "receiver_field_cache_mib": fields.nbytes/1048576,
              "events": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
