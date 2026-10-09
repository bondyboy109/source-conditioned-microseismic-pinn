"""Diagnose FMM grid quantization versus FD arrival-model mismatch on development data."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.invert_fmm_receiver_fields import interpolate_fields, receiver_fields, spatial_grid
from examples.validate_arrival_constrained import digest, event_files, load_protocol, pick_receivers, source_search

DEVELOPMENT = ROOT / "experiments/arrival_bias_development_results.json"
OUTPUT = ROOT / "experiments/arrival_spatial_resolution_development.json"


def rank_candidates(fields, picks, points, spec):
    """Rank the same clipped median-origin RMS objective used by source_search."""
    valid = np.isfinite(picks)
    if valid.sum() < spec["min_valid_picks"]:
        raise ValueError("Too few arrival picks")
    travel = interpolate_fields(fields[valid], points, spec["fmm_spacing_m"])
    origin = np.clip(np.median(picks[valid][None, :] - travel, axis=1),
                     *spec["origin_bounds_s"])
    rmse = np.sqrt(np.mean((travel + origin[:, None] - picks[valid]) ** 2, axis=1))
    return rmse, origin


def refined_search(fields, picks, spec, *, seeds=3, rounds=3):
    """Search coarse candidates, then successively finer neighborhoods of top seeds."""
    bounds = np.asarray(spec["source_bounds_m"], dtype=np.float64)
    coarse = spatial_grid(bounds, tuple(spec["coarse_grid"]))
    coarse_rmse, coarse_origin = rank_candidates(fields, picks, coarse, spec)
    order = np.argsort(coarse_rmse, kind="stable")[:seeds]
    current = coarse[order]
    step = (bounds[:, 1] - bounds[:, 0]) / (np.asarray(spec["coarse_grid"]) - 1)
    visited = len(coarse)
    offsets = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)))
    for _ in range(rounds):
        step = step / 2
        trial = np.clip(current[:, None, :] + offsets[None, :, :] * step,
                        bounds[:, 0], bounds[:, 1]).reshape(-1, 3)
        # Retain the best coarse seeds even when the clipped neighborhood duplicates them.
        candidates = np.unique(np.concatenate((current, trial)), axis=0)
        visited += len(candidates)
        rmse, _ = rank_candidates(fields, picks, candidates, spec)
        current = candidates[np.argsort(rmse, kind="stable")[:seeds]]
    scores, origin = rank_candidates(fields, picks, current, spec)
    best = int(np.argmin(scores))
    return np.r_[current[best], origin[best]], float(scores[best]), visited


def summary(rows, key):
    return {"mean_location_error_m": float(np.mean([row["methods"][key]["location_error_m"] for row in rows])),
            "median_location_error_m": float(np.median([row["methods"][key]["location_error_m"] for row in rows])),
            "mean_origin_error_s": float(np.mean([row["methods"][key]["origin_error_s"] for row in rows])),
            "at_depth_upper_bound": int(sum(row["methods"][key]["estimated_source_m_s"][2] >= 1499.999
                                            for row in rows))}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    spec = load_protocol()
    development = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    paths = event_files(spec, "development")
    recorded = {Path(event["observation"]): event for event in development["events"]}
    if (len(paths) != len(recorded) or any(path not in recorded or
            digest(path) != recorded[path]["observation_sha256"] for path in paths)):
        raise ValueError("Development data changed since bias calibration")
    velocity = np.asarray(np.load(ROOT / spec["velocity"], mmap_mode="r")
                          [::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"]],
                          dtype=np.float32)
    receivers = np.load(ROOT / spec["receivers"])
    fields = receiver_fields(velocity, receivers, spec["fmm_spacing_m"])

    def score(estimate, truth, rmse):
        return {"estimated_source_m_s": estimate.tolist(),
                "location_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
                "origin_error_s": float(abs(estimate[3] - truth[3])),
                "arrival_rmse_s": float(rmse)}

    rows = []
    bounds = np.asarray(spec["source_bounds_m"], dtype=np.float64)
    coarse = spatial_grid(bounds, tuple(spec["coarse_grid"]))
    for path in paths:
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float64)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
            if not np.array_equal(archive["receiver_coords"], receivers):
                raise ValueError(f"Receiver layout differs: {path}")
        row = recorded[path]
        picks = pick_receivers(data, times, development["selected_picker"], spec)
        bias = row["leave_one_event_out_bias_s"]
        observed = picks - bias
        # Preserve the exact receiver selection when constructing ideal travel times.
        ideal = truth[3] + interpolate_fields(fields, truth[None, :3],
                                              spec["fmm_spacing_m"])[0]
        ideal[~np.isfinite(picks)] = np.nan
        record = {"observation": str(path), "observation_sha256": digest(path),
                  "valid_picks": int(np.isfinite(picks).sum()),
                  "true_source_m_s": truth.tolist(),
                  "geometric_nearest_coarse_m": float(np.min(np.linalg.norm(coarse - truth[:3], axis=1))),
                  "receiver_centered_mismatch_rms_s": float(np.std((observed - ideal)[np.isfinite(ideal)])),
                  "receiver_centered_mismatch_mad_s": float(np.median(np.abs(
                      (observed - ideal)[np.isfinite(ideal)] -
                      np.median((observed - ideal)[np.isfinite(ideal)])))),
                  "methods": {}}
        for label, arrivals in (("observed", observed), ("ideal", ideal)):
            grid = source_search(fields, arrivals, spec)
            points, residuals, _ = grid
            best = int(np.argmin(residuals))
            record["methods"][f"{label}_existing"] = score(points[best], truth,
                                                             residuals[best])
            refined, rmse, count = refined_search(fields, arrivals, spec)
            record["methods"][f"{label}_refined"] = score(refined, truth, rmse)
            record["refined_candidate_evaluations"] = count
        rows.append(record)
        print(f"{path.parent.name}/{path.name}: observed "
              f"{record['methods']['observed_existing']['location_error_m']:.0f} -> "
              f"{record['methods']['observed_refined']['location_error_m']:.0f} m; "
              f"ideal {record['methods']['ideal_existing']['location_error_m']:.0f} -> "
              f"{record['methods']['ideal_refined']['location_error_m']:.0f} m", flush=True)
    result = {"role": "development-only diagnostic; never select settings from inspected holdouts",
              "development_sha256": digest(DEVELOPMENT),
              "selection": {"top_coarse_seeds": 3, "refinement_rounds": 3,
                            "step_per_round": "coarse_grid_spacing / (2, 4, 8)",
                            "objective": "clipped median origin, arrival RMSE as in source_search"},
              "summary": {f"{kind}_{method}": summary(rows, f"{kind}_{method}")
                          for kind in ("observed", "ideal")
                          for method in ("existing", "refined")},
              "mean_nearest_coarse_m": float(np.mean([row["geometric_nearest_coarse_m"] for row in rows])),
              "mean_receiver_centered_mismatch_rms_s": float(np.mean([
                  row["receiver_centered_mismatch_rms_s"] for row in rows])),
              "events": rows}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
