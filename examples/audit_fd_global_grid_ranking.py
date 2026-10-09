"""Diagnose grid discretization versus waveform ranking on the frozen FD pilot.

True locations select nearest grid nodes only for this post-hoc audit. They do
not alter the original 75-node global search or its reported estimates.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.run_fd_waveform_global_refined_confirmation import global_grid
from fd_waveform import best_origin_time, simulate_candidates

PROTOCOL = ROOT / "experiments/fd_waveform_global_refined_confirmation_protocol.json"
BASELINE = ROOT / "experiments/fd_waveform_global_refined_confirmation_results.json"
TRUTH_AUDIT = ROOT / "experiments/fd_truth_waveform_confirmation_audit.json"
RESULT = ROOT / "experiments/fd_global_grid_ranking_audit.json"
REPORT = RESULT.with_suffix(".md")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("Audit result already exists; refusing to overwrite")
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    truth_audit = json.loads(TRUTH_AUDIT.read_text(encoding="utf-8"))
    if (baseline["protocol_sha256"] != digest(PROTOCOL)
            or truth_audit["baseline_protocol_sha256"] != digest(PROTOCOL)
            or truth_audit["candidate_solver_sha256"] != digest(ROOT / "fd_waveform.py")):
        raise ValueError("Baseline protocol or solver changed")
    baseline_by_id = {r["id"]: r for r in baseline["events"]}
    if len(baseline_by_id) != len(spec["events"]):
        raise ValueError("Incomplete original baseline")
    cfg_path = ROOT / spec["fd_config"]
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if truth_audit["config_sha256"] != digest(cfg_path):
        raise ValueError("FD configuration changed")
    bounds = np.asarray(cfg["source_location_bounds_m"], dtype=np.float64).T
    grid = global_grid(bounds, spec["spatial_grid_counts"])
    if len(grid) != baseline["candidate_count"]:
        raise ValueError("Baseline candidate grid changed")

    observations = []
    required = []
    for event in spec["events"]:
        path = ROOT / event["observation"]
        if digest(path) != event["observation_sha256"]:
            raise ValueError(f"Observation changed: {path}")
        with np.load(path) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            mask = (times >= spec["waveform_window_s"][0]) & (
                times <= spec["waveform_window_s"][1] + 1e-9)
            observed = np.asarray(archive["data"][:, mask], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        old = baseline_by_id[event["id"]]
        if (not np.array_equal(truth, np.asarray(old["source_truth_m_s"]))
                or old["observation_sha256"] != event["observation_sha256"]):
            raise ValueError(f"Baseline event differs: {event['id']}")
        distances = np.linalg.norm(grid - truth[:3], axis=1)
        nearest_index = int(np.argmin(distances))
        winner = np.asarray(old["estimated_source_m_s"][:3], dtype=np.float64)
        matches = np.flatnonzero(np.all(grid == winner, axis=1))
        if len(matches) != 1 or not np.isclose(distances[nearest_index], old["nearest_grid_error_m"]):
            raise ValueError(f"Original grid reconstruction differs: {event['id']}")
        winner_index = int(matches[0])
        observations.append((event, observed, truth, nearest_index, winner_index))
        required.extend((nearest_index, winner_index))

    unique_indices = sorted(set(required))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Synthesizing {len(unique_indices)} unique diagnostic FD templates on {device}", flush=True)
    templates, fd_report = simulate_candidates(
        cfg_path, grid[unique_indices], spec["template_duration_s"],
        origin_time_s=spec["reference_origin_time_s"],
        batch_size=min(4, len(unique_indices)), device=device)
    template_by_index = dict(zip(unique_indices, templates))
    rows = []
    for event, observed, truth, nearest_index, winner_index in observations:
        old = baseline_by_id[event["id"]]
        nearest = best_origin_time(
            template_by_index[nearest_index], observed, fd_report["time_step_s"],
            spec["reference_origin_time_s"], tuple(spec["origin_time_bounds_s"]))
        winner = best_origin_time(
            template_by_index[winner_index], observed, fd_report["time_step_s"],
            spec["reference_origin_time_s"], tuple(spec["origin_time_bounds_s"]))
        if not np.isclose(winner["normalized_mse"], old["normalized_mse"],
                          rtol=2e-4, atol=1e-7):
            raise ValueError(f"Original winning score did not reproduce: {event['id']}")
        if nearest["normalized_mse"] + 1e-7 < winner["normalized_mse"]:
            raise ValueError(f"Original winner lost to nearest candidate: {event['id']}")
        row = {
            "id": event["id"], "observation_sha256": event["observation_sha256"],
            "truth_source_m_s_post_ranking": truth.tolist(),
            "nearest_grid_index": nearest_index,
            "nearest_grid_xyz_m": grid[nearest_index].tolist(),
            "nearest_grid_location_error_m": float(np.linalg.norm(grid[nearest_index] - truth[:3])),
            "nearest_grid_score": nearest,
            "winner_grid_index": winner_index,
            "winner_grid_xyz_m": grid[winner_index].tolist(),
            "winner_location_error_m": old["location_error_m"],
            "winner_score_recomputed": winner,
            "winner_score_original": old["normalized_mse"],
            "excess_location_error_over_nearest_m": (
                old["location_error_m"] - float(np.linalg.norm(grid[nearest_index] - truth[:3]))),
            "winner_is_nearest": winner_index == nearest_index,
        }
        rows.append(row)
        print(f"{event['id']}: nearest={row['nearest_grid_location_error_m']:.1f} m "
              f"(MSE={nearest['normalized_mse']:.4f}); "
              f"winner={old['location_error_m']:.1f} m "
              f"(MSE={winner['normalized_mse']:.4f})", flush=True)

    result = {
        "role": "post-hoc 75-node FD grid and waveform ranking audit; not a new inverse estimate",
        "protocol_sha256": digest(PROTOCOL),
        "baseline_sha256": digest(BASELINE),
        "truth_audit_sha256": digest(TRUTH_AUDIT),
        "config_sha256": digest(cfg_path),
        "solver_sha256": digest(ROOT / "fd_waveform.py"),
        "device": str(device), "unique_diagnostic_templates": len(unique_indices),
        "template_runtime_seconds": fd_report["runtime_seconds"],
        "peak_cuda_memory_mib": (fd_report["peak_cuda_memory_bytes"] / 2**20
                                 if fd_report["peak_cuda_memory_bytes"] else None),
        "events": rows,
        "summary": {
            "event_count": len(rows),
            "nearest_selected_count": sum(r["winner_is_nearest"] for r in rows),
            "median_nearest_grid_error_m": float(np.median([
                r["nearest_grid_location_error_m"] for r in rows])),
            "median_winner_location_error_m": float(np.median([
                r["winner_location_error_m"] for r in rows])),
            "median_excess_error_m": float(np.median([
                r["excess_location_error_over_nearest_m"] for r in rows])),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Post-hoc FD global-grid ranking audit", "",
             "The nearest grid node is selected using the true source only for "
             "this diagnostic. It was never supplied to the global FD search.", "",
             "| Event | Nearest-node error (m) | Winner error (m) | "
             "Nearest MSE | Winner MSE | Nearest selected? |",
             "|---|---:|---:|---:|---:|:---:|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['nearest_grid_location_error_m']:.1f} | "
                     f"{r['winner_location_error_m']:.1f} | "
                     f"{r['nearest_grid_score']['normalized_mse']:.4f} | "
                     f"{r['winner_score_recomputed']['normalized_mse']:.4f} | "
                     f"{'yes' if r['winner_is_nearest'] else 'no'} |")
    lines.extend(["", f"{result['summary']['nearest_selected_count']} of {len(rows)} "
                  "winning nodes are nearest by distance. The nearest-node error "
                  "is an oracle grid floor, not an achievable blind-search result. "
                  "Both columns optimize the same discrete origin-time shift, "
                  "and scores use the unchanged normalized waveform MSE.", "",
                  f"Median nearest-node distance: "
                  f"{result['summary']['median_nearest_grid_error_m']:.1f} m; "
                  f"median selected-node distance: "
                  f"{result['summary']['median_winner_location_error_m']:.1f} m. "
                  "The recomputed winning scores match the frozen baseline. "
                  "In some events the nearest node lies on the 100 m depth plane "
                  "and has a much larger waveform loss. This suggests examining "
                  "depth sampling and boundary sensitivity, but this audit does "
                  "not identify a unique cause or validate a finer inverse.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {RESULT} and {REPORT}", flush=True)


if __name__ == "__main__":
    main()
