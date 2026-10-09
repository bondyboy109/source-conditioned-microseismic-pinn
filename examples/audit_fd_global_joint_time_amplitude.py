"""Post-hoc global FD search with jointly fitted origin time and source gain.

This changes the original unit-amplitude objective and is a diagnostic, not
an independently selected or accuracy-matched benchmark against the PINN.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.audit_fd_joint_time_amplitude import score_joint
from examples.run_fd_waveform_global_refined_confirmation import (
    global_grid, observed_record,
)
from fd_waveform import simulate_candidates

PROTOCOL = ROOT / "experiments/fd_waveform_global_refined_confirmation_protocol.json"
BASELINE = ROOT / "experiments/fd_waveform_global_refined_confirmation_results.json"
RESULT = ROOT / "experiments/fd_global_joint_time_amplitude_audit.json"
REPORT = RESULT.with_suffix(".md")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("Global joint audit already exists; refusing to overwrite")
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if baseline["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Baseline protocol changed")
    cfg_path = ROOT / spec["fd_config"]
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    bounds = np.asarray(cfg["source_location_bounds_m"], dtype=np.float64).T
    candidates = global_grid(bounds, spec["spatial_grid_counts"])
    if len(candidates) != baseline["candidate_count"]:
        raise ValueError("Candidate grid differs from baseline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Synthesizing {len(candidates)} FD templates on {device}", flush=True)
    templates, fd_report = simulate_candidates(
        cfg_path, candidates, spec["template_duration_s"],
        spec["reference_origin_time_s"], batch_size=spec["batch_size"], device=device,
    )
    rows = []
    for event, old in zip(spec["events"], baseline["events"]):
        if event["id"] != old["id"] or event["observation_sha256"] != old["observation_sha256"]:
            raise ValueError("Event order or hash differs from baseline")
        obs_path = ROOT / event["observation"]
        if digest(obs_path) != event["observation_sha256"]:
            raise ValueError(f"Observation changed: {obs_path}")
        observed, truth = observed_record(obs_path, tuple(spec["waveform_window_s"]))
        if not np.allclose(truth, old["source_truth_m_s"], atol=1e-5):
            raise ValueError("Truth differs from baseline")
        started = time.perf_counter()
        scores = [score_joint(template, observed, fd_report["time_step_s"],
                              spec["reference_origin_time_s"],
                              tuple(spec["origin_time_bounds_s"]))
                  for template in templates]
        ranking_seconds = time.perf_counter() - started
        winner_index = min(range(len(scores)), key=lambda i: scores[i]["normalized_mse"])
        winner = scores[winner_index]
        estimate = np.append(candidates[winner_index], winner["origin_time_s"])
        delta = estimate - truth
        rows.append({
            "id": event["id"], "observation_sha256": event["observation_sha256"],
            "candidate_index": winner_index, "estimated_source_m_s": estimate.tolist(),
            "source_truth_m_s": truth.tolist(), "location_error_m": float(np.linalg.norm(delta[:3])),
            "origin_time_error_s": float(abs(delta[3])), "nonnegative_gain": winner["nonnegative_gain"],
            "normalized_mse": winner["normalized_mse"], "relative_l2": math.sqrt(winner["normalized_mse"]),
            "ranking_seconds": ranking_seconds, "unit_amplitude_baseline": {
                "location_error_m": old["location_error_m"],
                "origin_time_error_s": old["origin_time_error_s"],
                "normalized_mse": old["normalized_mse"],
            },
        })
        print(f"{event['id']}: eloc={rows[-1]['location_error_m']:.1f} m, "
              f"et={rows[-1]['origin_time_error_s']:.3f} s, "
              f"gain={winner['nonnegative_gain']:.4f}, MSE={winner['normalized_mse']:.4f}",
              flush=True)
    result = {
        "role": "post-hoc global FD joint time and nonnegative gain diagnostic",
        "protocol_sha256": digest(PROTOCOL), "baseline_sha256": digest(BASELINE),
        "fd_config_sha256": digest(cfg_path), "solver_sha256": digest(ROOT / "fd_waveform.py"),
        "candidate_count": len(candidates), "event_count": len(rows), "device": str(device),
        "template_runtime_seconds": fd_report["runtime_seconds"],
        "peak_cuda_memory_mib": (fd_report["peak_cuda_memory_bytes"] / 2**20
                                 if fd_report["peak_cuda_memory_bytes"] else None),
        "events": rows,
        "summary": {
            "median_location_error_m": float(np.median([r["location_error_m"] for r in rows])),
            "median_origin_time_error_s": float(np.median([r["origin_time_error_s"] for r in rows])),
            "median_relative_l2": float(np.median([r["relative_l2"] for r in rows])),
            "median_ranking_seconds": float(np.median([r["ranking_seconds"] for r in rows])),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Post-hoc global FD joint time/amplitude audit", "",
             "Same 75 grid points and eight confirmation records as the registered FD "
             "unit-amplitude baseline. Each candidate jointly optimizes a discrete "
             "origin-time shift and a nonnegative scalar source gain. Source truth is "
             "used only to calculate errors after ranking.", "",
             "| Event | Unit eloc (m) | Joint eloc (m) | Unit et (s) | Joint et (s) | "
             "Unit MSE | Joint MSE | Gain |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        old = row["unit_amplitude_baseline"]
        lines.append(f"| {row['id']} | {old['location_error_m']:.1f} | "
                     f"{row['location_error_m']:.1f} | {old['origin_time_error_s']:.3f} | "
                     f"{row['origin_time_error_s']:.3f} | {old['normalized_mse']:.4f} | "
                     f"{row['normalized_mse']:.4f} | {row['nonnegative_gain']:.4f} |")
    lines.extend(["", f"Joint median eloc = {result['summary']['median_location_error_m']:.1f} m; "
                  f"median et = {result['summary']['median_origin_time_error_s']:.3f} s; "
                  f"median relative L2 = {result['summary']['median_relative_l2']:.4f}.",
                  f"Templates: {fd_report['runtime_seconds']:.1f} s; median per-event "
                  f"ranking: {result['summary']['median_ranking_seconds']:.3f} s.", "",
                  "This objective differs from the original baseline. Lower waveform MSE "
                  "does not guarantee lower location error; tiny fitted gains can favor "
                  "nearly silent predictions. This post-hoc change does not establish an "
                  "accuracy-matched comparison with a neural workflow.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {RESULT} and {REPORT}", flush=True)


if __name__ == "__main__":
    main()
