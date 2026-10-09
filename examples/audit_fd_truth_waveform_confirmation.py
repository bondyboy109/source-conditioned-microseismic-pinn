"""Audit the FD waveform objective at known sources on locked observations.

This is a post-hoc forward-consistency diagnostic, not a localization method.
The source truths are never added to the candidate grids in the FD baselines.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fd_waveform import best_origin_time, simulate_candidates

PROTOCOL = ROOT / "experiments/fd_waveform_global_refined_confirmation_protocol.json"
RESULT = ROOT / "experiments/fd_truth_waveform_confirmation_audit.json"
REPORT = RESULT.with_suffix(".md")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_errors(predicted: np.ndarray, observed: np.ndarray) -> dict:
    a = np.asarray(predicted, dtype=np.float64)
    b = np.asarray(observed, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Trace shape mismatch: {a.shape} versus {b.shape}")
    energy = float(np.sum(b * b))
    if energy <= 0:
        raise ValueError("Observed waveform has zero energy")
    residual = float(np.sum((a - b) ** 2))
    return {"relative_l2": math.sqrt(residual / energy),
            "normalized_mse": residual / energy,
            "maximum_absolute_difference": float(np.max(np.abs(a - b)))}


def main() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("Audit already exists; refusing to overwrite")
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    config_path = ROOT / spec["fd_config"]
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    dt = float(cfg["time_step_s"])
    window = spec["waveform_window_s"]
    if window != [0.0, 2.0] or spec["reference_origin_time_s"] != 0.5:
        raise ValueError("Unexpected frozen baseline window or reference time")

    observations = []
    for event in spec["events"]:
        path = ROOT / event["observation"]
        if digest(path) != event["observation_sha256"]:
            raise ValueError(f"Observation changed: {path}")
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (metadata["boundary"] != "zero Dirichlet on all six faces"
                or metadata["frequency_hz"] != 5.0
                or metadata["sponge_width_m"] != 0.0):
            raise ValueError(f"Observation FD setup differs: {path}")
        with np.load(path) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            mask = (times >= window[0]) & (times <= window[1] + 1e-9)
            record = np.asarray(archive["data"][:, mask], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"])
        if (record.shape != (cfg["receiver_count"], int(round(window[1] / dt)) + 1)
                or not np.allclose(times[mask], np.arange(record.shape[1]) * dt,
                                   atol=2e-7, rtol=0)
                or not np.array_equal(receivers, np.load(cfg["receivers_file"]))):
            raise ValueError(f"Observation sampling or receiver geometry differs: {path}")
        observations.append((event, record, truth))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positions = np.stack([truth[:3] for _, _, truth in observations])
    templates, template_report = simulate_candidates(
        config_path, positions, spec["template_duration_s"],
        origin_time_s=spec["reference_origin_time_s"],
        batch_size=min(4, len(positions)), device=device)

    rows = []
    for (event, observed, truth), template in zip(observations, templates):
        direct, direct_report = simulate_candidates(
            config_path, truth[None, :3], window[1],
            origin_time_s=float(truth[3]), batch_size=1, device=device)
        direct_error = relative_errors(direct[0], observed)
        nearest_lag = round((truth[3] - spec["reference_origin_time_s"]) / dt)
        nearest_origin = spec["reference_origin_time_s"] + nearest_lag * dt
        fixed = best_origin_time(template, observed, dt,
                                 spec["reference_origin_time_s"],
                                 (nearest_origin, nearest_origin))
        optimized = best_origin_time(template, observed, dt,
                                     spec["reference_origin_time_s"],
                                     tuple(spec["origin_time_bounds_s"]))
        row = {"id": event["id"], "observation_sha256": event["observation_sha256"],
               "source_truth_m_s": truth.tolist(),
               "direct_truth": direct_error,
               "shifted_at_nearest_truth_time": fixed,
               "shifted_optimized_at_truth_location": optimized,
               "optimized_origin_error_s": abs(optimized["origin_time_s"] - truth[3]),
               "direct_runtime_seconds": direct_report["runtime_seconds"]}
        rows.append(row)
        print(f"{event['id']}: direct L2={direct_error['relative_l2']:.3e}, "
              f"shifted truth L2={fixed['relative_l2']:.3e}, "
              f"optimized L2={optimized['relative_l2']:.3e}, "
              f"et={row['optimized_origin_error_s']:.4f} s", flush=True)

    result = {
        "role": "post-hoc forward-consistency and time-shift diagnostic; not localization",
        "baseline_protocol_sha256": digest(PROTOCOL),
        "config_sha256": digest(config_path),
        "candidate_solver_sha256": digest(ROOT / "fd_waveform.py"),
        "observation_solver_sha256": digest(ROOT / "examples/generate_fd_observation.py"),
        "device": str(device), "time_step_s": dt, "waveform_window_s": window,
        "template_runtime_seconds": template_report["runtime_seconds"],
        "direct_total_runtime_seconds": sum(r["direct_runtime_seconds"] for r in rows),
        "events": rows,
        "summary": {
            "median_direct_truth_relative_l2": float(np.median([
                r["direct_truth"]["relative_l2"] for r in rows])),
            "median_shifted_truth_relative_l2": float(np.median([
                r["shifted_at_nearest_truth_time"]["relative_l2"] for r in rows])),
            "median_shifted_optimized_relative_l2": float(np.median([
                r["shifted_optimized_at_truth_location"]["relative_l2"] for r in rows])),
            "median_optimized_origin_error_s": float(np.median([
                r["optimized_origin_error_s"] for r in rows])),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# FD true-source forward consistency audit", "",
             "Post-hoc diagnostic on the eight frozen records. The truths were not "
             "used in either global FD candidate grid or event ranking.", "",
             "| Event | Direct truth relative L2 | Shift at truth time relative L2 | "
             "Best shift relative L2 | Best shift origin error (s) |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['direct_truth']['relative_l2']:.3e} | "
                     f"{r['shifted_at_nearest_truth_time']['relative_l2']:.3e} | "
                     f"{r['shifted_optimized_at_truth_location']['relative_l2']:.3e} | "
                     f"{r['optimized_origin_error_s']:.3e} |")
    lines.extend(["", "This tests solver agreement and the approximate discrete "
                  "time-shift interface at known locations. It is not an estimate "
                  "of location accuracy or an independent FD inverse benchmark.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {RESULT} and {REPORT}", flush=True)


if __name__ == "__main__":
    main()
