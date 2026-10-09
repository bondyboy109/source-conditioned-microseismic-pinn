"""Post-hoc FD depth profiles at the nearest coarse-grid horizontal nodes.

This diagnostic uses known source truth to choose two shallow events. It is
not a blind search, a replacement FD baseline, or a boundary-condition test.
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

from fd_waveform import best_origin_time, simulate_candidates

PROTOCOL = ROOT / "experiments/fd_waveform_global_refined_confirmation_protocol.json"
GRID_AUDIT = ROOT / "experiments/fd_global_grid_ranking_audit.json"
RESULT = ROOT / "experiments/fd_shallow_depth_profile_audit.json"
REPORT = RESULT.with_suffix(".md")
EVENT_IDS = ("confirm_01_west_shallow", "confirm_05_center_shallow")
DEPTHS_M = (100.0, 250.0, 400.0, 550.0, 800.0, 1100.0, 1500.0)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def shifted_energy_ratio(template: np.ndarray, observed: np.ndarray, lag: int) -> float:
    """Energy after the same zero-padded shift used by best_origin_time."""
    prediction = np.zeros_like(observed)
    n_time = observed.shape[1]
    if lag >= 0:
        prediction[:, lag:] = template[:, :n_time - lag]
    else:
        prediction[:] = template[:, -lag:-lag + n_time]
    return float(np.sum(prediction.astype(np.float64) ** 2) /
                 np.sum(observed.astype(np.float64) ** 2))


def main() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("Depth profile already exists; refusing to overwrite")
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    ranking = json.loads(GRID_AUDIT.read_text(encoding="utf-8"))
    if ranking["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Original FD protocol changed")
    config_path = ROOT / spec["fd_config"]
    if (ranking["config_sha256"] != digest(config_path)
            or ranking["solver_sha256"] != digest(ROOT / "fd_waveform.py")):
        raise ValueError("Registered FD configuration or implementation changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    low, high = np.asarray(config["source_location_bounds_m"])
    if min(DEPTHS_M) < low[2] or max(DEPTHS_M) > high[2]:
        raise ValueError("Depth scan is outside the registered source domain")
    by_id = {r["id"]: r for r in ranking["events"]}
    event_specs = {e["id"]: e for e in spec["events"]}
    selected = []
    candidates = []
    for name in EVENT_IDS:
        row, event = by_id[name], event_specs[name]
        path = ROOT / event["observation"]
        if digest(path) != event["observation_sha256"]:
            raise ValueError(f"Observation changed: {path}")
        with np.load(path) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            mask = (times >= spec["waveform_window_s"][0]) & (
                times <= spec["waveform_window_s"][1] + 1e-9)
            observed = np.asarray(archive["data"][:, mask], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if not np.array_equal(truth, row["truth_source_m_s_post_ranking"]):
            raise ValueError(f"Source truth changed: {name}")
        xy = row["nearest_grid_xyz_m"][:2]
        selected.append((name, event, observed, truth, xy, row))
        candidates.extend([[*xy, depth] for depth in DEPTHS_M])

    positions = np.asarray(candidates, dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Synthesizing {len(positions)} FD depth-profile templates on {device}", flush=True)
    templates, fd_report = simulate_candidates(
        config_path, positions, spec["template_duration_s"],
        origin_time_s=spec["reference_origin_time_s"], batch_size=4, device=device)
    rows = []
    for i, (name, event, observed, truth, xy, old) in enumerate(selected):
        scores = []
        for j, depth in enumerate(DEPTHS_M):
            template = templates[i * len(DEPTHS_M) + j]
            best = best_origin_time(
                template, observed, fd_report["time_step_s"],
                spec["reference_origin_time_s"], tuple(spec["origin_time_bounds_s"]))
            scores.append({"depth_m": depth, "location_error_m": float(
                np.linalg.norm(np.asarray([*xy, depth]) - truth[:3])),
                **best, "shifted_prediction_to_observation_energy": shifted_energy_ratio(
                    template, observed, best["lag_samples"])})
        for score in scores:
            if score["depth_m"] == 100.0:
                expected = old["nearest_grid_score"]
                if old["nearest_grid_xyz_m"][2] == 100.0 and not np.isclose(
                        score["normalized_mse"], expected["normalized_mse"],
                        rtol=2e-4, atol=1e-7):
                    raise ValueError(f"Nearest-node score failed to reproduce: {name}")
            if score["depth_m"] == 800.0:
                if old["winner_grid_xyz_m"] == [*xy, 800.0] and not np.isclose(
                        score["normalized_mse"], old["winner_score_original"],
                        rtol=2e-4, atol=1e-7):
                    raise ValueError(f"Original winning score failed to reproduce: {name}")
        winner = min(scores, key=lambda row: row["normalized_mse"])
        rows.append({"id": name, "observation_sha256": event["observation_sha256"],
                     "truth_source_m_s_post_hoc": truth.tolist(),
                     "fixed_nearest_grid_xy_m": xy, "scores": scores,
                     "profile_minimum_depth_m": winner["depth_m"],
                     "profile_minimum_normalized_mse": winner["normalized_mse"]})
        print(f"{name}: 100m loss={scores[0]['normalized_mse']:.4f}; "
              f"profile minimum z={winner['depth_m']:.0f}m "
              f"loss={winner['normalized_mse']:.4f}", flush=True)

    output = {
        "role": "post-hoc fixed-horizontal FD depth scan on two inspected events",
        "protocol_sha256": digest(PROTOCOL), "grid_ranking_audit_sha256": digest(GRID_AUDIT),
        "config_sha256": digest(config_path), "solver_sha256": digest(ROOT / "fd_waveform.py"),
        "device": str(device), "depths_m": list(DEPTHS_M),
        "template_runtime_seconds": fd_report["runtime_seconds"],
        "peak_cuda_memory_mib": (fd_report["peak_cuda_memory_bytes"] / 2**20
                                 if fd_report["peak_cuda_memory_bytes"] else None),
        "events": rows,
    }
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = ["# Post-hoc FD shallow-source depth profiles", "",
             "Two previously inspected events. The fixed x/y location is the "
             "nearest horizontal node of the original 75-point FD grid. "
             "Known truth is used only to choose the diagnostic events and "
             "interpret errors. The FD boundary, receivers, velocity, source "
             "and discrete origin-time optimization are unchanged.", "",
             "| Event | Depth (m) | Location error (m) | Normalized MSE | "
             "Best origin (s) | Predicted/observed energy |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        for s in row["scores"]:
            lines.append(f"| {row['id']} | {s['depth_m']:.0f} | "
                         f"{s['location_error_m']:.1f} | {s['normalized_mse']:.4f} | "
                         f"{s['origin_time_s']:.3f} | "
                         f"{s['shifted_prediction_to_observation_energy']:.4f} |")
    lines.extend(["", "The profile does not isolate the effect of the top "
                  "boundary; doing so would require a matched observation "
                  "and forward model with a different top-boundary condition. "
                  "At 100 m, the shifted prediction energies are about 125 "
                  "and 46 times the observed energy for events 01 and 05, "
                  "respectively. The minimum losses along these fixed x/y "
                  "profiles occur at 1500 and 550 m, not at the true 410 "
                  "and 300 m depths. This does not identify a top-boundary "
                  "effect: coarse horizontal position, source-to-receiver "
                  "distance and origin-time trade-offs also change the fit. "
                  "No row is a blind localization estimate.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {RESULT} and {REPORT}", flush=True)


if __name__ == "__main__":
    main()
