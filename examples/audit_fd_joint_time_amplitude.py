"""Post-hoc joint origin-time/nonnegative-amplitude FD depth diagnostic.

The inspected events and truth-selected horizontal nodes are not a blind
localization benchmark. Original unit-amplitude results remain unchanged.
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

from fd_waveform import best_origin_time, simulate_candidates

PROTOCOL = ROOT / "experiments/fd_waveform_global_refined_confirmation_protocol.json"
PROFILE = ROOT / "experiments/fd_shallow_depth_profile_audit.json"
FIXED = ROOT / "experiments/fd_amplitude_fixed_origin_audit.json"
RESULT = ROOT / "experiments/fd_joint_time_amplitude_depth_audit.json"
REPORT = RESULT.with_suffix(".md")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score_joint(template: np.ndarray, observed: np.ndarray, dt: float,
                reference_origin_s: float, bounds_s: tuple[float, float]) -> dict:
    """Minimize ||a*shift(template, lag)-observed||^2 / ||observed||^2.

    Search integer lags within the same bounds and zero-padding convention as
    fd_waveform.best_origin_time. The nonnegative amplitude is analytic.
    """
    template = np.asarray(template, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if (template.ndim != 2 or observed.ndim != 2
            or template.shape[0] != observed.shape[0]
            or template.shape[1] < observed.shape[1] or dt <= 0):
        raise ValueError("Invalid template, observation or sampling interval")
    energy = float(np.sum(observed ** 2))
    if energy <= 0:
        raise ValueError("Observed waveform has zero energy")
    first = int(math.ceil((bounds_s[0] - reference_origin_s) / dt - 1e-10))
    last = int(math.floor((bounds_s[1] - reference_origin_s) / dt + 1e-10))
    if first > last:
        raise ValueError("Invalid origin-time bounds")
    n = observed.shape[1]
    best = None
    for lag in range(first, last + 1):
        if lag >= 0:
            usable = n - lag
            if usable <= 0:
                continue
            pred = template[:, :usable]
            obs = observed[:, lag:]
        else:
            start = -lag
            stop = start + n
            if stop > template.shape[1]:
                continue
            pred = template[:, start:stop]
            obs = observed
        pred_energy = float(np.sum(pred ** 2))
        inner = float(np.sum(pred * obs))
        gain = max(0.0, inner / pred_energy) if pred_energy > 0 else 0.0
        loss = max(0.0, 1.0 - 2 * gain * inner / energy
                   + gain * gain * pred_energy / energy)
        if best is None or loss < best["normalized_mse"]:
            best = {"origin_time_s": reference_origin_s + lag * dt,
                    "lag_samples": lag, "nonnegative_gain": gain,
                    "normalized_mse": loss, "relative_l2": math.sqrt(loss),
                    "shifted_prediction_to_observation_energy": pred_energy / energy,
                    "normalized_inner_product": inner / energy}
    if best is None:
        raise ValueError("No valid origin-time shift")
    return best


def main() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("Joint audit already exists; refusing to overwrite")
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    fixed = json.loads(FIXED.read_text(encoding="utf-8"))
    cfg_path = ROOT / spec["fd_config"]
    if (profile["protocol_sha256"] != digest(PROTOCOL)
            or profile["config_sha256"] != digest(cfg_path)
            or profile["solver_sha256"] != digest(ROOT / "fd_waveform.py")
            or fixed["input_sha256"] != digest(PROFILE)):
        raise ValueError("FD protocol or saved diagnostic changed")
    events = {e["id"]: e for e in spec["events"]}
    fixed_events = {e["id"]: e for e in fixed["events"]}
    positions, records = [], []
    for entry in profile["events"]:
        event = events[entry["id"]]
        path = ROOT / event["observation"]
        if digest(path) != event["observation_sha256"]:
            raise ValueError(f"Observation changed: {path}")
        with np.load(path) as archive:
            t = np.asarray(archive["times"], dtype=np.float64)
            mask = ((t >= spec["waveform_window_s"][0])
                    & (t <= spec["waveform_window_s"][1] + 1e-9))
            observed = np.asarray(archive["data"][:, mask], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if not np.array_equal(truth, entry["truth_source_m_s_post_hoc"]):
            raise ValueError(f"Truth changed: {entry['id']}")
        xy = entry["fixed_nearest_grid_xy_m"]
        depths = [s["depth_m"] for s in entry["scores"]]
        if depths != profile["depths_m"]:
            raise ValueError("Profile depth sequence changed")
        positions.extend([[*xy, z] for z in depths])
        records.append((entry, observed, truth, fixed_events[entry["id"]]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Synthesizing {len(positions)} depth templates on {device}", flush=True)
    templates, fd_report = simulate_candidates(
        cfg_path, np.asarray(positions, dtype=np.float64),
        spec["template_duration_s"], spec["reference_origin_time_s"],
        batch_size=4, device=device)
    rows = []
    scoring_seconds = 0.0
    for i, (entry, observed, truth, fixed_entry) in enumerate(records):
        scores = []
        for j, (prior, frozen) in enumerate(zip(entry["scores"], fixed_entry["scores"])):
            template = templates[i * len(profile["depths_m"]) + j]
            old = best_origin_time(template, observed, fd_report["time_step_s"],
                                   spec["reference_origin_time_s"],
                                   tuple(spec["origin_time_bounds_s"]))
            if (not np.isclose(old["normalized_mse"], prior["normalized_mse"],
                               rtol=2e-4, atol=1e-7)
                    or not np.isclose(old["origin_time_s"], prior["origin_time_s"])):
                raise ValueError(f"Unit-amplitude score differs: {entry['id']}, {j}")
            started = time.perf_counter()
            joint = score_joint(template, observed, fd_report["time_step_s"],
                                spec["reference_origin_time_s"],
                                tuple(spec["origin_time_bounds_s"]))
            scoring_seconds += time.perf_counter() - started
            if joint["normalized_mse"] > (
                    frozen["gain_adjusted_normalized_mse_at_fixed_origin"] + 1e-8):
                raise ValueError(f"Joint optimum worse than fixed shift: {entry['id']}, {j}")
            scores.append({"depth_m": prior["depth_m"],
                           "location_error_m_post_hoc": prior["location_error_m"],
                           "unit_amplitude_score": old,
                           "gain_at_unit_amplitude_time": frozen[
                               "best_nonnegative_gain_at_fixed_origin"],
                           "fixed_time_gain_adjusted_mse": frozen[
                               "gain_adjusted_normalized_mse_at_fixed_origin"],
                           "joint_score": joint})
        winner = min(scores, key=lambda s: s["joint_score"]["normalized_mse"])
        rows.append({"id": entry["id"],
                     "observation_sha256": events[entry["id"]]["observation_sha256"],
                     "truth_source_m_s_post_hoc": truth.tolist(),
                     "fixed_nearest_grid_xy_m": entry["fixed_nearest_grid_xy_m"],
                     "scores": scores,
                     "joint_profile_minimum_depth_m": winner["depth_m"],
                     "joint_profile_minimum_mse": winner["joint_score"]["normalized_mse"]})
        print(f"{entry['id']}: joint minimum depth={winner['depth_m']:.0f} m, "
              f"time={winner['joint_score']['origin_time_s']:.3f} s, "
              f"gain={winner['joint_score']['nonnegative_gain']:.4f}, "
              f"MSE={winner['joint_score']['normalized_mse']:.4f}", flush=True)
    result = {"role": "post-hoc joint discrete time/nonnegative gain diagnostic "
                      "on two previously inspected fixed-horizontal depth profiles",
              "protocol_sha256": digest(PROTOCOL), "profile_sha256": digest(PROFILE),
              "fixed_gain_sha256": digest(FIXED), "config_sha256": digest(cfg_path),
              "solver_sha256": digest(ROOT / "fd_waveform.py"),
              "device": str(device), "depths_m": profile["depths_m"],
              "template_runtime_seconds": fd_report["runtime_seconds"],
              "scoring_seconds": scoring_seconds, "events": rows}
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# FD joint origin-time and amplitude depth diagnostic", "",
             "Two already inspected events at truth-selected coarse horizontal "
             "nodes. Each depth uses the original FD template and discrete "
             "time bounds. At every shift, an independent nonnegative scalar "
             "gain minimizes normalized MSE; the best shift and gain are "
             "selected jointly. Original unit-gain results are unchanged.", "",
             "| Event | Depth (m) | Unit MSE | Fixed-time gain MSE | "
             "Joint MSE | Joint origin (s) | Joint gain |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        for s in row["scores"]:
            j = s["joint_score"]
            lines.append(f"| {row['id']} | {s['depth_m']:.0f} | "
                         f"{s['unit_amplitude_score']['normalized_mse']:.4f} | "
                         f"{s['fixed_time_gain_adjusted_mse']:.4f} | "
                         f"{j['normalized_mse']:.4f} | "
                         f"{j['origin_time_s']:.3f} | {j['nonnegative_gain']:.4f} |")
    lines.extend(["", "Gain zero gives MSE one (zero waveform); such a "
                  "candidate is not a meaningful successful fit. This "
                  "diagnostic changes the objective and uses inspected "
                  "events. It is neither a blind localization result nor an "
                  "accuracy-matched comparison with the neural workflow. "
                  "A global 75-node gain-profiled search would require "
                  "rescoring every node using the same rule.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {RESULT} and {REPORT}", flush=True)


if __name__ == "__main__":
    main()
