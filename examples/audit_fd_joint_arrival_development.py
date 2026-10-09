"""Compare FD waveform/gain fitting with FMM arrival regularization on development data."""

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
from examples.run_fd_waveform_refinement_development import candidate_grid
from examples.run_traveltime_aligned_moe_localization import pick_arrivals
from fd_waveform import simulate_candidates
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator

PROTOCOL = ROOT / "experiments/fd_joint_arrival_development_protocol.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score_all_weights(template: np.ndarray, observed: np.ndarray, dt: float,
                      reference_s: float, bounds_s: tuple[float, float],
                      travel_s: np.ndarray, picks_s: np.ndarray,
                      scale_s: float, weights: list[float]) -> dict:
    template = np.asarray(template, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    valid = np.isfinite(picks_s)
    if (template.ndim != 2 or observed.ndim != 2
            or template.shape[0] != observed.shape[0]
            or template.shape[1] < observed.shape[1]
            or travel_s.shape != (len(observed),)
            or picks_s.shape != travel_s.shape
            or not valid.any() or scale_s <= 0 or dt <= 0
            or any(weight < 0 for weight in weights)):
        raise ValueError("Invalid waveform, arrival, or objective inputs")
    energy = float(np.sum(observed ** 2))
    if energy <= 0:
        raise ValueError("Observed waveform has zero energy")
    first = int(math.ceil((bounds_s[0] - reference_s) / dt - 1e-10))
    last = int(math.floor((bounds_s[1] - reference_s) / dt + 1e-10))
    if first > last:
        raise ValueError("Invalid origin-time bounds")
    best = {}
    n = observed.shape[1]
    for lag in range(first, last + 1):
        if lag >= 0:
            if lag >= n:
                continue
            pred, obs = template[:, :n - lag], observed[:, lag:]
        else:
            start = -lag
            if start + n > template.shape[1]:
                continue
            pred, obs = template[:, start:start + n], observed
        pred_energy = float(np.sum(pred ** 2))
        inner = float(np.sum(pred * obs))
        gain = max(0.0, inner / pred_energy) if pred_energy > 0 else 0.0
        waveform = max(0.0, 1 - 2 * gain * inner / energy
                       + gain * gain * pred_energy / energy)
        origin = reference_s + lag * dt
        arrival = float(np.mean(((origin + travel_s[valid] - picks_s[valid])
                                 / scale_s) ** 2))
        for weight in weights:
            total = waveform + weight * arrival
            if weight not in best or total < best[weight]["total_loss"]:
                best[weight] = {"origin_time_s": origin, "lag_samples": lag,
                                "nonnegative_gain": gain,
                                "waveform_normalized_mse": waveform,
                                "arrival_normalized_mse": arrival,
                                "total_loss": total}
    if len(best) != len(set(weights)):
        raise ValueError("No valid origin time shifts")
    return best


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path, report_path = ROOT / spec["result"], ROOT / spec["report"]
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Development result already exists")
    base = json.loads((ROOT / spec["base_protocol"]).read_text(encoding="utf-8"))
    previous = json.loads((ROOT / spec["base_result"]).read_text(encoding="utf-8"))
    picker = json.loads((ROOT / spec["picker_protocol"]).read_text(encoding="utf-8"))[
        "arrival_constraint"]
    if (previous["protocol_sha256"] != digest(ROOT / spec["base_protocol"])
            or spec["arrival_scale_s"] != picker["arrival_scale_s"]
            or spec["minimum_valid_picks"] != picker["minimum_valid_picks"]):
        raise ValueError("Frozen base result or picker protocol differs")
    config = json.loads((ROOT / base["fd_config"]).read_text(encoding="utf-8"))
    if digest(ROOT / base["fd_config"]) != base["fd_config_sha256"]:
        raise ValueError("FD config changed")
    receivers = np.load(config["receivers_file"])
    fields_path = ROOT / spec["fmm_fields"]
    fields = np.load(fields_path, mmap_mode="r")
    if fields.shape[0] != len(receivers):
        raise ValueError("FMM field and receiver geometry differ")
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)), spec["fmm_grid_spacing_m"])
    spatial_bounds = np.asarray(config["source_location_bounds_m"]).T
    weights = [float(a) for a in spec["arrival_weights"]]
    if not weights or weights[0] != 0 or len(set(weights)) != len(weights):
        raise ValueError("Weights must be distinct and start at zero")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for event, old in zip(base["events"], previous["events"]):
        if event["id"] != old["id"] or digest(ROOT / event["observation"]) != event["observation_sha256"]:
            raise ValueError("Development observation or order changed")
        with np.load(ROOT / event["observation"]) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            full_data = np.asarray(archive["data"], dtype=np.float32)
            event_receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if not np.array_equal(event_receivers, receivers):
            raise ValueError("Receiver geometry changed")
        _, corrected = pick_arrivals({"data": full_data, "times": times}, picker)
        valid_count = int(np.isfinite(corrected).sum())
        if valid_count < spec["minimum_valid_picks"]:
            raise ValueError(f"Too few arrival picks for {event['id']}: {valid_count}")
        mask = ((times >= base["waveform_window_s"][0])
                & (times <= base["waveform_window_s"][1] + 1e-9))
        observed = full_data[:, mask]
        candidates = candidate_grid(
            np.asarray(old["fmm"]["estimated_source_m_s"][:3]),
            base["candidate_offsets_m"], spatial_bounds)
        if not np.allclose(candidates, old["candidate_source_xyz_m"]):
            raise ValueError("Candidate grid changed")
        with torch.inference_mode():
            points = torch.as_tensor(np.repeat(candidates, len(receivers), axis=0),
                                     dtype=torch.float32)
            ids = torch.arange(len(receivers)).repeat(len(candidates))
            travel = interpolator(points, ids).reshape(len(candidates), -1).numpy()
        started = time.perf_counter()
        templates, fd_report = simulate_candidates(
            ROOT / base["fd_config"], candidates,
            duration_s=(base["waveform_window_s"][1]
                        + base["reference_origin_time_s"]
                        - base["origin_time_bounds_s"][0]),
            origin_time_s=base["reference_origin_time_s"],
            batch_size=base["batch_size"], device=device)
        scores = []
        for template, travel_s in zip(templates, travel):
            scores.append(score_all_weights(
                template, observed, fd_report["time_step_s"],
                base["reference_origin_time_s"],
                tuple(base["origin_time_bounds_s"]), travel_s, corrected,
                spec["arrival_scale_s"], weights))
        # Check the unregularized branch against the established implementation.
        for template, item in zip(templates, scores):
            original = score_joint(template, observed, fd_report["time_step_s"],
                                   base["reference_origin_time_s"],
                                   tuple(base["origin_time_bounds_s"]))
            if (not np.isclose(item[0.0]["waveform_normalized_mse"],
                               original["normalized_mse"], atol=1e-8)
                    or not np.isclose(item[0.0]["origin_time_s"],
                                      original["origin_time_s"], atol=1e-8)):
                raise ValueError("Zero-weight joint score disagrees with baseline")
        selected = {}
        for weight in weights:
            index = min(range(len(scores)), key=lambda i: scores[i][weight]["total_loss"])
            estimate = [*(float(value) for value in candidates[index]),
                        scores[index][weight]["origin_time_s"]]
            selected[str(weight)] = {"candidate_index": index,
                                     "estimated_source_m_s": estimate,
                                     "location_error_m": float(np.linalg.norm(
                                         candidates[index] - truth[:3])),
                                     "origin_time_error_s": abs(estimate[3] - truth[3]),
                                     "at_origin_bound": bool(np.isclose(
                                         estimate[3], base["origin_time_bounds_s"]).any()),
                                     **scores[index][weight]}
        rows.append({"id": event["id"], "truth_source_m_s": truth.tolist(),
                     "valid_picks": valid_count,
                     "candidate_count": len(candidates), "selection": selected,
                     "fd_runtime_seconds": fd_report["runtime_seconds"],
                     "total_runtime_seconds": time.perf_counter() - started})
        print(f"{event['id']}: " + ", ".join(
            f"alpha={a:g}, eloc={selected[str(a)]['location_error_m']:.1f} m, "
            f"et={selected[str(a)]['origin_time_error_s']:.3f} s"
            for a in weights), flush=True)
    summary = {}
    for weight in weights:
        chosen = [row["selection"][str(weight)] for row in rows]
        summary[str(weight)] = {
            "median_location_error_m": float(np.median([x["location_error_m"] for x in chosen])),
            "median_origin_time_error_s": float(np.median([x["origin_time_error_s"] for x in chosen])),
            "origin_bound_count": sum(x["at_origin_bound"] for x in chosen),
            "median_waveform_normalized_mse": float(np.median([
                x["waveform_normalized_mse"] for x in chosen])),
            "median_gain": float(np.median([x["nonnegative_gain"] for x in chosen])),
        }
    result = {"role": spec["role"], "protocol_sha256": digest(PROTOCOL),
              "input_sha256": {path: digest(ROOT / path) for path in (
                  spec["base_protocol"], spec["base_result"], spec["fmm_fields"],
                  spec["picker_protocol"])}, "device": str(device),
              "picker_bias_s": picker["pick_bias_s"], "arrival_scale_s": spec["arrival_scale_s"],
              "arrival_weights": weights, "events": rows, "summary": summary,
              "limitations": spec["interpretation"]}
    lines = ["# FMM arrival constraint on local FD joint fits", "",
             spec["interpretation"], "",
             "| Weight | Median eloc (m) | Median et (s) | Bound hits | Median waveform MSE | Median gain |",
             "|---:|---:|---:|---:|---:|---:|"]
    for weight in weights:
        s = summary[str(weight)]
        lines.append(f"| {weight:g} | {s['median_location_error_m']:.1f} | "
                     f"{s['median_origin_time_error_s']:.4f} | {s['origin_bound_count']} | "
                     f"{s['median_waveform_normalized_mse']:.4f} | {s['median_gain']:.4f} |")
    lines.extend(["", "| Event | Weight | Estimate (x,y,z,t) | eloc (m) | et (s) | Bound | Waveform MSE | Arrival loss | Gain |",
                  "|---|---:|---|---:|---:|---|---:|---:|---:|"])
    for row in rows:
        for weight in weights:
            s = row["selection"][str(weight)]
            lines.append(f"| {row['id']} | {weight:g} | "
                         f"{tuple(round(v, 3) for v in s['estimated_source_m_s'])} | "
                         f"{s['location_error_m']:.1f} | {s['origin_time_error_s']:.4f} | "
                         f"{s['at_origin_bound']} | {s['waveform_normalized_mse']:.4f} | "
                         f"{s['arrival_normalized_mse']:.4f} | {s['nonnegative_gain']:.4f} |")
    lines.extend(["", "The spatial grid and source observations were already inspected. "
                  "The 0-weight row fits time and a nonnegative scalar gain using FD waveforms; "
                  "all other weights add corrected receiver picks from the frozen FMM fields. "
                  "FD time shifting and the FMM arrival residual use the same trial origin time. "
                  "The tabulated waveform MSE excludes the arrival penalty; total loss includes it.", "",
                  "On these two previously inspected and already well-localized events, "
                  "arrival regularization did not reduce the location error. Weights 0.1 and 0.25 "
                  "increased median origin-time error, and weight 1 worsened both errors and "
                  "waveform fit. There were no bound hits even at zero weight, so this small "
                  "local pilot does not test whether the penalty prevents the bound failures "
                  "previously seen in the separate 75-node fresh-source check. Do not treat "
                  "this weight sweep as a new independent confirmation or transfer its "
                  "favorable local-grid errors to a global search.", ""])
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {result_path} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
