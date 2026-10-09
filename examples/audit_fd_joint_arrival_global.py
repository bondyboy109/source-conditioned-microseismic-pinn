"""Post-hoc FMM arrival constraint audit on previously inspected FD grid events."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.audit_fd_joint_arrival_development import score_all_weights
from examples.run_fd_waveform_global_refined_confirmation import global_grid, observed_record
from examples.run_traveltime_aligned_moe_localization import pick_arrivals
from fd_waveform import simulate_candidates
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator

PROTOCOL = ROOT / "experiments/fd_joint_arrival_global_audit_protocol.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path, report_path = ROOT / spec["result"], ROOT / spec["report"]
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Audit output exists; refusing to overwrite")
    base_path = ROOT / spec["base_protocol"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    prior = json.loads((ROOT / spec["base_result"]).read_text(encoding="utf-8"))
    local = json.loads((ROOT / spec["local_protocol"]).read_text(encoding="utf-8"))
    arrival = json.loads((ROOT / spec["picker_protocol"]).read_text(encoding="utf-8"))[
        "arrival_constraint"]
    cfg_path = ROOT / base["fd_config"]
    if (prior["protocol_sha256"] != digest(base_path)
            or prior["fd_config_sha256"] != digest(cfg_path)
            or prior["solver_sha256"] != digest(ROOT / "fd_waveform.py")
            or spec["arrival_weights"] != local["arrival_weights"]
            or spec["arrival_scale_s"] != local["arrival_scale_s"]
            or spec["minimum_valid_picks"] != arrival["minimum_valid_picks"]):
        raise ValueError("Frozen protocol, solver or weights changed")
    fields_path = ROOT / spec["fmm_fields"]
    if digest(fields_path) != json.loads((ROOT / spec["picker_protocol"]).read_text(
            encoding="utf-8"))["receiver_fmm_fields_sha256"]:
        raise ValueError("Receiver FMM field changed")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    receivers = np.load(config["receivers_file"])
    candidates = global_grid(np.asarray(config["source_location_bounds_m"]).T,
                             base["spatial_grid_counts"])
    if len(candidates) != prior["candidate_count"]:
        raise ValueError("Candidate grid count changed")
    fields = np.load(fields_path, mmap_mode="r")
    if fields.shape[0] != len(receivers):
        raise ValueError("FMM receiver count differs")
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)),
        float(json.loads((ROOT / "experiments/traveltime_aligned_moe_development_protocol.json").read_text(
            encoding="utf-8"))["fmm"]["grid_spacing_m"]))
    with torch.inference_mode():
        points = torch.as_tensor(np.repeat(candidates, len(receivers), axis=0),
                                 dtype=torch.float32)
        ids = torch.arange(len(receivers)).repeat(len(candidates))
        travel = interpolator(points, ids).reshape(len(candidates), -1).numpy()
    records = []
    for entry, previous in zip(base["observations"], prior["events"]):
        path = ROOT / base["observation_directory"] / f"{entry['id']}.npz"
        if entry["id"] != previous["id"] or digest(path) != previous["observation_sha256"]:
            raise ValueError("Observation identity differs from earlier check")
        with np.load(path) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            data = np.asarray(archive["data"], dtype=np.float32)
            coords = np.asarray(archive["receiver_coords"], dtype=np.float32)
        if not np.array_equal(coords, receivers):
            raise ValueError("Receiver geometry differs")
        _, picks = pick_arrivals({"data": data, "times": times}, arrival)
        count = int(np.isfinite(picks).sum())
        if count < spec["minimum_valid_picks"]:
            raise ValueError(f"Too few arrivals for {entry['id']}: {count}")
        observed, truth = observed_record(path, tuple(base["waveform_window_s"]))
        if not np.allclose(truth, previous["truth_source_m_s"], atol=1e-5):
            raise ValueError("Truth changed since previous check")
        records.append((entry["id"], previous, observed, truth, picks, count))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Synthesizing {len(candidates)} FD templates on {device}", flush=True)
    templates, fd_report = simulate_candidates(
        cfg_path, candidates, base["template_duration_s"],
        base["reference_origin_time_s"], base["template_batch_size"], device)
    weights = [float(w) for w in spec["arrival_weights"]]
    rows = []
    for name, previous, observed, truth, picks, count in records:
        start = time.perf_counter()
        scores = [score_all_weights(template, observed, fd_report["time_step_s"],
                                    base["reference_origin_time_s"],
                                    tuple(base["origin_time_bounds_s"]), travel[i],
                                    picks, spec["arrival_scale_s"], weights)
                  for i, template in enumerate(templates)]
        selected = {}
        for weight in weights:
            index = min(range(len(scores)),
                        key=lambda i: scores[i][weight]["total_loss"])
            score = scores[index][weight]
            estimate = [*(float(x) for x in candidates[index]), score["origin_time_s"]]
            selected[str(weight)] = {
                "candidate_index": index, "estimated_source_m_s": estimate,
                "location_error_m": float(np.linalg.norm(candidates[index] - truth[:3])),
                "origin_time_error_s": float(abs(estimate[3] - truth[3])),
                "at_origin_bound": bool(np.isclose(estimate[3],
                                                     base["origin_time_bounds_s"]).any()),
                **score,
            }
        control = selected["0.0"]
        old = previous["joint"]
        if (control["candidate_index"] != old["candidate_index"]
                or not np.isclose(control["waveform_normalized_mse"],
                                  old["normalized_mse"], rtol=1e-5, atol=1e-6)
                or not np.isclose(control["origin_time_s"],
                                  old["estimated_source_m_s"][3], atol=1e-6)):
            raise ValueError(f"Baseline replay mismatch: {name}")
        rows.append({"id": name, "truth_source_m_s": truth.tolist(),
                     "valid_picks": count, "scores": selected,
                     "ranking_seconds": time.perf_counter() - start})
        print(f"{name}: " + ", ".join(
            f"alpha={weight:g}, eloc={selected[str(weight)]['location_error_m']:.1f} m, "
            f"et={selected[str(weight)]['origin_time_error_s']:.3f} s, "
            f"bound={selected[str(weight)]['at_origin_bound']}"
            for weight in weights), flush=True)
    summary = {}
    for weight in weights:
        chosen = [row["scores"][str(weight)] for row in rows]
        summary[str(weight)] = {
            "median_location_error_m": float(np.median([s["location_error_m"] for s in chosen])),
            "median_origin_time_error_s": float(np.median([s["origin_time_error_s"] for s in chosen])),
            "origin_bound_count": sum(s["at_origin_bound"] for s in chosen),
            "median_waveform_normalized_mse": float(np.median([
                s["waveform_normalized_mse"] for s in chosen])),
        }
    result = {"role": spec["role"], "protocol_sha256": digest(PROTOCOL),
              "input_sha256": {path: digest(ROOT / path) for path in (
                  spec["base_protocol"], spec["base_result"], spec["local_protocol"],
                  spec["picker_protocol"], spec["fmm_fields"])},
              "candidate_count": len(candidates), "device": str(device),
              "template_runtime_seconds": fd_report["runtime_seconds"],
              "peak_cuda_memory_mib": (fd_report["peak_cuda_memory_bytes"] / 2**20
                                       if fd_report["peak_cuda_memory_bytes"] else None),
              "weights": weights, "picker_bias_s": arrival["pick_bias_s"],
              "arrival_scale_s": spec["arrival_scale_s"], "events": rows,
              "summary": summary, "limitations": spec["limitations"]}
    lines = ["# Post-hoc FMM arrival constraint on 75-node FD search", "",
             spec["limitations"], "",
             "| Weight | Median eloc (m) | Median et (s) | Time-bound hits | Median waveform MSE |",
             "|---:|---:|---:|---:|---:|"]
    for weight in weights:
        s = summary[str(weight)]
        lines.append(f"| {weight:g} | {s['median_location_error_m']:.1f} | "
                     f"{s['median_origin_time_error_s']:.3f} | "
                     f"{s['origin_bound_count']} | "
                     f"{s['median_waveform_normalized_mse']:.4f} |")
    lines.extend(["", "| Event | Weight | Estimate (x,y,z,t) | eloc (m) | et (s) | Bound | Waveform MSE | Arrival loss | Gain |",
                  "|---|---:|---|---:|---:|---|---:|---:|---:|"])
    for row in rows:
        for weight in weights:
            s = row["scores"][str(weight)]
            source = tuple(round(v, 3) for v in s["estimated_source_m_s"])
            lines.append(f"| {row['id']} | {weight:g} | {source} | "
                         f"{s['location_error_m']:.1f} | {s['origin_time_error_s']:.3f} | "
                         f"{s['at_origin_bound']} | {s['waveform_normalized_mse']:.4f} | "
                         f"{s['arrival_normalized_mse']:.4f} | "
                         f"{s['nonnegative_gain']:.4f} |")
    lines.extend(["", "Each score evaluates the same FD template, discrete time shifts, "
                  "and analytic nonnegative gain, adding alpha times mean squared "
                  "corrected FMM pick residual divided by 0.10 s. The alpha=0 "
                  "winner is checked against the earlier saved result. "
                  "Waveform MSE is reported separately from the penalized total. "
                  f"The {len(candidates)} shared FD templates took "
                  f"{fd_report['runtime_seconds']:.1f} s to synthesize.", "",
                  "At weights 0.1 and 0.25, the east-deep event moves off the time "
                  "bound and its time error decreases, while the west-mid event "
                  "remains on a bound and its location error increases. Its fitted "
                  "gain of 0.0083 produces an almost-zero waveform (normalized MSE "
                  "about 1), exposing a degenerate solution that arrival "
                  "regularization alone does not prevent. The third event keeps "
                  "the same location. The lower three-event median is therefore "
                  "not evidence of a robust localization improvement. The 75-node "
                  "grid, source-waveform scale model, pick validity, and FMM/FD "
                  "mismatch need further diagnosis before freezing a new method.", ""])
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {result_path} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
