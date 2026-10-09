"""Compare frozen transfer objectives at true and recovered development sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.diagnose_source_transfer_trajectory import (
    FINAL_OUTPUT, PARENT_PROTOCOL, PROTOCOL as TRAJECTORY_PROTOCOL,
    config, intermediate,
)
from examples.run_source_transfer_moe import digest, new_model
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, load_event, pick_arrivals, sampled_waveforms, waveform_scores,
)

PROTOCOL = ROOT / "experiments/source_transfer_objective_audit_protocol.json"


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    trajectory, parent, setup, loc, _ = config()
    if (ROOT / spec["trajectory_protocol"] != TRAJECTORY_PROTOCOL
            or ROOT / spec["localization_protocol"] !=
            ROOT / parent["localization"]
            or spec["arms"] != trajectory["arms"]
            or spec["seeds"] != trajectory["seeds"]
            or spec["target_updates"] != trajectory["target_updates"]):
        raise ValueError("Audit configuration differs from trajectory")
    result_path = ROOT / spec["output_json"]
    report_path = ROOT / spec["output_report"]
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite objective audit")
    source_path = ROOT / spec["trajectory_result"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (source["protocol_sha256"] != digest(TRAJECTORY_PROTOCOL)
            or source["parent_protocol_sha256"] != digest(PARENT_PROTOCOL)
            or len(source["rows"]) != 288):
        raise ValueError("Trajectory result does not match protocol")
    selected = [row for row in source["rows"] if "localization" in row]
    expected = len(spec["seeds"]) * len(spec["arms"]) * len(spec["target_updates"]) * 4
    if len(selected) != expected:
        raise ValueError("Unexpected selected event count")
    event_names = set(trajectory["localization_events"])
    if {row["event"] for row in selected} != event_names:
        raise ValueError("Selected events differ from frozen trajectory")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    receiver_file = ROOT / "data/observations/receiver_coords.npy"
    receivers = np.load(receiver_file).astype(np.float32)
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    bounds = np.asarray(loc["source_bounds_m_s"], dtype=np.float64)
    search, arrival = loc["search"], loc["arrival_constraint"]
    events = {}
    for name in event_names:
        path = ROOT / name
        observation = load_event(path, receivers)
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if truth.shape != (4,) or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1]):
            raise ValueError(f"True source outside inversion bounds: {path}")
        _, picks = pick_arrivals(observation, arrival)
        if np.isfinite(picks).sum() < arrival["minimum_valid_picks"]:
            raise ValueError(f"Insufficient picks: {path}")
        sampled = sampled_waveforms(observation, domain,
                                    search["refine_receiver_stride"],
                                    search["refine_time_stride"], device)
        events[name] = (observation, truth, picks, sampled)

    rows = []
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            final_path = FINAL_OUTPUT / f"{arm}_seed{seed}.pth"
            final_hash = digest(final_path)
            for step in spec["target_updates"]:
                checkpoint = (final_path if step == parent["phase_updates"]
                              else intermediate(trajectory, arm, seed, step))
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if step == parent["phase_updates"]:
                    valid = (payload["protocol_sha256"] == digest(PARENT_PROTOCOL)
                             and payload["arm"] == arm and payload["seed"] == seed)
                else:
                    valid = (payload["trajectory_protocol_sha256"] == digest(TRAJECTORY_PROTOCOL)
                             and payload["final_checkpoint_sha256"] == final_hash
                             and payload["target_updates"] == step
                             and payload["arm"] == arm and payload["seed"] == seed)
                if not valid:
                    raise ValueError(f"Checkpoint mismatch: {checkpoint}")
                model = new_model(fields, receivers, setup, device)
                model.load_state_dict(payload["state_dict"])
                model.eval().requires_grad_(False)
                matches = [row for row in selected if row["seed"] == seed
                           and row["arm"] == arm and row["target_updates"] == step]
                for original in matches:
                    name = original["event"]
                    observation, truth, picks, sampled = events[name]
                    if digest(observation["path"]) != original["sha256"]:
                        raise ValueError(f"Development event changed: {name}")
                    estimate = np.asarray(original["localization"]["estimate_m_s"],
                                          dtype=np.float64)
                    candidates = np.stack((truth, estimate)).astype(np.float32)
                    wave = waveform_scores(model, candidates, domain, sampled, 2)
                    travel = arrival_scores(model, candidates, picks,
                                            arrival["arrival_scale_s"], 2)
                    hybrid = wave + arrival["hybrid_weight"] * travel
                    rows.append({"seed": seed, "arm": arm, "target_updates": step,
                                 "event": name, "truth_m_s": truth.tolist(),
                                 "estimate_m_s": estimate.tolist(),
                                 "location_error_m": original["localization"]["location_error_m"],
                                 "waveform_at_truth": float(wave[0]),
                                 "waveform_at_estimate": float(wave[1]),
                                 "arrival_at_truth": float(travel[0]),
                                 "arrival_at_estimate": float(travel[1]),
                                 "hybrid_at_truth": float(hybrid[0]),
                                 "hybrid_at_estimate": float(hybrid[1]),
                                 "estimate_minus_truth_hybrid": float(hybrid[1] - hybrid[0])})
                print(f"Audited {arm} seed={seed} target={step}: {len(matches)} events",
                      flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary = {}
    for arm in spec["arms"]:
        for step in spec["target_updates"]:
            group = [row for row in rows if row["arm"] == arm
                     and row["target_updates"] == step]
            summary[f"{arm}_{step}"] = {
                "evaluations": len(group),
                "estimate_lower_than_truth": sum(
                    row["estimate_minus_truth_hybrid"] < 0 for row in group),
                "mean_estimate_minus_truth_hybrid": float(np.mean([
                    row["estimate_minus_truth_hybrid"] for row in group])),
                "mean_waveform_at_truth": float(np.mean([
                    row["waveform_at_truth"] for row in group])),
                "mean_arrival_at_truth": float(np.mean([
                    row["arrival_at_truth"] for row in group])),
            }
    result = {"purpose": spec["purpose"], "protocol_sha256": digest(PROTOCOL),
              "trajectory_result_sha256": digest(source_path),
              "target_fmm_sha256": digest(ROOT / parent["target_fmm_fields"]),
              "device": str(device), "development_only": True,
              "summary": summary, "rows": rows, "restriction": spec["restriction"]}
    lines = ["# Frozen transfer objective at truth and recovered source", "",
             spec["comparison"], "",
             "| Arm | Target updates | Estimate objective below truth | "
             "Mean estimate minus truth hybrid loss | Mean waveform loss at truth | "
             "Mean arrival loss at truth |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in spec["arms"]:
        for step in spec["target_updates"]:
            row = summary[f"{arm}_{step}"]
            lines.append(f"| {arm} | {step} | {row['estimate_lower_than_truth']}/12 | "
                         f"{row['mean_estimate_minus_truth_hybrid']:+.4f} | "
                         f"{row['mean_waveform_at_truth']:.4f} | "
                         f"{row['mean_arrival_at_truth']:.4f} |")
    lines.extend(["", "Positive estimate-minus-truth loss indicates that the known "
                  "source has a lower objective than the recovered estimate; it is a "
                  "diagnostic of search, not proof that a practical search can find it. "
                  "Negative values show that the frozen surrogate and arrival objective "
                  "prefer a wrong location over truth. This audit evaluates only the "
                  "existing estimates and truth, not a global minimum.", "",
                  spec["restriction"]])
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
