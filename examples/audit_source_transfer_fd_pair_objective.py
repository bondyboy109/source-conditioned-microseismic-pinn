"""Audit whether FD-pair supervision changes truth-versus-wrong-source ordering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.diagnose_source_transfer_trajectory import config, intermediate
from examples.run_source_transfer_fd_pair import setup as pair_setup
from examples.run_source_transfer_fd_pair_weight import PROTOCOL as SWEEP_PROTOCOL, checkpoint
from examples.run_source_transfer_moe import digest, new_model
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, load_event, pick_arrivals, sampled_waveforms, waveform_scores,
)

PROTOCOL = ROOT / "experiments/source_transfer_fd_pair_objective_protocol.json"


def main():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sweep = json.loads(SWEEP_PROTOCOL.read_text(encoding="utf-8"))
    base, control, _, trajectory, parent, setup, _ = pair_setup()
    _, _, _, loc, _ = config()
    if (ROOT / spec["weight_sweep_protocol"] != SWEEP_PROTOCOL
            or spec["selected_weight"] not in sweep["weights"]
            or spec["selected_arm"] not in sweep["arms"]
            or spec["seeds"] != sweep["seeds"]):
        raise ValueError("Selected weight or sweep provenance changed")
    output = ROOT / spec["result_json"]
    report = ROOT / spec["report_file"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite objective audit")
    original_path = ROOT / spec["original_objective_result"]
    original = json.loads(original_path.read_text(encoding="utf-8"))
    original_rows = [r for r in original["rows"]
                     if r["arm"] == spec["selected_arm"]
                     and r["target_updates"] == parent["phase_updates"]]
    if len(original_rows) != len(spec["seeds"]) * 4:
        raise ValueError("Unexpected original comparison count")
    receivers = np.load(ROOT / "data/observations/receiver_coords.npy").astype(np.float32)
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    search, arrival = loc["search"], loc["arrival_constraint"]
    event_names = sorted({r["event"] for r in original_rows})
    events = {}
    for name in event_names:
        observation = load_event(ROOT / name, receivers)
        _, picks = pick_arrivals(observation, arrival)
        sampled = sampled_waveforms(observation, domain,
                                    search["refine_receiver_stride"],
                                    search["refine_time_stride"], device)
        events[name] = (picks, sampled)
    rows = []
    for seed in spec["seeds"]:
        initial = intermediate(trajectory, spec["selected_arm"], seed,
                               base["initial_target_updates"])
        branches = {
            "restarted_control": ROOT / control["checkpoint_directory"] /
            f"{spec['selected_arm']}_seed{seed}_restarted_control.pth",
            "fd_pair_weight_0.05": checkpoint(
                sweep, spec["selected_arm"], seed, spec["selected_weight"]),
        }
        for branch, path in branches.items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            expected = digest(ROOT / base["control_protocol"] if branch == "restarted_control"
                              else SWEEP_PROTOCOL)
            if (payload["protocol_sha256"] != expected
                    or payload["initial_checkpoint_sha256"] != digest(initial)):
                raise ValueError(f"Checkpoint provenance mismatch: {path}")
            model = new_model(fields, receivers, setup, device)
            model.load_state_dict(payload["state_dict"])
            model.eval().requires_grad_(False)
            for old in [r for r in original_rows if r["seed"] == seed]:
                picks, sampled = events[old["event"]]
                candidates = np.stack((old["truth_m_s"], old["estimate_m_s"])).astype(np.float32)
                wave = waveform_scores(model, candidates, domain, sampled, 2)
                travel = arrival_scores(model, candidates, picks,
                                        arrival["arrival_scale_s"], 2)
                hybrid = wave + arrival["hybrid_weight"] * travel
                rows.append({"seed": seed, "branch": branch, "event": old["event"],
                             "waveform_at_truth": float(wave[0]),
                             "waveform_at_estimate": float(wave[1]),
                             "arrival_at_truth": float(travel[0]),
                             "arrival_at_estimate": float(travel[1]),
                             "estimate_minus_truth_waveform": float(wave[1] - wave[0]),
                             "estimate_minus_truth_hybrid": float(hybrid[1] - hybrid[0])})
            print(f"{branch} seed={seed}: audited", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = {}
    for branch in branches:
        group = [r for r in rows if r["branch"] == branch]
        summary[branch] = {
            "comparisons": len(group),
            "wrong_estimate_lower_waveform": sum(
                r["estimate_minus_truth_waveform"] < 0 for r in group),
            "wrong_estimate_lower_hybrid": sum(
                r["estimate_minus_truth_hybrid"] < 0 for r in group),
            "mean_estimate_minus_truth_waveform": float(np.mean([
                r["estimate_minus_truth_waveform"] for r in group])),
            "mean_estimate_minus_truth_hybrid": float(np.mean([
                r["estimate_minus_truth_hybrid"] for r in group])),
        }
    result = {"protocol_sha256": digest(PROTOCOL),
              "sweep_protocol_sha256": digest(SWEEP_PROTOCOL),
              "original_objective_result_sha256": digest(original_path),
              "device": str(device), "summary": summary, "rows": rows,
              "restriction": spec["restriction"]}
    lines = ["# FD-pair objective-ordering audit", "",
             "The selected 0.05 branch and its paired restarted control score the same "
             "truth and previously recovered wrong-source candidates. Positive gaps favor truth.", "",
             "| Branch | Wrong source wins waveform | Wrong source wins hybrid | "
             "Mean wrong-minus-truth waveform | Mean wrong-minus-truth hybrid |",
             "|---|---:|---:|---:|---:|"]
    for branch, row in summary.items():
        lines.append(f"| {branch} | {row['wrong_estimate_lower_waveform']}/12 | "
                     f"{row['wrong_estimate_lower_hybrid']}/12 | "
                     f"{row['mean_estimate_minus_truth_waveform']:+.4f} | "
                     f"{row['mean_estimate_minus_truth_hybrid']:+.4f} |")
    lines.extend(["", spec["restriction"]])
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
