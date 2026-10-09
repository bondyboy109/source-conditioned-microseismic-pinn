"""Compare trained gate with raw and postprocessed frozen baselines on spent data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_eikonal_causal_gate import waveform_metrics
from examples.audit_eikonal_depth_early_picks import early_picks
from examples.run_eikonal_depth_sweep import DATA, preflight
from examples.run_eikonal_directional_grid import checked_record, predict, response
from examples.run_fmm_seeded_waveform_development import hash_file
from examples.run_trained_onset_gate_development import PROTOCOL, check_inputs
from source_conditioned_pinn import load_conditioned_checkpoint

OUTPUT = ROOT / "experiments/trained_onset_gate_development_results.json"


def mean(records: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in records]))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Development result already exists: {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    training = check_inputs(protocol)
    baseline = json.loads((ROOT / protocol["development_baseline"]).read_text())
    depth_protocol = json.loads((ROOT / "experiments/eikonal_depth_sweep_protocol.json").read_text())
    events, _ = preflight(depth_protocol)
    if len(events) != len(baseline["events"]) or len(events) != 12:
        raise ValueError("Development FD event count changed")
    trained_path = ROOT / protocol["hybrid_output"] / "conditioned_5Hz.pth"
    pretrain_path = ROOT / protocol["receiver_output"] / "conditioned_5Hz.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_conditioned_checkpoint(trained_path, device)
    model.eval().requires_grad_(False)
    if (meta["architecture"].get("onset_gate") != protocol["onset_gate"]
            or meta["epoch"] != 1000 or meta["frequency_hz"] != 5
            or meta["training_record"]["initial_checkpoint"]["sha256"]
            != hash_file(pretrain_path)
            or [row["sha256"] for row in meta["training_record"]["receiver_observations"]]
            != [hash_file(path) for path in training]):
        raise ValueError("Trained checkpoint does not match registered gate and training events")
    scale = np.asarray([meta["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    anchors = {item["name"]: ROOT / item["observation"]
               for item in depth_protocol["anchors"]}
    anchor_predictions = {}
    for item in depth_protocol["anchors"]:
        record = checked_record(anchors[item["name"]], item["source_m_s"],
                                anchors[item["name"]])
        anchor_predictions[item["name"]] = (record[0], predict(model, scale, *record))
    rows = []
    for event, original in zip(events, baseline["events"]):
        path = DATA / event["file"]
        if original["file"] != event["file"] or original["sha256"] != hash_file(path):
            raise ValueError(f"Development record changed: {path}")
        fd, times, receivers, source = checked_record(
            path, event["source_m_s"], anchors[event["anchor"]])
        output = predict(model, scale, fd, times, receivers, source)
        anchor_fd, anchor_output = anchor_predictions[event["anchor"]]
        measurements = waveform_metrics(fd, output)
        changes = response(anchor_fd, fd, anchor_output, output)
        picks = early_picks(fd, output, times, float(source[3]))
        row = {"file": event["file"], "anchor": event["anchor"],
               "offset_m": event["offset_m"], "sha256": original["sha256"],
               "trained_gate": {"waveform": measurements,
                                "response": changes, "early": picks},
               "baseline_raw": {"waveform": original["models"]["eikonal"]["before"],
                                "response": original["models"]["eikonal"]["response_before"],
                                "early": original["models"]["eikonal"]["early_before"]},
               "baseline_postprocessed": {
                   "waveform": original["models"]["eikonal"]["after"],
                   "response": original["models"]["eikonal"]["response_after"],
                   "early": original["models"]["eikonal"]["early_after"]}}
        rows.append(row)
        print(f"{event['file']}: active L2 {row['baseline_raw']['waveform']['active_relative_l2']:.3f}"
              f" -> {measurements['active_relative_l2']:.3f}; "
              f"depth response {row['baseline_raw']['response']['change_relative_l2']:.3f}"
              f" -> {changes['change_relative_l2']:.3f}", flush=True)
    summary = {}
    for label in ("baseline_raw", "baseline_postprocessed", "trained_gate"):
        group = [row[label] for row in rows]
        summary[label] = {
            "mean_full_l2": mean([r["waveform"] for r in group], "raw_relative_l2"),
            "mean_active_l2": mean([r["waveform"] for r in group], "active_relative_l2"),
            "mean_depth_change_l2": mean([r["response"] for r in group], "change_relative_l2"),
            "mean_depth_change_cosine": mean([r["response"] for r in group], "change_cosine"),
            "early_pick_count": sum(r["early"]["model_picks_before_origin_minus_0_10_s"]
                                    for r in group),
            "paired_pick_count": sum(r["early"]["original_threshold_arrival_metrics"]
                                     ["paired_pick_count"] for r in group),
            "mean_pick_rmse_s": mean([r["early"]["original_threshold_arrival_metrics"]
                                      for r in group], "rmse_s")}
    training_summaries = {
        stage: json.loads((ROOT / protocol[f"{stage}_output"] /
                           "training_summary.json").read_text())["5Hz"]
        for stage in ("receiver", "hybrid")}
    report = {"role": protocol["role"], "protocol_sha256": hash_file(PROTOCOL),
              "baseline_sha256": protocol["baseline_sha256"],
              "trained_checkpoint_sha256": hash_file(trained_path),
              "pretrain_checkpoint_sha256": hash_file(pretrain_path),
              "training_seconds": {stage: value["seconds"]
                                   for stage, value in training_summaries.items()},
              "summary": summary, "events": rows,
              "limitations": "Spent development data, one velocity and seed; gate uses straight distance/maximum speed, not an eikonal arrival; no new localization claim"}
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
