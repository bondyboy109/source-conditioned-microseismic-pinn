"""Paired depth-coverage audit on previously inspected development waveforms."""

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
from examples.run_eikonal_depth_coverage33 import PROTOCOL, preflight
from examples.run_eikonal_depth_sweep import DATA
from examples.run_eikonal_directional_grid import checked_record, predict, response
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

OUTPUT = ROOT / "experiments/eikonal_depth_coverage33_development_results.json"


def summarize(rows: list[dict], label: str) -> dict:
    values = [row[label] for row in rows]
    return {
        "count": len(rows),
        "mean_full_l2": float(np.mean([v["waveform"]["raw_relative_l2"] for v in values])),
        "mean_active_l2": float(np.mean([v["waveform"]["active_relative_l2"] for v in values])),
        "mean_depth_change_l2": float(np.mean([v["response"]["change_relative_l2"]
                                               for v in values])),
        "mean_depth_change_cosine": float(np.mean([v["response"]["change_cosine"]
                                                   for v in values])),
        "early_pick_count": sum(v["early"]["model_picks_before_origin_minus_0_10_s"]
                                for v in values),
        "paired_pick_count": sum(v["early"]["original_threshold_arrival_metrics"]
                                 ["paired_pick_count"] for v in values),
        "mean_pick_rmse_s": float(np.mean([v["early"]["original_threshold_arrival_metrics"]
                                           ["rmse_s"] for v in values])),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Development result already exists: {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text())
    training, events = preflight(protocol)
    reference = json.loads((ROOT / protocol["original_development_result"]).read_text())
    old = {row["file"]: row for row in reference["events"]}
    depth = json.loads((ROOT / protocol["depth_protocol"]).read_text())
    path = ROOT / protocol["hybrid_output"] / "conditioned_5Hz.pth"
    receiver = ROOT / protocol["receiver_output"] / "conditioned_5Hz.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_conditioned_checkpoint(path, device)
    model.eval().requires_grad_(False)
    record = meta["training_record"]
    if (meta["architecture"]["kind"] != "eikonal_travel_time_carrier"
            or meta["architecture"].get("onset_gate") is not None
            or meta["frequency_hz"] != 5 or meta["epoch"] != 1000
            or record["initial_checkpoint"]["sha256"] != hash_file(receiver)
            or [item["sha256"] for item in record["receiver_observations"]]
            != [hash_file(item) for item in training]
            or len(training) != 33):
        raise ValueError("Candidate checkpoint has wrong architecture, training files or schedule")
    trained_hashes = {item["sha256"] for item in record["receiver_observations"]}
    scale = np.asarray([meta["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    anchors = {row["name"]: ROOT / row["observation"] for row in depth["anchors"]}
    anchor_data = {}
    for row in depth["anchors"]:
        values = checked_record(anchors[row["name"]], row["source_m_s"],
                                anchors[row["name"]])
        anchor_data[row["name"]] = (values[0], predict(model, scale, *values))
    rows = []
    for event in events:
        fd_path = DATA / event["file"]
        if hash_file(fd_path) in trained_hashes:
            raise ValueError(f"Development source is in candidate training: {fd_path}")
        fd, times, receivers, source = checked_record(
            fd_path, event["source_m_s"], anchors[event["anchor"]])
        wave = predict(model, scale, fd, times, receivers, source)
        anchor_fd, anchor_wave = anchor_data[event["anchor"]]
        previous = old[event["file"]]["models"]["eikonal"]
        row = {"file": event["file"], "anchor": event["anchor"],
               "offset_m": event["offset_m"], "sha256": hash_file(fd_path),
               "baseline": {"waveform": previous["before"],
                            "response": previous["response_before"],
                            "early": previous["early_before"]},
               "augmented": {"waveform": waveform_metrics(fd, wave),
                             "response": response(anchor_fd, fd, anchor_wave, wave),
                             "early": early_picks(fd, wave, times, float(source[3]))}}
        rows.append(row)
        print(f"{event['file']}: active L2 {row['baseline']['waveform']['active_relative_l2']:.3f}"
              f" -> {row['augmented']['waveform']['active_relative_l2']:.3f}; "
              f"response {row['baseline']['response']['change_relative_l2']:.3f}"
              f" -> {row['augmented']['response']['change_relative_l2']:.3f}", flush=True)
    partitions = {"all": rows,
                  "offset_80": [r for r in rows if abs(r["offset_m"]) == 80],
                  "offset_280": [r for r in rows if abs(r["offset_m"]) == 280],
                  **{anchor: [r for r in rows if r["anchor"] == anchor]
                     for anchor in protocol["anchors"]}}
    groups = {name: {label: summarize(records, label)
                     for label in ("baseline", "augmented")}
              for name, records in partitions.items()}
    training_times = {
        stage: json.loads((ROOT / protocol[f"{stage}_output"] /
                           "training_summary.json").read_text())["5Hz"]["seconds"]
        for stage in ("receiver", "hybrid")}
    report = {"role": protocol["role"], "protocol_sha256": hash_file(PROTOCOL),
              "baseline_checkpoint_sha256": protocol["baseline_sha256"],
              "candidate_checkpoint_sha256": hash_file(path),
              "training_manifest_sha256": {str(item): hash_file(item) for item in training},
              "training_seconds": training_times,
              "groups": groups, "events": rows,
              "limitations": "All 12 depth records had already been opened for development; four now also train the model. One seed, one velocity, no source inversion or independent confirmation."}
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
