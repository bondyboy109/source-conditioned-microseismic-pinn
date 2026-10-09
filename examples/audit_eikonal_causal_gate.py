"""Evaluate one fixed earliest-signal taper on spent depth development records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_eikonal_depth_early_picks import early_picks
from examples.run_eikonal_depth_sweep import DATA, preflight
from examples.run_eikonal_directional_grid import checked_record, predict, response
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_causal_gate_development_protocol.json"
OUTPUT = ROOT / "experiments/eikonal_causal_gate_development_results.json"


def earliest_signal_gate(receivers: np.ndarray, times: np.ndarray,
                         source: np.ndarray, vmax: float, support_m: float,
                         lead_s: float, rise_s: float) -> np.ndarray:
    """C2 approximate onset, with an upper bound on propagation speed."""
    if (receivers.ndim != 2 or receivers.shape[1] != 3 or times.ndim != 1
            or source.shape != (4,) or not np.isfinite(receivers).all()
            or not np.isfinite(source).all() or not np.isfinite(times).all()
            or not np.isfinite(vmax) or vmax <= 0 or support_m < 0
            or not np.isfinite(lead_s) or lead_s < 0 or not np.isfinite(rise_s)
            or rise_s <= 0):
        raise ValueError("Invalid source, receiver or onset parameters")
    distance = np.linalg.norm(receivers.astype(np.float64) - source[:3], axis=1)
    earliest = source[3] - lead_s + np.maximum(distance - support_m, 0) / vmax
    phase = np.clip((times.astype(np.float64)[None, :] - earliest[:, None]) / rise_s,
                    0., 1.)
    return (phase**3 * (10. + phase * (-15. + 6. * phase))).astype(np.float32)


def waveform_metrics(fd: np.ndarray, pred: np.ndarray) -> dict:
    observed = fd.astype(np.float64)
    predicted = pred.astype(np.float64)
    norm = float(np.sum(observed**2))
    if norm <= 0:
        raise ValueError("Empty FD observation")
    active = np.abs(observed) >= .05 * np.max(np.abs(observed))
    return {"raw_relative_l2": float(np.sqrt(np.sum((predicted-observed)**2) / norm)),
            "active_relative_l2": float(np.sqrt(np.sum((predicted[active]-observed[active])**2)
                                                / np.sum(observed[active]**2)))}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Development diagnostic already exists: {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    depth_protocol_path = ROOT / protocol["depth_protocol"]
    depth_protocol = json.loads(depth_protocol_path.read_text(encoding="utf-8"))
    depth_results_path = ROOT / protocol["depth_result"]
    depth_results = json.loads(depth_results_path.read_text(encoding="utf-8"))
    if (depth_results["protocol_sha256"] != hash_file(depth_protocol_path)
            or depth_results["checkpoint_sha256"] != {name: protocol[f"{name}_sha256"]
                                                   for name in ("eikonal", "straight")}
            or depth_results["velocity_sha256"] != protocol["velocity_sha256"]):
        raise ValueError("Registered depth results or frozen inputs changed")
    events, _ = preflight(depth_protocol)
    if len(events) != len(depth_results["events"]) == 12:
        raise ValueError("Unexpected depth event count")
    velocity = ROOT / depth_protocol["velocity_file"]
    if hash_file(velocity) != protocol["velocity_sha256"]:
        raise ValueError("Velocity file changed")
    vmax = float(np.max(np.load(velocity, mmap_mode="r")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name in ("eikonal", "straight"):
        checkpoint = ROOT / depth_protocol[f"{name}_checkpoint"]
        if hash_file(checkpoint) != protocol[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} model changed")
        model, meta = load_conditioned_checkpoint(checkpoint, device)
        model.eval().requires_grad_(False)
        scale = np.asarray([meta["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        models[name] = model, scale
    anchors = {a["name"]: ROOT / a["observation"] for a in depth_protocol["anchors"]}
    anchor_records = {}
    anchor_predictions = {}
    anchor_metrics = {}
    for item in depth_protocol["anchors"]:
        name = item["name"]
        record = checked_record(anchors[name], item["source_m_s"], anchors[name])
        anchor_records[name] = record
        fd, times, receivers, truth = record
        gate = earliest_signal_gate(receivers, times, truth, vmax,
                                    protocol["source_support_radius_m"],
                                    protocol["source_lead_s"], protocol["gate_rise_s"])
        anchor_metrics[name] = {}
        anchor_predictions[name] = {}
        for method, (model, scale) in models.items():
            raw = predict(model, scale, *record)
            corrected = raw * gate
            anchor_predictions[name][method] = (raw, corrected)
            anchor_metrics[name][method] = {"before": waveform_metrics(fd, raw),
                                            "after": waveform_metrics(fd, corrected)}
    rows = []
    for event, original in zip(events, depth_results["events"]):
        path = DATA / event["file"]
        if original["file"] != event["file"] or hash_file(path) != original["sha256"]:
            raise ValueError(f"Source file or original result changed: {path}")
        record = checked_record(path, event["source_m_s"], anchors[event["anchor"]])
        fd, times, receivers, truth = record
        gate = earliest_signal_gate(receivers, times, truth, vmax,
                                    protocol["source_support_radius_m"],
                                    protocol["source_lead_s"], protocol["gate_rise_s"])
        removed_fraction = float(np.sum((fd.astype(np.float64) * (1. - gate))**2)
                                 / np.sum(fd.astype(np.float64)**2))
        row = {"file": event["file"], "anchor": event["anchor"],
               "offset_m": event["offset_m"], "sha256": original["sha256"],
               "fd_energy_removed_fraction": removed_fraction,
               "models": {}}
        for method, (model, scale) in models.items():
            raw = predict(model, scale, *record)
            corrected = raw * gate
            before = waveform_metrics(fd, raw)
            after = waveform_metrics(fd, corrected)
            if not np.isclose(before["raw_relative_l2"],
                              original["models"][method]["full_record_relative_l2"],
                              rtol=1e-5, atol=1e-5):
                raise ValueError(f"Unmodified forward error changed: {path}, {method}")
            anchor_fd = anchor_records[event["anchor"]][0]
            anchor_raw, anchor_corrected = anchor_predictions[event["anchor"]][method]
            row["models"][method] = {
                "before": before, "after": after,
                "response_before": response(anchor_fd, fd, anchor_raw, raw),
                "response_after": response(anchor_fd, fd, anchor_corrected, corrected),
                "early_before": early_picks(fd, raw, times, float(truth[3])),
                "early_after": early_picks(fd, corrected, times, float(truth[3])),
            }
        rows.append(row)
        print(f"{event['file']}: FD gate loss={removed_fraction:.4%}; "
              f"eikonal L2 {row['models']['eikonal']['before']['raw_relative_l2']:.3f} -> "
              f"{row['models']['eikonal']['after']['raw_relative_l2']:.3f}; "
              f"early {row['models']['eikonal']['early_before']['model_picks_before_origin_minus_0_10_s']} -> "
              f"{row['models']['eikonal']['early_after']['model_picks_before_origin_minus_0_10_s']}",
              flush=True)
    summary = {}
    for method in models:
        summary[method] = {
            "mean_raw_l2_before": float(np.mean([r["models"][method]["before"]["raw_relative_l2"]
                                            for r in rows])),
            "mean_raw_l2_after": float(np.mean([r["models"][method]["after"]["raw_relative_l2"]
                                           for r in rows])),
            "mean_change_l2_before": float(np.mean([
                r["models"][method]["response_before"]["change_relative_l2"] for r in rows])),
            "mean_change_l2_after": float(np.mean([
                r["models"][method]["response_after"]["change_relative_l2"] for r in rows])),
            "early_pick_count_before": sum(r["models"][method]["early_before"][
                "model_picks_before_origin_minus_0_10_s"] for r in rows),
            "early_pick_count_after": sum(r["models"][method]["early_after"][
                "model_picks_before_origin_minus_0_10_s"] for r in rows),
            "start_pick_count_before": sum(r["models"][method]["early_before"][
                "model_picks_at_start_count"] for r in rows),
            "start_pick_count_after": sum(r["models"][method]["early_after"][
                "model_picks_at_start_count"] for r in rows),
            "paired_pick_count_before": sum(r["models"][method]["early_before"][
                "original_threshold_arrival_metrics"]["paired_pick_count"] for r in rows),
            "paired_pick_count_after": sum(r["models"][method]["early_after"][
                "original_threshold_arrival_metrics"]["paired_pick_count"] for r in rows),
        }
    output = {"purpose": protocol["purpose"], "role": protocol["data_role"],
              "protocol_sha256": hash_file(PROTOCOL),
              "depth_result_sha256": hash_file(depth_results_path),
              "checkpoint_sha256": depth_results["checkpoint_sha256"],
              "velocity_sha256": protocol["velocity_sha256"],
              "maximum_speed_m_s": vmax,
              "gate_parameters": {key: protocol[key] for key in
                                  ("source_support_radius_m", "source_lead_s", "gate_rise_s")},
              "anchor_metrics": anchor_metrics,
              "mean_fd_energy_removed_fraction": float(np.mean([
                  row["fd_energy_removed_fraction"] for row in rows])),
              "max_fd_energy_removed_fraction": float(np.max([
                  row["fd_energy_removed_fraction"] for row in rows])),
              "summary": summary, "events": rows,
              "limitations": protocol["limitations"]}
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
