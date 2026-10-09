"""Rank fixed candidate depths for spent FD records with frozen paired heads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_eikonal_causal_gate import earliest_signal_gate
from examples.audit_fmm_receiver_head_depth import build_model, checked_inputs
from examples.audit_fmm_carrier_waveform_swap import frozen_head
from examples.run_fmm_receiver_head_paired import fmm_tables
from examples.run_fmm_seeded_waveform_development import hash_file

PROTOCOL = ROOT / "experiments/fmm_receiver_head_depth_objective_protocol.json"


def predict_at_source(model, receivers: np.ndarray, times: np.ndarray,
                      source: np.ndarray, scale: np.ndarray,
                      travel: np.ndarray | None) -> np.ndarray:
    device = next(model.parameters()).device
    source_t = torch.as_tensor(source.astype(np.float32) / scale, device=device)
    station_t = torch.as_tensor(receivers.astype(np.float32) / scale[:3], device=device)
    time_t = torch.as_tensor(times.astype(np.float32) / scale[3], device=device)
    predicted = np.empty((len(receivers), len(times)), dtype=np.float64)
    with torch.inference_mode():
        for station in range(len(receivers)):
            points = torch.cat((station_t[station].expand(len(times), 3),
                                time_t[:, None]), dim=1)
            source_batch = source_t.expand(len(times), 4)
            if travel is None:
                amplitude = model(points, source_batch)
            else:
                arrival = torch.full((len(times), 1), float(travel[station]), device=device)
                amplitude = frozen_head(model, points, source_batch, arrival)
            predicted[station] = amplitude[:, 0].cpu().numpy()
    return predicted


def rank_errors(observed: np.ndarray, candidates: list[np.ndarray],
                true_index: int) -> dict:
    energy = float(np.sum(observed ** 2))
    if energy <= 0 or not all(np.isfinite(item).all() for item in candidates):
        raise ValueError("Empty FD observation or nonfinite candidate waveform")
    loss = [float(np.sum((item - observed) ** 2) / energy) for item in candidates]
    order = np.argsort(loss, kind="stable")
    return {"normalized_mse_by_depth": loss,
            "best_index": int(order[0]),
            "true_index": true_index,
            "true_rank": int(np.flatnonzero(order == true_index)[0] + 1),
            "true_mse": loss[true_index],
            "anchor_mse": loss[3]}


def main():
    spec = json.loads(PROTOCOL.read_text())
    out = ROOT / spec["result_file"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    gate_spec = json.loads((ROOT / spec["fixed_gate_protocol"]).read_text())
    gate_report = json.loads((ROOT / spec["fixed_gate_result"]).read_text())
    depth_spec = json.loads((ROOT / spec["depth_audit_protocol"]).read_text())
    depth_report = json.loads((ROOT / spec["depth_audit_result"]).read_text())
    if (hash_file(ROOT / spec["fixed_gate_result"]) != spec["fixed_gate_result_sha256"]
            or hash_file(ROOT / spec["depth_audit_result"]) != spec["depth_audit_result_sha256"]
            or hash_file(ROOT / spec["fixed_gate_protocol"]) != gate_report["protocol_sha256"]
            or hash_file(ROOT / spec["depth_audit_protocol"]) != depth_report["protocol_sha256"]
            or spec["candidate_offset_m"] != [-280, -160, -80, 0, 80, 160, 280]
            or gate_report["depth_audit_results_sha256"] != spec["depth_audit_result_sha256"]):
        raise ValueError("Registered inputs, scan grid or gate audit changed")
    protocol, paired, depth, training, development = checked_inputs(depth_spec)
    if (len(depth["events"]) != len(depth_report["events"])
            or len(depth["events"]) != len(gate_report["events"])
            or len(depth["events"]) != len(development) or len(development) != 12):
        raise ValueError("Depth or gate audit does not contain the same twelve events")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fmm_train, fmm_dev, setup = fmm_tables(protocol, training, development, device)
    models = {name: build_model(ROOT / protocol["base_receiver_checkpoint"],
                               Path(paired["model_paths"][name]), name,
                               paired["protocol_sha256"], device, protocol)
              for name in ("learned", "fmm")}
    origin_spec = json.loads((ROOT / depth_spec["depth_protocol"]).read_text())
    anchors = {row["name"]: row for row in origin_spec["anchors"]}
    cached = {}
    for name, anchor in anchors.items():
        with np.load(ROOT / anchor["observation"]) as record:
            receivers = np.asarray(record["receiver_coords"])
            times = np.asarray(record["times"])
        indexed = next(i for i, path in enumerate(training)
                       if path.resolve() == (ROOT / anchor["observation"]).resolve())
        variants = {key: [] for key in ("learned", "fmm", "learned_gate", "fmm_gate")}
        for offset in spec["candidate_offset_m"]:
            candidate = np.asarray(anchor["source_m_s"], dtype=np.float64).copy()
            candidate[2] += offset
            if offset == 0:
                fmm_times = fmm_train[indexed].cpu().numpy()
            else:
                position = next(i for i, row in enumerate(depth["events"])
                                if row["anchor"] == name and row["offset_m"] == offset)
                fmm_times = fmm_dev[position]
            gate_params = gate_report["gate_parameters"]
            gate = earliest_signal_gate(receivers, times, candidate,
                                        gate_params["maximum_speed_m_s"],
                                        gate_params["source_support_radius_m"],
                                        gate_params["source_lead_s"],
                                        gate_params["gate_rise_s"])
            for method, (model, scale) in models.items():
                waveform = predict_at_source(model, receivers, times, candidate, scale,
                                             fmm_times if method == "fmm" else None)
                variants[method].append(waveform)
                variants[method + "_gate"].append(waveform * gate)
        cached[name] = variants
        print(f"Predicted seven candidate depths around {name}", flush=True)
    rows = []
    for original, previous, gated, path in zip(depth["events"], depth_report["events"],
                                                gate_report["events"], development):
        if original["file"] != previous["event"] or hash_file(path) != original["sha256"]:
            raise ValueError(f"Development source changed: {path}")
        if gated["event"] != original["file"] or gated["sha256"] != original["sha256"]:
            raise ValueError(f"Gated reference changed: {path}")
        with np.load(path) as record:
            fd = np.asarray(record["data"], dtype=np.float64)
            source = np.asarray(record["source_truth"], dtype=np.float64)
        true_index = spec["candidate_offset_m"].index(original["offset_m"])
        if not np.allclose(source, original["source_m_s"], rtol=0, atol=1e-5):
            raise ValueError(f"Truth differs: {path}")
        rankings = {name: rank_errors(fd, candidates, true_index)
                    for name, candidates in cached[original["anchor"]].items()}
        for name in ("learned", "fmm"):
            if (not np.isclose(np.sqrt(rankings[name]["true_mse"]),
                               previous["models"][name]["raw_l2"], rtol=0, atol=1e-6)
                    or not np.isclose(np.sqrt(rankings[name + "_gate"]["true_mse"]),
                                      gated["models"][name]["after"]["raw_l2"],
                                      rtol=0, atol=1e-6)):
                raise ValueError(f"True-source waveform disagrees with previous audit: {path}")
        rows.append({"event": original["file"], "sha256": original["sha256"],
                     "anchor": original["anchor"], "true_offset_m": original["offset_m"],
                     "rankings": rankings})
        print(f"{original['file']}: best learned {spec['candidate_offset_m'][rankings['learned']['best_index']]:+d} m, "
              f"FMM {spec['candidate_offset_m'][rankings['fmm']['best_index']]:+d} m", flush=True)
    report = {"role": spec["role"], "protocol_sha256": hash_file(PROTOCOL),
              "candidate_offset_m": spec["candidate_offset_m"],
              "fixed_gate_result_sha256": spec["fixed_gate_result_sha256"],
              "depth_audit_result_sha256": spec["depth_audit_result_sha256"],
              "fmm_setup": setup, "events": rows, "limits": spec["limits"]}
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
