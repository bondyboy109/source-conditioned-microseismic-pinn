"""Apply the previously fixed approximate onset gate to paired frozen heads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_eikonal_causal_gate import earliest_signal_gate
from examples.audit_fmm_receiver_head_depth import (build_model, checked_inputs,
                                                    pick_diagnostic)
from examples.run_fmm_receiver_head_paired import fmm_tables, raw_prediction, waveform_error
from examples.run_fmm_seeded_waveform_development import hash_file

PROTOCOL = ROOT / "experiments/fmm_receiver_head_fixed_gate_protocol.json"


def main():
    spec = json.loads(PROTOCOL.read_text())
    output = ROOT / spec["result_file"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    depth_spec = json.loads((ROOT / spec["depth_audit_protocol"]).read_text())
    depth_report = json.loads((ROOT / spec["depth_audit_results"]).read_text())
    gate_source = ROOT / spec["gate_source_protocol"]
    if (hash_file(ROOT / spec["depth_audit_results"]) != spec["depth_audit_results_sha256"]
            or hash_file(ROOT / spec["depth_audit_protocol"]) != depth_report["protocol_sha256"]
            or hash_file(gate_source) != spec["gate_source_protocol_sha256"]
            or hash_file(ROOT / spec["gate_source_results"]) != spec["gate_source_results_sha256"]):
        raise ValueError("Fixed input changed")
    gate_spec = json.loads(gate_source.read_text())
    gate_result = json.loads((ROOT / spec["gate_source_results"]).read_text())
    protocol, paired, depth, training, dev = checked_inputs(depth_spec)
    if (len(depth_report["events"]) != len(dev) or len(dev) != 12
            or depth_report["head_sha256"] != depth_spec["head_sha256"]
            or gate_spec["velocity_sha256"] != protocol["velocity_sha256"]
            or gate_result["protocol_sha256"] != spec["gate_source_protocol_sha256"]):
        raise ValueError("Depth audit or gate velocity mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fmm_train, fmm_dev, setup = fmm_tables(protocol, training, dev, device)
    models = {name: build_model(ROOT / protocol["base_receiver_checkpoint"],
                               Path(paired["model_paths"][name]), name,
                               paired["protocol_sha256"], device, protocol)
              for name in ("learned", "fmm")}
    anchor_spec = json.loads((ROOT / depth_spec["depth_protocol"]).read_text())["anchors"]
    anchors = {}
    gate_params = (gate_result["maximum_speed_m_s"],
                   gate_spec["source_support_radius_m"], gate_spec["source_lead_s"],
                   gate_spec["gate_rise_s"])
    for item in anchor_spec:
        path = ROOT / item["observation"]
        index = next(i for i, p in enumerate(training) if p.resolve() == path.resolve())
        with np.load(path) as fd_record:
            gate = earliest_signal_gate(fd_record["receiver_coords"], fd_record["times"],
                                        fd_record["source_truth"], *gate_params)
        anchors[item["name"]] = {}
        for name, (model, scale) in models.items():
            fd, prediction = raw_prediction(model, path, scale,
                                            fmm_train[index].cpu().numpy() if name == "fmm" else None)
            anchors[item["name"]][name] = fd, prediction * gate
    rows = []
    for source_row, previous, path, fmm_arrival in zip(depth["events"],
                                                         depth_report["events"], dev, fmm_dev):
        if source_row["file"] != previous["event"] or source_row["sha256"] != previous["sha256"]:
            raise ValueError(f"Previous event differs: {path}")
        with np.load(path) as record:
            receivers = np.asarray(record["receiver_coords"])
            times = np.asarray(record["times"])
            truth = np.asarray(record["source_truth"])
        gate = earliest_signal_gate(receivers, times, truth, *gate_params)
        variants = {}
        for name, (model, scale) in models.items():
            fd, prediction = raw_prediction(model, path, scale,
                                            fmm_arrival if name == "fmm" else None)
            before = waveform_error(fd, prediction)
            if (abs(before["raw_l2"] - previous["models"][name]["raw_l2"]) > 1e-5
                    or abs(before["active_l2"] - previous["models"][name]["active_l2"]) > 1e-5):
                raise ValueError(f"Unmodified waveform no longer matches archive: {path}")
            corrected = prediction * gate
            anchor_fd, anchor_corrected = anchors[source_row["anchor"]][name]
            fd_change, pred_change = fd - anchor_fd, corrected - anchor_corrected
            variants[name] = {
                "before": before, "after": waveform_error(fd, corrected),
                "response_l2_after": float(np.linalg.norm(pred_change - fd_change) /
                                           np.linalg.norm(fd_change)),
                "pick_after": pick_diagnostic(fd, corrected, times, float(truth[3]))}
        rows.append({"event": previous["event"], "sha256": previous["sha256"],
                     "offset_m": previous["offset_m"],
                     "fd_energy_removed_fraction": float(np.sum((fd * (1 - gate)) ** 2) /
                                                         np.sum(fd ** 2)),
                     "models": variants})
        print(f"{previous['event']}: learned {variants['learned']['before']['raw_l2']:.3f} -> "
              f"{variants['learned']['after']['raw_l2']:.3f}; FMM "
              f"{variants['fmm']['before']['raw_l2']:.3f} -> "
              f"{variants['fmm']['after']['raw_l2']:.3f}", flush=True)
    report = {"role": spec["role"], "protocol_sha256": hash_file(PROTOCOL),
              "depth_audit_results_sha256": spec["depth_audit_results_sha256"],
              "gate_source_protocol_sha256": spec["gate_source_protocol_sha256"],
              "gate_parameters": {"maximum_speed_m_s": gate_params[0],
                                  **{k: gate_spec[k] for k in
                                     ("source_support_radius_m", "source_lead_s", "gate_rise_s")}},
              "fmm_setup": setup, "events": rows, "limits": spec["limits"]}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
