"""Evaluate two frozen paired receiver heads on archived depth development data."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_fmm_traveltime import first_arrival
from examples.run_eikonal_depth_sweep import arrival_metrics
from examples.run_fmm_receiver_head_paired import fmm_tables, raw_prediction, waveform_error
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import (EikonalTravelTimeCarrierWavePINN,
                                     SourceConditionedConfig, load_conditioned_checkpoint)

PROTOCOL = ROOT / "experiments/fmm_receiver_head_depth_audit_protocol.json"


def checked_inputs(spec: dict):
    paired_spec = json.loads((ROOT / spec["paired_protocol"]).read_text())
    paired = json.loads((ROOT / spec["paired_results"]).read_text())
    depth = json.loads((ROOT / spec["depth_results"]).read_text())
    if (hash_file(ROOT / spec["paired_results"]) != spec["paired_results_sha256"]
            or hash_file(ROOT / spec["depth_results"]) != spec["depth_results_sha256"]
            or hash_file(ROOT / spec["paired_protocol"]) != paired["protocol_sha256"]
            or hash_file(ROOT / spec["depth_protocol"]) != depth["protocol_sha256"]):
        raise ValueError("Archived input or protocol changed")
    if [abs(row["offset_m"]) for row in depth["events"]] != [80, 80, 160, 160, 280, 280] * 2:
        raise ValueError("Unexpected depth offsets")
    training = []
    base, metadata = load_conditioned_checkpoint(
        ROOT / paired_spec["base_receiver_checkpoint"], torch.device("cpu"))
    del base
    if hash_file(ROOT / paired_spec["base_receiver_checkpoint"]) != paired_spec["base_receiver_sha256"]:
        raise ValueError("Base receiver checkpoint changed")
    for record, digest in zip(metadata["training_record"]["receiver_observations"],
                              paired["training_records_sha256"]):
        path = Path(record["path"])
        if hash_file(path) != record["sha256"] or digest != record["sha256"]:
            raise ValueError(f"Training record changed: {path}")
        training.append(path)
    if len(training) != 29:
        raise ValueError("Training source count changed")
    development = []
    for row in depth["events"]:
        path = ROOT / "data/observations/eikonal_depth_sweep" / row["file"]
        if hash_file(path) != row["sha256"] or path in training:
            raise ValueError(f"Depth record changed: {path}")
        development.append(path)
    for name in ("learned", "fmm"):
        if hash_file(Path(paired["model_paths"][name])) != spec["head_sha256"][name]:
            raise ValueError(f"Receiver head changed: {name}")
    if len(development) != 12 or len({r["file"] for r in depth["events"]}) != 12:
        raise ValueError("Expected twelve distinct development records")
    return paired_spec, paired, depth, training, development


def build_model(base_checkpoint: Path, head_path: Path, name: str,
                digest: str, device: torch.device, protocol: dict):
    base, metadata = load_conditioned_checkpoint(base_checkpoint, device)
    config = SourceConditionedConfig()
    model = EikonalTravelTimeCarrierWavePINN(
        5., config.time_length_s, eikonal_model=copy.deepcopy(base.eikonal_model),
        reference_velocity_m_s=3500., distance_core_m=20.,
        hidden_dim=protocol["hidden_dim"], num_layers=protocol["num_layers"],
        output_scale=protocol["output_scale"]).to(device)
    saved = torch.load(head_path, map_location=device, weights_only=True)
    if (saved["input"] != name or saved["protocol_sha256"] != digest
            or saved["frozen_carrier_checkpoint_sha256"] != protocol["base_receiver_sha256"]):
        raise ValueError("Receiver head provenance mismatch")
    model.net.load_state_dict(saved["head_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    return model, scale


def pick_diagnostic(fd: np.ndarray, predicted: np.ndarray, times: np.ndarray,
                    origin: float) -> dict:
    fd_picks = np.asarray([first_arrival(row, times) for row in fd])
    model_picks = np.asarray([first_arrival(row, times) for row in predicted])
    peaks = np.max(np.abs(fd), axis=1)
    valid = np.isfinite(fd_picks) & (peaks >= .01 * peaks.max()) & (fd_picks <= 1.85)
    early = times < origin - .10
    return {"valid_fd_station_count": int(valid.sum()),
            "model_picks_before_origin_minus_0_10_s": int(np.sum(valid & (model_picks < origin - .10))),
            "model_picks_at_t0": int(np.sum(valid & (model_picks == times[0]))),
            "model_early_rms_over_fd_full_rms": float(np.sqrt(np.mean(predicted[:, early] ** 2)) /
                                                      np.sqrt(np.mean(fd ** 2))),
            "raw_threshold_pick": arrival_metrics(fd, predicted, times)}


def main():
    spec = json.loads(PROTOCOL.read_text())
    out = ROOT / spec["result_file"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    protocol, paired, depth, training, development = checked_inputs(spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fmm_train, fmm_depth, fmm_cost = fmm_tables(protocol, training, development, device)
    models = {name: build_model(ROOT / protocol["base_receiver_checkpoint"],
                               Path(paired["model_paths"][name]), name,
                               paired["protocol_sha256"], device, protocol)
              for name in ("learned", "fmm")}
    anchor_spec = json.loads((ROOT / spec["depth_protocol"]).read_text())["anchors"]
    anchors = {}
    for item in anchor_spec:
        path = ROOT / item["observation"]
        index = next(i for i, p in enumerate(training) if p.resolve() == path.resolve())
        anchors[item["name"]] = {
            name: raw_prediction(model, path, scale,
                                 fmm_train[index].cpu().numpy() if name == "fmm" else None)
            for name, (model, scale) in models.items()}
    rows = []
    for original, path, arrival in zip(depth["events"], development, fmm_depth):
        with np.load(path) as record:
            times = np.asarray(record["times"], dtype=np.float64)
            source = np.asarray(record["source_truth"], dtype=np.float64)
        if not np.allclose(source, original["source_m_s"], rtol=0, atol=1e-5):
            raise ValueError(f"Source mismatch: {path}")
        results = {}
        for name, (model, scale) in models.items():
            fd, predicted = raw_prediction(model, path, scale,
                                           arrival if name == "fmm" else None)
            anchor_fd, anchor_predicted = anchors[original["anchor"]][name]
            fd_change, pred_change = fd - anchor_fd, predicted - anchor_predicted
            results[name] = {**waveform_error(fd, predicted),
                             "change_relative_l2": float(np.linalg.norm(pred_change - fd_change) /
                                                         np.linalg.norm(fd_change)),
                             "change_cosine": float(np.sum(fd_change * pred_change) /
                                                    (np.linalg.norm(fd_change) * np.linalg.norm(pred_change))),
                             "pick_diagnostic": pick_diagnostic(fd, predicted, times, float(source[3]))}
        rows.append({"event": original["file"], "sha256": original["sha256"],
                     "anchor": original["anchor"], "offset_m": original["offset_m"],
                     "models": results})
        print(f"{original['file']}: learned={results['learned']['raw_l2']:.3f}, "
              f"FMM={results['fmm']['raw_l2']:.3f}", flush=True)
    report = {"role": spec["role"], "protocol_sha256": hash_file(PROTOCOL),
              "paired_results_sha256": spec["paired_results_sha256"],
              "depth_results_sha256": spec["depth_results_sha256"],
              "head_sha256": spec["head_sha256"], "fmm_setup": fmm_cost,
              "events": rows, "limits": spec["limits"]}
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
