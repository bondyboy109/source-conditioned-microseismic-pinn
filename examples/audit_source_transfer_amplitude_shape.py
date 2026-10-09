"""Attribute current transfer-MoE errors to amplitude at fixed true sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.diagnose_source_transfer_trajectory import intermediate
from examples.run_source_transfer_fd_pair_weight import (
    PROTOCOL as SWEEP_PROTOCOL, checkpoint, read_setup,
)
from examples.run_source_transfer_moe import digest, new_model
from examples.run_traveltime_aligned_moe_development import (
    aligned_prediction, load_observations,
)

PROTOCOL = ROOT / "experiments/source_transfer_amplitude_shape_protocol.json"


def amplitude_metrics(observed, predicted, active_fraction):
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if (observed.ndim != 2 or observed.shape != predicted.shape
            or not np.isfinite(observed).all() or not np.isfinite(predicted).all()
            or not 0 < active_fraction < 1):
        raise ValueError("Invalid waveform arrays or activity threshold")
    energy = float(np.sum(observed ** 2))
    if energy <= 0:
        raise ValueError("Reference has zero energy")
    predicted_energy = float(np.sum(predicted ** 2))
    dot = float(np.sum(observed * predicted))
    gain = max(0.0, dot / predicted_energy) if predicted_energy > 0 else 0.0
    raw = float(np.sum((predicted - observed) ** 2) / energy)
    global_adjusted = float(np.sum((gain * predicted - observed) ** 2) / energy)
    receiver_energy = np.sum(observed ** 2, axis=1)
    receiver_prediction_energy = np.sum(predicted ** 2, axis=1)
    receiver_dot = np.sum(observed * predicted, axis=1)
    receiver_gains = np.maximum(0.0, np.divide(
        receiver_dot, receiver_prediction_energy,
        out=np.zeros_like(receiver_dot), where=receiver_prediction_energy > 0))
    receiver_adjusted = float(np.sum(
        (receiver_gains[:, None] * predicted - observed) ** 2) / energy)
    if receiver_adjusted > global_adjusted + 1e-10 or global_adjusted > raw + 1e-10:
        raise ValueError("Amplitude attribution violated nested-fit ordering")
    active = receiver_energy >= active_fraction * energy
    return {
        "raw_relative_l2": float(np.sqrt(raw)),
        "prediction_to_observation_energy_ratio": predicted_energy / energy,
        "waveform_cosine": (dot / np.sqrt(energy * predicted_energy)
                            if predicted_energy > 0 else None),
        "best_nonnegative_global_gain": gain,
        "global_gain_relative_l2": float(np.sqrt(global_adjusted)),
        "receiver_gain_oracle_relative_l2": float(np.sqrt(receiver_adjusted)),
        "raw_normalized_squared_error": raw,
        "squared_error_removed_by_global_gain": raw - global_adjusted,
        "additional_squared_error_removed_by_receiver_gains": global_adjusted - receiver_adjusted,
        "remaining_normalized_squared_error": receiver_adjusted,
        "active_receiver_count": int(active.sum()),
        "receiver_gains": receiver_gains.tolist(),
        "receiver_energy_ratios": np.divide(
            receiver_prediction_energy, receiver_energy,
            out=np.zeros_like(receiver_energy), where=receiver_energy > 0).tolist(),
        "active_receiver_ids": np.flatnonzero(active).tolist(),
    }


def main():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sweep, base, control, _, trajectory, parent, model_spec, train_paths = read_setup()
    if (ROOT / spec["weight_sweep_protocol"] != SWEEP_PROTOCOL
            or spec["arms"] != sweep["arms"] or spec["seeds"] != sweep["seeds"]
            or spec["weights"] != [0.0, 0.05] or 0.05 not in sweep["weights"]):
        raise ValueError("Amplitude audit differs from the registered comparison")
    output, report = ROOT / spec["result_json"], ROOT / spec["report_file"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite amplitude audit results")
    paths = [path for directory in trajectory["development_directories"]
             for path in sorted((ROOT / directory).glob("*.npz"))]
    if len(paths) != spec["development_event_count"]:
        raise ValueError("Development event count changed")
    events, receivers, times = load_observations(paths)
    training, train_receivers, train_times = load_observations(train_paths)
    if (not np.array_equal(receivers, train_receivers)
            or not np.array_equal(times, train_times)):
        raise ValueError("Development acquisition differs from training")
    if any(e["sha256"] == t["sha256"] or np.allclose(
            e["source"], t["source"], rtol=0, atol=1e-3)
           for e in events for t in training):
        raise ValueError("Development event overlaps a training event")
    field_path = ROOT / parent["target_fmm_fields"]
    fields = np.load(field_path, mmap_mode="r")
    field_hash = digest(field_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*model_spec["model"]["spatial_lengths_m"],
                         model_spec["model"]["time_length_s"]], dtype=np.float32)
    rows, checkpoints = [], []
    for arm in spec["arms"]:
        for seed in spec["seeds"]:
            initial = intermediate(trajectory, arm, seed, base["initial_target_updates"])
            initial_hash = digest(initial)
            for weight in spec["weights"]:
                path = (ROOT / control["checkpoint_directory"] /
                        f"{arm}_seed{seed}_restarted_control.pth" if weight == 0.0
                        else checkpoint(sweep, arm, seed, weight))
                payload = torch.load(path, map_location="cpu", weights_only=False)
                expected = digest(ROOT / base["control_protocol"] if weight == 0.0
                                  else SWEEP_PROTOCOL)
                if (payload["protocol_sha256"] != expected
                        or payload["initial_checkpoint_sha256"] != initial_hash
                        or payload["target_fmm_sha256"] != field_hash
                        or payload["arm"] != arm or payload["seed"] != seed):
                    raise ValueError(f"Checkpoint provenance mismatch: {path}")
                checkpoints.append({"arm": arm, "seed": seed, "weight": weight,
                                    "path": str(path.relative_to(ROOT)), "sha256": digest(path)})
                model = new_model(fields, receivers, model_spec, device)
                model.load_state_dict(payload["state_dict"])
                model.eval().requires_grad_(False)
                for event in events:
                    prediction = aligned_prediction(model, event, receivers, times, domain, device)
                    stats = amplitude_metrics(event["data"] / event["rms"], prediction,
                                              spec["active_receiver_energy_fraction"])
                    rows.append({"arm": arm, "seed": seed, "weight": weight,
                                 "event": str(event["path"].relative_to(ROOT)),
                                 "sha256": event["sha256"],
                                 "true_source_m_s": event["source"].tolist(), **stats})
                print(f"{arm} seed={seed} weight={weight}: {len(events)} events", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    metrics = ("raw_relative_l2", "global_gain_relative_l2",
               "receiver_gain_oracle_relative_l2", "prediction_to_observation_energy_ratio",
               "best_nonnegative_global_gain", "squared_error_removed_by_global_gain",
               "additional_squared_error_removed_by_receiver_gains",
               "remaining_normalized_squared_error")
    summaries = []
    for arm in spec["arms"]:
        for weight in spec["weights"]:
            group = [r for r in rows if r["arm"] == arm and r["weight"] == weight]
            summaries.append({"arm": arm, "weight": weight, "outcomes": len(group),
                              "means": {m: float(np.mean([r[m] for r in group])) for m in metrics},
                              "medians": {m: float(np.median([r[m] for r in group])) for m in metrics}})
    result = {"protocol_sha256": digest(PROTOCOL), "sweep_protocol_sha256": digest(SWEEP_PROTOCOL),
              "target_fmm_sha256": field_hash, "device": str(device),
              "development_event_count": len(events), "checkpoints": checkpoints,
              "summary": summaries, "rows": rows, "restriction": spec["restriction"]}
    lines = ["# Fixed-source amplitude and shape development audit", "",
             "All predictions use the true source position and origin time, with no time shift. "
             "Full waveforms at identical receiver/time samples share the reference event RMS. "
             "Nonnegative gains are fitted afterward for attribution only.", "",
             "| Path | Pair weight | Outcomes | Mean raw L2 | Mean global-gain L2 | "
             "Mean receiver-gain oracle L2 | Median predicted/reference energy |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        means, medians = row["means"], row["medians"]
        lines.append(f"| {row['arm']} | {row['weight']:.2f} | {row['outcomes']} | "
                     f"{means['raw_relative_l2']:.4f} | {means['global_gain_relative_l2']:.4f} | "
                     f"{means['receiver_gain_oracle_relative_l2']:.4f} | "
                     f"{medians['prediction_to_observation_energy_ratio']:.4f} |")
    lines.extend(["", "Energy ratios use squared waveform norms, not RMS or L2-norm ratios. "
                  "The receiver-gain oracle can absorb station-specific amplitude distortions; "
                  "its remaining residual still contains waveform-shape and timing error. "
                  "This audit alone cannot distinguish those two errors or attribute them "
                  "to a specific architectural cause.", "", spec["restriction"], "",
                  "Each group contains 3 seeds x 16 previously inspected development positions, "
                  "not 48 independent geologies. No weights, models or localization settings change."])
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
