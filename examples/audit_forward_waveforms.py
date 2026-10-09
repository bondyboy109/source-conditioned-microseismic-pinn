"""Compare a source-conditioned PINN with an independent FD receiver record.

The FD source truth is used only for this forward-model diagnostic, never
estimated here. A small training checkpoint is a smoke test, not validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source_conditioned_pinn import SourceConditionedConfig, load_conditioned_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "forward_audit.json")
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_conditioned_checkpoint(args.checkpoint, device)
    architecture = checkpoint.get("architecture", {})
    if (architecture.get("ray_travel_time", False)
            or architecture.get("kind") == "velocity_conditioned_travel_time_carrier"):
        velocity_path = ROOT / "data" / "velocity" / "vp_model.npy"
        model.set_velocity_grid(torch.as_tensor(
            np.array(np.load(velocity_path, mmap_mode="r"), dtype=np.float32),
            device=device))
    cfg = SourceConditionedConfig()
    expected = [cfg.x_length_m, cfg.y_length_m, cfg.z_length_m, cfg.time_length_s]
    stored = checkpoint["config"]
    if any(stored.get(k) != v for k, v in zip(
            ("x_length_m", "y_length_m", "z_length_m", "time_length_s"), expected)):
        raise ValueError("Checkpoint domain does not match current experiment")
    with np.load(args.observation) as archive:
        observed = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        truth = np.asarray(archive["source_truth"], dtype=np.float32)
    if observed.shape != (len(receivers), len(times)) or truth.shape != (4,):
        raise ValueError("Invalid observation dimensions or missing source truth")
    if not (np.isfinite(observed).all() and np.isfinite(receivers).all()
            and np.isfinite(times).all() and np.isfinite(truth).all()):
        raise ValueError("Observation contains nonfinite values")
    if times.min() < 0 or times.max() > cfg.time_length_s + 1e-5:
        raise ValueError("Observation time lies outside the trained PINN domain")

    scale = np.asarray(expected, dtype=np.float32)
    coords = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*observed.shape, 3)),
        np.broadcast_to(times[None, :, None], (*observed.shape, 1)),
    ), axis=-1).reshape(-1, 4) / scale
    source = torch.as_tensor(truth / scale, device=device)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(coords), args.chunk_size):
            xyz_t = torch.as_tensor(coords[start:start + args.chunk_size], device=device)
            src = source.expand(len(xyz_t), 4)
            predictions.append(model(xyz_t, src).squeeze(-1).cpu().numpy())
    predicted = np.concatenate(predictions).reshape(observed.shape).astype(np.float64)
    reference = observed.astype(np.float64)
    if not np.isfinite(predicted).all():
        raise ValueError("PINN prediction contains nonfinite values")
    energy = float(np.sum(reference ** 2))
    if energy == 0:
        raise ValueError("FD observations are all zero")
    dot = float(np.sum(reference * predicted))
    prediction_energy = float(np.sum(predicted ** 2))
    gain = dot / prediction_energy if prediction_energy > 0 else 0.0
    residual = predicted - reference
    adjusted = gain * predicted - reference
    per_receiver = np.sqrt(np.sum(residual ** 2, axis=1)
                           / np.maximum(np.sum(reference ** 2, axis=1), 1e-30))
    observation_sha256 = hashlib.sha256(args.observation.read_bytes()).hexdigest()
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_frequency_hz": checkpoint["frequency_hz"],
        "observation": str(args.observation),
        "observation_sha256": observation_sha256,
        "source_truth": truth.tolist(),
        "checkpoint_training_sources": checkpoint.get("training_record", {}).get(
            "receiver_observations", []),
        "source_in_receiver_training_set": any(
            np.allclose(truth, entry["source_truth"], rtol=0, atol=1e-3)
            for entry in checkpoint.get("training_record", {}).get(
                "receiver_observations", [])),
        "file_in_receiver_training_set": any(
            observation_sha256 == entry.get("sha256")
            for entry in checkpoint.get("training_record", {}).get(
                "receiver_observations", [])),
        "receiver_count": len(receivers),
        "sample_count": len(times),
        "fd_peak_abs": float(np.max(np.abs(reference))),
        "pinn_peak_abs": float(np.max(np.abs(predicted))),
        "zero_prediction_relative_l2": 1.0,
        "relative_l2": float(np.sqrt(np.sum(residual ** 2) / energy)),
        "optimal_global_amplitude_gain_diagnostic_only": float(gain),
        "relative_l2_after_global_amplitude_gain_diagnostic_only": float(
            np.sqrt(np.sum(adjusted ** 2) / energy)),
        "per_receiver_relative_l2": per_receiver.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"FD peak={report['fd_peak_abs']:.3e}; PINN peak={report['pinn_peak_abs']:.3e}")
    print(f"Relative L2={report['relative_l2']:.4g}; zero baseline=1.0")
    print(f"Amplitude-adjusted diagnostic L2={report['relative_l2_after_global_amplitude_gain_diagnostic_only']:.4g}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
