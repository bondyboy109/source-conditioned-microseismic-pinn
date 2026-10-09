"""Decompose frozen-checkpoint waveform error on registered development events."""

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

from source_conditioned_pinn import load_conditioned_checkpoint


def predict(checkpoint: Path, observation: Path, chunk_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(checkpoint, device)
    cfg = metadata["config"]
    scale = np.asarray([cfg["x_length_m"], cfg["y_length_m"],
                        cfg["z_length_m"], cfg["time_length_s"]], dtype=np.float32)
    with np.load(observation) as archive:
        observed = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        source = np.asarray(archive["source_truth"], dtype=np.float32)
    if observed.shape != (len(receivers), len(times)) or source.shape != (4,):
        raise ValueError(f"Invalid observation data: {observation}")
    coordinates = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*observed.shape, 3)),
        np.broadcast_to(times[None, :, None], (*observed.shape, 1)),
    ), axis=-1).reshape(-1, 4) / scale
    src = torch.as_tensor(source / scale, device=device)
    predicted = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coordinates), chunk_size):
            points = torch.as_tensor(coordinates[start:start + chunk_size], device=device)
            predicted.append(model(points, src.expand(len(points), 4))
                             .squeeze(-1).cpu().numpy())
    prediction = np.concatenate(predicted).reshape(observed.shape).astype(np.float64)
    return observed.astype(np.float64), prediction, times.astype(np.float64), source


def shifted(array: np.ndarray, samples: int) -> np.ndarray:
    result = np.zeros_like(array)
    if samples > 0:
        result[:, samples:] = array[:, :-samples]
    elif samples < 0:
        result[:, :samples] = array[:, -samples:]
    else:
        result[:] = array
    return result


def best_common_shift(observed: np.ndarray, predicted: np.ndarray,
                      max_shift_samples: int) -> dict:
    energy = float(np.sum(observed ** 2))
    best = None
    for lag in range(-max_shift_samples, max_shift_samples + 1):
        candidate = shifted(predicted, lag)
        pred_energy = float(np.sum(candidate ** 2))
        dot = float(np.sum(observed * candidate))
        gain = dot / pred_energy if pred_energy > 0 else 0.0
        residual = observed - gain * candidate
        score = float(np.sum(residual ** 2) / energy)
        if best is None or score < best["relative_l2"] ** 2:
            best = {"lag_samples": lag, "gain": gain,
                    "relative_l2": float(np.sqrt(score))}
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("observations", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-label", default="development",
                        choices=("development", "training-diagnostic",
                                 "independent-reaudit", "synthetic-confirmation"))
    parser.add_argument("--max-shift-s", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    if args.max_shift_s <= 0 or args.chunk_size < 1:
        parser.error("Shift window and chunk size must be positive")
    model, checkpoint_metadata = load_conditioned_checkpoint(
        args.checkpoint, torch.device("cpu"))
    del model
    training = checkpoint_metadata.get("training_record", {}).get(
        "receiver_observations", [])
    training_hashes = {item.get("sha256") for item in training}
    training_sources = [item.get("source_truth") for item in training]
    windows = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0)]
    events = []
    for path in args.observations:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observed, predicted, times, source = predict(
            args.checkpoint, path, args.chunk_size)
        if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
            raise ValueError(f"Nonfinite waveforms in {path}")
        total_energy = float(np.sum(observed ** 2))
        full_relative_l2 = float(np.sqrt(np.sum((predicted - observed) ** 2)
                                         / total_energy))
        peak_ratio = float(np.max(np.abs(predicted))
                           / max(np.max(np.abs(observed)), 1e-30))
        window_rows = []
        for start, stop in windows:
            selected = (times >= start) & (times < stop)
            if not np.any(selected):
                continue
            ref, pred = observed[:, selected], predicted[:, selected]
            observed_window_energy = float(np.sum(ref ** 2))
            window_rows.append({
                "window_s": [start, stop],
                "observed_energy_fraction": float(observed_window_energy / total_energy),
                "relative_l2_normalized_by_window_energy": (
                    float(np.sqrt(np.sum((pred - ref) ** 2) / observed_window_energy))
                    if observed_window_energy > total_energy * 1e-4 else None),
                "window_energy_too_small_for_local_relative_l2":
                    observed_window_energy <= total_energy * 1e-4,
                "error_energy_fraction_of_full_record": float(
                    np.sum((pred - ref) ** 2) / total_energy),
            })
        obs_receiver = np.linalg.norm(observed, axis=1)
        pred_receiver = np.linalg.norm(predicted, axis=1)
        active = obs_receiver > max(float(obs_receiver.max()) * 1e-3, 1e-30)
        ratios = pred_receiver[active] / obs_receiver[active]
        best = best_common_shift(observed, predicted,
                                 int(round(args.max_shift_s / np.median(np.diff(times)))))
        events.append({
            "observation": str(path), "sha256": digest,
            "file_in_receiver_training_set": digest in training_hashes,
            "source_truth": source.tolist(),
            "source_in_receiver_training_set": any(
                item is not None and np.allclose(source, item, rtol=0, atol=1e-3)
                for item in training_sources),
            "receiver_count": int(len(observed)), "sample_count": int(len(times)),
            "original_amplitude_relative_l2": full_relative_l2,
            "prediction_to_observation_peak_ratio": peak_ratio,
            "active_receiver_count": int(active.sum()),
            "active_receiver_energy_ratio_median": float(np.median(ratios)),
            "active_receiver_energy_ratio_p10_p90": [
                float(np.quantile(ratios, 0.1)), float(np.quantile(ratios, 0.9))],
            "common_time_shift_and_gain_diagnostic_only": {
                **best, "lag_s": float(best["lag_samples"] * np.median(np.diff(times)))},
            "time_windows": window_rows,
        })
        print(f"{path.name}: raw L2={full_relative_l2:.4f}; peak ratio={peak_ratio:.3f}; "
              f"receiver energy ratio median={np.median(ratios):.3f}; "
              f"shift+gain L2={best['relative_l2']:.4f} at {best['lag_samples'] * np.median(np.diff(times)):.3f}s",
              flush=True)
    result = {
        "purpose": ("frozen waveform error decomposition; no fitting or tuning; "
                    + args.evaluation_label),
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint_metadata["epoch"],
        "checkpoint_training_events": [item.get("path") for item in training],
        "time_windows_s": [list(row) for row in windows],
        "best_shift_limit_s": args.max_shift_s,
        "events": events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
