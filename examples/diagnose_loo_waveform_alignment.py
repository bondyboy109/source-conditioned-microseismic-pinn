"""Inspect timing and waveform overlap for development-only LOO checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source_conditioned_pinn import load_conditioned_checkpoint


def trace_alignment(reference: np.ndarray, predicted: np.ndarray,
                    max_shift_samples: int) -> dict:
    reference_norm = np.linalg.norm(reference)
    predicted_norm = np.linalg.norm(predicted)
    if reference_norm == 0 or predicted_norm == 0:
        raise ValueError("Cannot correlate a zero-energy trace")
    denominator = reference_norm * predicted_norm
    unshifted = float(np.dot(reference, predicted) / denominator)
    # A positive lag delays the prediction. Norms retain the full-record
    # energy so a short overlap cannot achieve an artificially perfect score.
    best_shift, best_cosine = 0, unshifted
    for shift in range(-max_shift_samples, max_shift_samples + 1):
        if shift < 0:
            overlap = np.dot(reference[:shift], predicted[-shift:])
        elif shift > 0:
            overlap = np.dot(reference[shift:], predicted[:-shift])
        else:
            overlap = np.dot(reference, predicted)
        cosine = float(overlap / denominator)
        if abs(cosine) > abs(best_cosine):
            best_shift, best_cosine = shift, cosine
    return {"unshifted_cosine": unshifted,
            "best_signed_cosine": best_cosine,
            "best_shift_samples": best_shift,
            "best_gain_and_shift_relative_l2_diagnostic": float(
                np.sqrt(max(0.0, 1.0 - best_cosine ** 2)))}


def predict(model: torch.nn.Module, observation: Path,
            device: torch.device, domain: np.ndarray):
    with np.load(observation) as archive:
        observed = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        source = np.asarray(archive["source_truth"], dtype=np.float32)
    if observed.shape != (len(receivers), len(times)) or source.shape != (4,):
        raise ValueError(f"Invalid observation: {observation}")
    dt = np.diff(times.astype(np.float64))
    if not np.isfinite(observed).all() or not np.allclose(dt, dt[0], atol=1e-6):
        raise ValueError(f"Invalid time axis or waveform: {observation}")
    coords = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*observed.shape, 3)),
        np.broadcast_to(times[None, :, None], (*observed.shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    source_t = torch.as_tensor(source / domain, device=device)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coords), 8192):
            points = torch.as_tensor(coords[start:start + 8192], device=device)
            predictions.append(model(points, source_t.expand(len(points), 4))
                               .squeeze(-1).cpu().numpy())
    predicted = np.concatenate(predictions).reshape(observed.shape)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite model prediction")
    return observed.astype(np.float64), predicted.astype(np.float64), float(dt[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=Path,
                        default=ROOT / "outputs" / "dev_loo_relative_carrier_5hz")
    parser.add_argument("--max-shift-s", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.max_shift_s <= 1:
        parser.error("--max-shift-s must lie in (0, 1] seconds")

    folds = args.folds.resolve()
    summary = json.loads((folds / "summary.json").read_text(encoding="utf-8"))
    if summary["completed_folds"] != 8 or len(summary["results"]) != 8:
        raise ValueError("Eight completed LOO folds are required")
    registered = json.loads((ROOT / "experiments" / "5hz_protocol.json").read_text(
        encoding="utf-8"))["data"]["training_sources"]
    if [r["held_out"] for r in summary["results"]] != [s["file"] for s in registered]:
        raise ValueError("LOO fold names differ from registered training data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for index, row in enumerate(summary["results"], 1):
        fold = folds / f"leave_{index:02d}"
        model, checkpoint = load_conditioned_checkpoint(
            fold / "conditioned_5Hz.pth", device)
        if checkpoint.get("training_record", {}).get("training_type") != "supervised_receiver_diagnostic":
            raise ValueError("Expected supervised LOO diagnostic checkpoint")
        cfg = checkpoint["config"]
        domain = np.asarray([cfg["x_length_m"], cfg["y_length_m"],
                             cfg["z_length_m"], cfg["time_length_s"]],
                            dtype=np.float32)
        observed, predicted, dt = predict(model, ROOT / "data" / "observations"
                                          / row["held_out"], device, domain)
        shift_samples = int(round(args.max_shift_s / dt))
        alignment = [trace_alignment(ref, pred, shift_samples)
                     for ref, pred in zip(observed, predicted)]
        zero = np.asarray([a["unshifted_cosine"] for a in alignment])
        best = np.asarray([a["best_signed_cosine"] for a in alignment])
        delay = np.asarray([a["best_shift_samples"] * dt for a in alignment])
        record = {"held_out": row["held_out"],
                  "global_relative_l2": row["relative_l2"],
                  "max_shift_s": shift_samples * dt,
                  "median_unshifted_abs_cosine": float(np.median(np.abs(zero))),
                  "median_best_shifted_abs_cosine": float(np.median(np.abs(best))),
                  "median_absolute_best_shift_s": float(np.median(np.abs(delay))),
                  "receivers_best_shifted_abs_cosine_above_0_5": int(
                      np.count_nonzero(np.abs(best) >= 0.5)),
                  "receivers": [{**entry,
                                 "best_shift_s": float(entry["best_shift_samples"] * dt)}
                                for entry in alignment]}
        records.append(record)
        print(f"{row['held_out']}: median |cos| 0-lag={record['median_unshifted_abs_cosine']:.3f}, "
              f"shifted={record['median_best_shifted_abs_cosine']:.3f}, "
              f"aligned receivers={record['receivers_best_shifted_abs_cosine_above_0_5']}/25",
              flush=True)
    output = folds / "waveform_alignment.json"
    output.write_text(json.dumps({
        "purpose": "development-only timing and shape diagnostic; shifts/gains use FD truth",
        "max_shift_s": args.max_shift_s, "events": records,
    }, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
