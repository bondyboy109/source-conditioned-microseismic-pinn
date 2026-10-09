"""Compare finite source-displacement responses of FD data and a frozen PINN.

This is a development diagnostic of spatial sensitivity, not an estimate of
localization precision or an infinitesimal source-coordinate derivative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_forward_error_decomposition import predict
from source_conditioned_pinn import load_conditioned_checkpoint


def response_metrics(reference: np.ndarray, prediction: np.ndarray,
                     reference_change: np.ndarray, prediction_change: np.ndarray) -> dict:
    fd_norm = float(np.linalg.norm(reference_change))
    model_norm = float(np.linalg.norm(prediction_change))
    if fd_norm <= 0 or model_norm <= 0:
        raise ValueError("Both FD and model changes must have nonzero energy")
    return {
        "reference_change_relative_to_original": fd_norm / float(np.linalg.norm(reference)),
        "model_change_relative_to_original": model_norm / float(np.linalg.norm(prediction)),
        "model_to_fd_change_norm_ratio": model_norm / fd_norm,
        "change_cosine": float(np.sum(reference_change * prediction_change)
                               / (fd_norm * model_norm)),
        "change_relative_l2": float(np.linalg.norm(prediction_change - reference_change)
                                    / fd_norm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("reference", type=Path, help="FD recording at a training source")
    parser.add_argument("displaced", type=Path, nargs="+", help="FD development events")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _, meta = load_conditioned_checkpoint(args.checkpoint, torch.device("cpu"))
    recorded = meta.get("training_record", {}).get("receiver_observations", [])
    hashes = {row.get("sha256") for row in recorded}
    training_sources = [row.get("source_truth") for row in recorded]
    original, original_pred, times, source = predict(args.checkpoint, args.reference, 8192)
    if hashlib.sha256(args.reference.read_bytes()).hexdigest() not in hashes:
        raise ValueError("Reference FD event must appear in the checkpoint training set")
    with np.load(args.reference) as archive:
        stations = archive["receiver_coords"]
    results = []
    for path in args.displaced:
        if hashlib.sha256(path.read_bytes()).hexdigest() in hashes:
            raise ValueError(f"Displaced event appears in the checkpoint training set: {path}")
        with np.load(path) as archive:
            if not (np.array_equal(archive["times"], times.astype(np.float32))
                    and np.array_equal(archive["receiver_coords"], stations)):
                raise ValueError(f"Geometry or times changed: {path}")
        fd, model, _, new_source = predict(args.checkpoint, path, 8192)
        if any(item is not None and np.allclose(new_source, item, atol=1e-3, rtol=0)
               for item in training_sources):
            raise ValueError(f"Displaced source matches a training source: {path}")
        if not np.isclose(new_source[3], source[3], atol=1e-6, rtol=0):
            raise ValueError("Source origin time must remain unchanged")
        shift = new_source[:3] - source[:3]
        if np.count_nonzero(np.abs(shift) > 1e-3) != 1:
            raise ValueError("Change exactly one source spatial coordinate")
        displacement = float(np.linalg.norm(shift))
        metrics = response_metrics(original, original_pred,
                                   fd - original, model - original_pred)
        metrics.update({"observation": str(path), "displacement_m": displacement,
                        "displacement_xyz_m": shift.tolist(),
                        "raw_waveform_relative_l2": float(np.linalg.norm(model - fd)
                                                          / np.linalg.norm(fd)),
                        "prediction_to_fd_peak_ratio": float(np.max(np.abs(model))
                                                             / np.max(np.abs(fd)))})
        results.append(metrics)
        print(f"shift {shift.tolist()} m: waveform L2={metrics['raw_waveform_relative_l2']:.4f}, "
              f"response cosine={metrics['change_cosine']:.4f}, "
              f"response norm ratio={metrics['model_to_fd_change_norm_ratio']:.4f}",
              flush=True)
    report = {
        "purpose": "frozen-model development audit of finite spatial source response",
        "checkpoint": str(args.checkpoint), "reference": str(args.reference),
        "source": source.tolist(), "reference_training_sha256":
            hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
