"""Diagnostic: compare source-time shifts against shifted training FD traces.

Zero-padded integer-sample delays are an approximate control: truncation of
the nonzero Ricker tail before the original recording starts means they must
not replace newly generated FD records. This is not independent validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_forward_error_decomposition import predict
from source_conditioned_pinn import load_conditioned_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--shift-s", type=float, nargs="+", default=[0.04, 0.08, 0.16])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(shift <= 0 for shift in args.shift_s):
        parser.error("Each source-time delay must be positive")

    with np.load(args.observation) as archive:
        times = np.asarray(archive["times"], dtype=np.float32)
        data = np.asarray(archive["data"], dtype=np.float64)
        source = np.asarray(archive["source_truth"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
    if data.shape != (len(receivers), len(times)) or source.shape != (4,):
        raise ValueError("Invalid observation geometry")
    dt = np.diff(times.astype(np.float64))
    if not np.allclose(dt, dt[0], rtol=1e-4, atol=1e-7):
        raise ValueError("Time samples must have uniform spacing")
    if np.any(~np.isfinite(data)) or np.sum(data ** 2) <= 0:
        raise ValueError("Reference waveforms must be finite and nonzero")

    original, baseline, _, _ = predict(args.checkpoint, args.observation, 8192)
    baseline_l2 = float(np.linalg.norm(baseline - original) / np.linalg.norm(original))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(args.checkpoint, device)
    cfg = metadata["config"]
    scale = np.asarray([cfg["x_length_m"], cfg["y_length_m"],
                        cfg["z_length_m"], cfg["time_length_s"]], dtype=np.float32)
    coordinates = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*data.shape, 3)),
        np.broadcast_to(times[None, :, None], (*data.shape, 1)),
    ), axis=-1).reshape(-1, 4) / scale
    model.eval()
    results = []
    for shift_s in args.shift_s:
        n = int(round(shift_s / dt[0]))
        if n <= 0 or n >= len(times) or not np.isclose(n * dt[0], shift_s, atol=1e-5):
            raise ValueError("Each delay must be an integer number of samples in the record")
        shifted = np.zeros_like(data)
        shifted[:, n:] = data[:, :-n]
        shifted_source = source.copy()
        shifted_source[3] += n * dt[0]
        if shifted_source[3] > cfg["source_time_max"] * cfg["time_length_s"]:
            raise ValueError("Delayed source time falls outside checkpoint bounds")
        src = torch.as_tensor(shifted_source / scale, device=device)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(coordinates), 8192):
                points = torch.as_tensor(coordinates[start:start + 8192], device=device)
                chunks.append(model(points, src.expand(len(points), 4))
                              .squeeze(-1).cpu().numpy())
        pred = np.concatenate(chunks).reshape(shifted.shape).astype(np.float64)
        l2 = float(np.linalg.norm(pred - shifted) / np.linalg.norm(shifted))
        results.append({"shift_s": float(n * dt[0]), "samples": n,
                        "shifted_source": shifted_source.tolist(),
                        "raw_relative_l2": l2,
                        "peak_ratio": float(np.max(np.abs(pred)) / np.max(np.abs(shifted)))})
        print(f"+{n * dt[0]:.3f} s: raw L2={l2:.4f}", flush=True)
    report = {"purpose": "training-source time-conditioning diagnostic; no model selection",
              "checkpoint": str(args.checkpoint), "observation": str(args.observation),
              "source": source.tolist(), "baseline_raw_relative_l2": baseline_l2,
              "reference": "zero-padded integer-sample shift of original FD recording",
              "delays": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
