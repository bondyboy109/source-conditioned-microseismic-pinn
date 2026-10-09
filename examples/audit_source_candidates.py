"""Diagnostic ranking of a held-out source against known training sources.

This is not a localization benchmark: it includes the true source as an
explicit candidate and reports only whether the forward model ranks it first.
"""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(args.checkpoint, device)
    training = metadata.get("training_record", {}).get("receiver_observations", [])
    if not training:
        raise ValueError("Checkpoint does not record receiver training sources")
    config = metadata["config"]
    domain = np.array([config["x_length_m"], config["y_length_m"],
                       config["z_length_m"], config["time_length_s"]], dtype=np.float32)
    with np.load(args.observation) as archive:
        observed = np.asarray(archive["data"], dtype=np.float32)[:, ::args.stride]
        times = np.asarray(archive["times"], dtype=np.float32)[::args.stride]
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        truth = np.asarray(archive["source_truth"], dtype=np.float32)
    if observed.shape != (len(receivers), len(times)) or truth.shape != (4,):
        raise ValueError("Invalid observation geometry")
    if any(np.allclose(truth, e["source_truth"], atol=1e-3, rtol=0) for e in training):
        raise ValueError("True source is recorded in receiver training set")
    inputs = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*observed.shape, 3)),
        np.broadcast_to(times[None, :, None], (*observed.shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    st = torch.as_tensor(inputs, device=device)
    records = []
    model.eval()
    candidates = [("held_out_true", truth)] + [
        (f"training_source_{i:02d}", np.asarray(e["source_truth"], dtype=np.float32))
        for i, e in enumerate(training, start=1)
    ]
    with torch.inference_mode():
        for name, source in candidates:
            source_t = torch.as_tensor(source / domain, device=device)
            chunks = []
            for start in range(0, len(st), 8192):
                chunk = st[start:start + 8192]
                chunks.append(model(chunk, source_t.expand(len(chunk), 4)).squeeze(-1).cpu().numpy())
            prediction = np.concatenate(chunks).reshape(observed.shape).astype(np.float64)
            reference = observed.astype(np.float64)
            rel_l2 = float(np.linalg.norm(prediction - reference)
                           / np.linalg.norm(reference))
            records.append({"candidate": name, "source": source.tolist(),
                            "relative_l2": rel_l2,
                            "prediction_to_observation_energy": float(
                                np.linalg.norm(prediction) / np.linalg.norm(reference))})
    result = {"purpose": "fixed-candidate forward-model diagnostic only",
              "checkpoint": str(args.checkpoint), "observation": str(args.observation),
              "time_stride": args.stride, "candidates": records,
              "true_source_ranks_first": bool(records[0]["relative_l2"] ==
                                              min(r["relative_l2"] for r in records))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for record in records:
        print(f"{record['candidate']}: relative L2={record['relative_l2']:.4f}")
    print(f"True source ranks first: {result['true_source_ranks_first']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
