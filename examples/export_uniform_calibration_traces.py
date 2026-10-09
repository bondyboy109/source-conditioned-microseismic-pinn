"""Export the same energetic receiver from uniform train and test events."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source_conditioned_pinn import load_conditioned_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_conditioned_checkpoint(args.checkpoint, device)
    training = checkpoint.get("training_record", {})
    if training.get("training_type") != "supervised_receiver_diagnostic":
        raise ValueError("Expected supervised uniform-medium diagnostic")
    cfg = checkpoint["config"]
    scale = np.asarray([cfg["x_length_m"], cfg["y_length_m"],
                        cfg["z_length_m"], cfg["time_length_s"]], dtype=np.float32)
    events = {}
    for name in ("train", "test"):
        path = ROOT / "data" / "observations" / "uniform_3500" / f"{name}_0to2s.npz"
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            source = np.asarray(archive["source_truth"], dtype=np.float32)
        if data.shape != (25, 1001):
            raise ValueError(f"Unexpected observation shape: {path}")
        recorded = [entry["source_truth"] for entry in training["receiver_observations"]]
        belongs = any(np.allclose(source, member, atol=1e-3, rtol=0)
                      for member in recorded)
        if belongs != (name == "train"):
            raise ValueError(f"Incorrect training split for {name}")
        events[name] = (data, times, receivers, source)

    if not np.array_equal(events["train"][2], events["test"][2]):
        raise ValueError("Station geometry differs between events")
    index = int(np.argmax(np.sum(events["test"][0].astype(np.float64) ** 2, axis=1)))
    result = {"receiver_index": np.int32(index),
              "receiver_m": events["test"][2][index]}
    model.eval()
    with torch.inference_mode():
        for name, (data, times, receivers, source) in events.items():
            points = np.column_stack((
                np.broadcast_to(receivers[index], (len(times), 3)), times)) / scale
            point_t = torch.as_tensor(points, device=device)
            source_t = torch.as_tensor(source / scale, device=device)
            prediction = model(point_t, source_t.expand(len(times), 4))
            result[f"{name}_times_s"] = times
            result[f"{name}_fd"] = data[index]
            result[f"{name}_predicted"] = prediction.squeeze(-1).cpu().numpy()
            result[f"{name}_source"] = source
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"Saved receiver {index} at {result['receiver_m'].tolist()} m: {args.output}")


if __name__ == "__main__":
    main()
