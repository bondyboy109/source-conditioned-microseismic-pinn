"""Post-hoc comparison of a fixed local FD interpolator and neural surrogate.

The script deliberately uses only already-spent records. It tests whether a
simple, fixed local fallback can improve the existing source-conditioned
forward model without retraining or generating new FD data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments" / "hybrid_local_neural_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_record(path: Path):
    with np.load(path) as archive:
        return (np.asarray(archive["data"], dtype=np.float64),
                np.asarray(archive["times"], dtype=np.float64),
                np.asarray(archive["receiver_coords"], dtype=np.float64),
                np.asarray(archive["source_truth"], dtype=np.float64))


def shift_origin(data: np.ndarray, times: np.ndarray,
                 training_origin: float, target_origin: float) -> np.ndarray:
    sample_times = times + training_origin - target_origin
    return np.asarray([
        np.interp(sample_times, times, trace, left=0.0, right=0.0)
        for trace in data
    ])


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    energy = float(np.sum(observed ** 2))
    active = np.abs(observed) >= 0.05 * np.max(np.abs(observed))
    active_energy = float(np.sum(observed[active] ** 2))
    return {
        "full_l2": float(np.sqrt(np.sum((predicted - observed) ** 2) / energy)),
        "active_l2": float(np.sqrt(np.sum((predicted[active] - observed[active]) ** 2)
                                   / max(active_energy, 1e-30))),
        "zero_lag_correlation": float(np.sum(predicted * observed)
                                       / max(np.linalg.norm(predicted)
                                             * np.linalg.norm(observed), 1e-30)),
    }


def interpolate(training: list[dict], target_source: np.ndarray,
                 times: np.ndarray, spec: dict) -> tuple[np.ndarray, float, list[dict]]:
    sources = np.asarray([row["source"] for row in training])
    scale = np.asarray([spec["xy_scale_m"], spec["xy_scale_m"],
                        spec["z_scale_m"]])
    distances = np.linalg.norm((sources[:, :3] - target_source[:3]) / scale, axis=1)
    order = np.argsort(distances, kind="stable")[:int(spec["neighbour_count"])]
    if distances[order[0]] <= 1e-12:
        order = order[:1]
        weights = np.ones(1)
    else:
        weights = distances[order] ** (-float(spec["inverse_distance_power"]))
        weights /= np.sum(weights)
    prediction = np.zeros_like(training[0]["data"])
    neighbours = []
    for index, weight in zip(order, weights):
        row = training[int(index)]
        prediction += float(weight) * shift_origin(
            row["data"], times, row["source"][3], target_source[3])
        neighbours.append({"source_m_s": row["source"].tolist(),
                           "scaled_distance": float(distances[index]),
                           "weight": float(weight)})
    return prediction, float(distances[order[0]]), neighbours


def neural_predict(model, source: np.ndarray, data_shape: tuple[int, int],
                   times: np.ndarray, receivers: np.ndarray, scale: np.ndarray,
                   device: torch.device) -> np.ndarray:
    coords = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*data_shape, 3)),
        np.broadcast_to(times[None, :, None], (*data_shape, 1)),
    ), axis=-1).reshape(-1, 4) / scale
    source_t = torch.as_tensor(source / scale, dtype=torch.float32, device=device)
    output = []
    with torch.inference_mode():
        for start in range(0, len(coords), 8192):
            batch = torch.as_tensor(coords[start:start + 8192], dtype=torch.float32,
                                    device=device)
            output.append(model(batch, source_t.expand(len(batch), 4))
                          .squeeze(-1).cpu().numpy())
    return np.concatenate(output).reshape(data_shape).astype(np.float64)


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["result_file"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoint = ROOT / spec["training_checkpoint"]
    if sha256(checkpoint) != spec["training_checkpoint_sha256"]:
        raise ValueError("Registered checkpoint changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_conditioned_checkpoint(checkpoint, device)
    model.eval()
    manifest = payload.get("training_record", {}).get("receiver_observations", [])
    if len(manifest) != 33:
        raise ValueError(f"Expected 33 registered training records, got {len(manifest)}")
    training = []
    ref_times = ref_receivers = None
    for entry in manifest:
        path = Path(entry["path"])
        data, times, receivers, source = load_record(path)
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Training record changed: {path}")
        if ref_times is None:
            ref_times, ref_receivers = times, receivers
        elif not (np.array_equal(times, ref_times) and np.array_equal(receivers, ref_receivers)):
            raise ValueError("Training geometry differs")
        training.append({"data": data, "source": source})

    targets = []
    neural_lookup = {}
    for item in spec["historical_results"]:
        result_path = ROOT / item["path"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for event in result["events"]:
            obs = Path(event.get("observation", ""))
            if not obs.is_absolute():
                obs = ROOT / item["data_directory"] / event["event"]
            obs = obs.resolve()
            if not obs.exists():
                obs = ROOT / item["data_directory"] / event["event"]
            data, times, receivers, source = load_record(obs)
            if not (np.array_equal(times, ref_times) and np.array_equal(receivers, ref_receivers)):
                raise ValueError(f"Target geometry differs: {obs}")
            key = str(obs)
            base = event.get("pinn", event.get("base", {}))
            neural_lookup[key] = base
            targets.append({"event": event["event"], "observation": str(obs),
                            "sha256": sha256(obs), "data": data, "times": times,
                            "receivers": receivers, "source": source,
                            "neural_reference": base})
    if len(targets) != 16:
        raise ValueError(f"Expected 16 historical targets, got {len(targets)}")

    cfg_scale = np.asarray([payload["config"][k] for k in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
    events = []
    for target in targets:
        local, nearest, neighbours = interpolate(training, target["source"],
                                                 ref_times, spec)
        neural = neural_predict(model, target["source"], target["data"].shape,
                                target["times"], target["receivers"], cfg_scale, device)
        observed = target["data"]
        local_metrics = metrics(observed, local)
        neural_metrics = metrics(observed, neural)
        use_local = nearest <= float(spec["local_use_threshold_scaled_distance"])
        hybrid = local if use_local else neural
        hybrid_metrics = metrics(observed, hybrid)
        events.append({"event": target["event"], "observation": target["observation"],
                       "sha256": target["sha256"], "source_m_s": target["source"].tolist(),
                       "nearest_scaled_distance": nearest, "use_local": use_local,
                       "local": local_metrics, "neural": neural_metrics,
                       "hybrid": hybrid_metrics, "neighbours": neighbours})

    def mean(name: str, metric: str) -> float:
        return float(np.mean([row[name][metric] for row in events]))

    summary = {
        "event_count": len(events),
        "local_mean_full_l2": mean("local", "full_l2"),
        "neural_mean_full_l2": mean("neural", "full_l2"),
        "hybrid_mean_full_l2": mean("hybrid", "full_l2"),
        "local_mean_active_l2": mean("local", "active_l2"),
        "neural_mean_active_l2": mean("neural", "active_l2"),
        "hybrid_mean_active_l2": mean("hybrid", "active_l2"),
        "hybrid_improves_over_neural_count": int(sum(
            row["hybrid"]["full_l2"] < row["neural"]["full_l2"] for row in events)),
        "local_selected_count": int(sum(row["use_local"] for row in events)),
    }
    result = {"role": spec["role"], "protocol_sha256": sha256(PROTOCOL),
              "checkpoint_sha256": spec["training_checkpoint_sha256"],
              "summary": summary, "events": events, "limits": spec["limits"]}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
