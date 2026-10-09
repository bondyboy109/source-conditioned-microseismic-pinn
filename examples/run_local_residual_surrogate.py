"""Learn a source-dependent correction to local FD waveform interpolation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments" / "local_residual_surrogate_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_record(path: Path):
    with np.load(path) as archive:
        return (np.asarray(archive["data"], dtype=np.float32),
                np.asarray(archive["times"], dtype=np.float32),
                np.asarray(archive["receiver_coords"], dtype=np.float32),
                np.asarray(archive["source_truth"], dtype=np.float32))


def shift_origin(data: np.ndarray, times: np.ndarray,
                 training_origin: float, target_origin: float) -> np.ndarray:
    sample_times = times + training_origin - target_origin
    return np.asarray([np.interp(sample_times, times, trace, left=0.0, right=0.0)
                       for trace in data], dtype=np.float32)


def interpolate(training: list[dict], target: np.ndarray, times: np.ndarray,
                spec: dict, exclude: int | None = None) -> tuple[np.ndarray, float]:
    sources = np.asarray([row["source"] for row in training])
    scale = np.asarray([spec["xy_scale_m"], spec["xy_scale_m"], spec["z_scale_m"]])
    distances = np.linalg.norm((sources[:, :3] - target[:3]) / scale, axis=1)
    if exclude is not None:
        distances[exclude] = np.inf
    order = np.argsort(distances, kind="stable")[:int(spec["neighbour_count"])]
    weights = distances[order] ** (-float(spec["inverse_distance_power"]))
    weights /= np.sum(weights)
    prediction = np.zeros_like(training[0]["data"])
    for index, weight in zip(order, weights):
        row = training[int(index)]
        prediction += float(weight) * shift_origin(
            row["data"], times, float(row["source"][3]), float(target[3]))
    return prediction, float(distances[order[0]])


class ResidualSurrogate(nn.Module):
    def __init__(self, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(9, hidden_dim), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def features(receivers: np.ndarray, times: np.ndarray, source: np.ndarray,
             baseline: np.ndarray, domain: np.ndarray, waveform_scale: float) -> np.ndarray:
    shape = baseline.shape
    receiver_time = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*shape, 3)),
        np.broadcast_to(times[None, :, None], (*shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    source_values = np.broadcast_to(source / domain, (len(receiver_time), 4))
    baseline_value = baseline.reshape(-1, 1) * waveform_scale
    return np.concatenate((receiver_time, source_values, baseline_value), axis=1).astype(np.float32)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed64, predicted64 = observed.astype(np.float64), predicted.astype(np.float64)
    energy = float(np.sum(observed64 ** 2))
    active = np.abs(observed64) >= 0.05 * np.max(np.abs(observed64))
    active_energy = float(np.sum(observed64[active] ** 2))
    return {
        "full_l2": float(np.sqrt(np.sum((predicted64 - observed64) ** 2) / energy)),
        "active_l2": float(np.sqrt(np.sum((predicted64[active] - observed64[active]) ** 2)
                                   / max(active_energy, 1e-30))),
        "zero_lag_correlation": float(np.sum(predicted64 * observed64)
                                       / max(np.linalg.norm(predicted64)
                                             * np.linalg.norm(observed64), 1e-30)),
    }


def neural_prediction(model, observed_shape: tuple[int, int], receivers: np.ndarray,
                      times: np.ndarray, source: np.ndarray, domain: np.ndarray,
                      device: torch.device) -> np.ndarray:
    coords = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*observed_shape, 3)),
        np.broadcast_to(times[None, :, None], (*observed_shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    source_t = torch.as_tensor(source / domain, device=device)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coords), 8192):
            x = torch.as_tensor(coords[start:start + 8192], device=device)
            chunks.append(model(x, source_t.expand(len(x), 4)).squeeze(-1).cpu().numpy())
    return np.concatenate(chunks).reshape(observed_shape).astype(np.float32)


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path = ROOT / spec["result_file"]
    output_dir = ROOT / spec["output_directory"]
    checkpoint_path = output_dir / "residual_surrogate.pth"
    if result_path.exists() or checkpoint_path.exists():
        raise FileExistsError("Refusing to overwrite existing residual-surrogate outputs")
    base_path = ROOT / spec["training_checkpoint"]
    if sha256(base_path) != spec["training_checkpoint_sha256"]:
        raise ValueError("Registered training checkpoint changed")

    seed = int(spec["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, payload = load_conditioned_checkpoint(base_path, device)
    domain = np.asarray([payload["config"][key] for key in
                         ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                        dtype=np.float32)
    manifest = payload.get("training_record", {}).get("receiver_observations", [])
    if len(manifest) != 33:
        raise ValueError(f"Expected 33 training records, got {len(manifest)}")
    training = []
    reference_times = reference_receivers = None
    for row in manifest:
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Training record changed: {path}")
        data, times, receivers, source = load_record(path)
        if reference_times is None:
            reference_times, reference_receivers = times, receivers
        elif not (np.array_equal(times, reference_times)
                  and np.array_equal(receivers, reference_receivers)):
            raise ValueError("Training observation geometry differs")
        training.append({"path": path, "data": data, "source": source})

    feature_blocks, residual_blocks, active_blocks = [], [], []
    training_baseline_l2 = []
    for index, row in enumerate(training):
        baseline, _ = interpolate(training, row["source"], reference_times, spec,
                                  exclude=index)
        feature_blocks.append(features(reference_receivers, reference_times,
                                       row["source"], baseline, domain,
                                       float(spec["waveform_scale"])))
        residual = (row["data"] - baseline).reshape(-1, 1) * float(spec["waveform_scale"])
        residual_blocks.append(residual.astype(np.float32))
        threshold = 0.05 * np.max(np.abs(row["data"]))
        active_blocks.append(np.abs(row["data"].reshape(-1)) >= threshold)
        training_baseline_l2.append(metrics(row["data"], baseline)["full_l2"])
    all_features = np.concatenate(feature_blocks)
    all_residuals = np.concatenate(residual_blocks)
    active = np.flatnonzero(np.concatenate(active_blocks))
    if not np.isfinite(all_features).all() or not np.isfinite(all_residuals).all():
        raise ValueError("Nonfinite residual training data")

    model = ResidualSurrogate(int(spec["hidden_dim"]), int(spec["hidden_layers"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
    rng = np.random.default_rng(seed)
    batch_size = int(spec["batch_size"])
    half = batch_size // 2
    history = []
    started = time.perf_counter()
    for update in range(1, int(spec["training_updates"]) + 1):
        indices = np.concatenate((rng.choice(active, half, replace=True),
                                  rng.integers(0, len(all_features), batch_size - half)))
        x = torch.as_tensor(all_features[indices], device=device)
        y = torch.as_tensor(all_residuals[indices], device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.mean((prediction - y) ** 2)
        loss.backward()
        optimizer.step()
        if update == 1 or update % 200 == 0:
            value = float(loss.detach().cpu())
            history.append({"update": update, "loss": value})
            print(f"update {update}/{spec['training_updates']}: loss={value:.6e}", flush=True)
    training_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "spec": spec,
                "domain": domain.tolist(), "training_history": history}, checkpoint_path)

    targets = []
    for item in spec["historical_results"]:
        historical = json.loads((ROOT / item["path"]).read_text(encoding="utf-8"))
        for event in historical["events"]:
            path = Path(event.get("observation", ""))
            if not path.is_absolute() or not path.exists():
                path = ROOT / item["data_directory"] / event["event"]
            data, times, receivers, source = load_record(path)
            targets.append({"event": event["event"], "path": path, "data": data,
                            "times": times, "receivers": receivers, "source": source})
    if len(targets) != 16:
        raise ValueError(f"Expected 16 historical targets, got {len(targets)}")

    events = []
    model.eval()
    for target in targets:
        baseline, nearest = interpolate(training, target["source"], target["times"], spec)
        x = features(target["receivers"], target["times"], target["source"],
                     baseline, domain, float(spec["waveform_scale"]))
        corrections = []
        with torch.inference_mode():
            for start in range(0, len(x), 8192):
                corrections.append(model(torch.as_tensor(x[start:start + 8192], device=device))
                                   .squeeze(-1).cpu().numpy())
        corrected = baseline + np.concatenate(corrections).reshape(baseline.shape) / float(
            spec["waveform_scale"])
        neural = neural_prediction(base_model, target["data"].shape, target["receivers"],
                                   target["times"], target["source"], domain, device)
        event = {"event": target["event"], "observation": str(target["path"]),
                 "sha256": sha256(target["path"]),
                 "source_m_s": target["source"].tolist(),
                 "nearest_scaled_training_distance": nearest,
                 "local_interpolation": metrics(target["data"], baseline),
                 "residual_surrogate": metrics(target["data"], corrected),
                 "neural_checkpoint": metrics(target["data"], neural)}
        event["residual_improves_neural"] = (
            event["residual_surrogate"]["full_l2"]
            < event["neural_checkpoint"]["full_l2"])
        events.append(event)

    def mean(branch: str, metric: str) -> float:
        return float(np.mean([row[branch][metric] for row in events]))

    neural_mean = mean("neural_checkpoint", "full_l2")
    residual_mean = mean("residual_surrogate", "full_l2")
    improvement = (neural_mean - residual_mean) / neural_mean
    improve_count = int(sum(row["residual_improves_neural"] for row in events))
    passed = improvement >= 0.05 and improve_count >= 9
    result = {
        "role": spec["role"], "protocol_sha256": sha256(PROTOCOL),
        "training_checkpoint_sha256": spec["training_checkpoint_sha256"],
        "residual_checkpoint": str(checkpoint_path),
        "residual_checkpoint_sha256": sha256(checkpoint_path),
        "summary": {
            "event_count": len(events),
            "training_seconds": training_seconds,
            "training_leave_one_out_baseline_mean_full_l2": float(np.mean(training_baseline_l2)),
            "local_mean_full_l2": mean("local_interpolation", "full_l2"),
            "neural_mean_full_l2": neural_mean,
            "residual_mean_full_l2": residual_mean,
            "local_mean_active_l2": mean("local_interpolation", "active_l2"),
            "neural_mean_active_l2": mean("neural_checkpoint", "active_l2"),
            "residual_mean_active_l2": mean("residual_surrogate", "active_l2"),
            "relative_improvement_over_neural": improvement,
            "residual_improves_neural_count": improve_count,
            "decision_gate_passed": passed,
        },
        "training_history": history, "events": events,
        "decision_rule": spec["decision_rule"], "limits": spec["limits"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
