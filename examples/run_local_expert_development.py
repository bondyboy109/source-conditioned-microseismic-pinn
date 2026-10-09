"""Train spatially partitioned local waveform experts and audit old records."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments" / "local_expert_development_protocol.json"


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


def deterministic_kmeans(points: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic k-means used only on registered training sources."""
    rng = np.random.default_rng(seed)
    centres = [points[int(rng.integers(len(points)))]]
    while len(centres) < count:
        squared = np.min(np.sum((points[:, None] - np.asarray(centres)[None]) ** 2,
                                axis=2), axis=1)
        centres.append(points[int(np.argmax(squared))])
    centres = np.asarray(centres)
    labels = np.zeros(len(points), dtype=int)
    for _ in range(100):
        new_labels = np.argmin(np.sum((points[:, None] - centres[None]) ** 2, axis=2), axis=1)
        new_centres = np.asarray([points[new_labels == index].mean(axis=0)
                                  for index in range(count)])
        if np.array_equal(new_labels, labels) and np.allclose(new_centres, centres):
            break
        labels, centres = new_labels, new_centres
    if any(np.sum(labels == index) < 2 for index in range(count)):
        raise ValueError("A local expert received fewer than two training sources")
    return labels, centres


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


def predict(model, source: np.ndarray, observed_shape: tuple[int, int],
            times: np.ndarray, receivers: np.ndarray, scale: np.ndarray,
            device: torch.device) -> np.ndarray:
    coords = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*observed_shape, 3)),
        np.broadcast_to(times[None, :, None], (*observed_shape, 1)),
    ), axis=-1).reshape(-1, 4) / scale
    source_t = torch.as_tensor(source / scale, dtype=torch.float32, device=device)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coords), 8192):
            x = torch.as_tensor(coords[start:start + 8192], dtype=torch.float32,
                                device=device)
            chunks.append(model(x, source_t.expand(len(x), 4)).squeeze(-1).cpu().numpy())
    return np.concatenate(chunks).reshape(observed_shape).astype(np.float64)


def train_expert(spec: dict, expert_index: int, records: list[dict],
                 output_root: Path) -> tuple[Path, float]:
    expert_dir = output_root / f"expert_{expert_index + 1:02d}"
    if expert_dir.exists() and any(expert_dir.iterdir()):
        checkpoint = expert_dir / "conditioned_5Hz.pth"
        if checkpoint.is_file():
            return checkpoint, 0.0
        raise FileExistsError(f"Incomplete expert output exists: {expert_dir}")
    expert_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(ROOT / "train_source_conditioned_pinn.py"),
        "--travel-time-carrier", "--eikonal-checkpoint",
        str(ROOT / spec["eikonal_checkpoint"]),
        "--reference-velocity", "3500", "--distance-core-m", "20",
        "--wavefield-scale", "0.00001", "--initial-checkpoint",
        str(ROOT / spec["base_checkpoint"]), "--frequencies", "5",
        "--epochs", str(spec["finetune_updates"]), "--save-every",
        str(spec["finetune_updates"]), "--learning-rate", str(spec["learning_rate"]),
        "--receiver-only", "--receiver-observations",
        *[str(row["path"]) for row in records],
        "--receiver-batch-size", str(spec["receiver_batch_size"]),
        "--receiver-events-per-batch",
        str(min(len(records), spec["maximum_events_per_batch"])),
        "--receiver-loss-weight", "0.1", "--seed", "42",
        "--output", str(expert_dir),
    ]
    started = time.perf_counter()
    with (expert_dir / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    elapsed = time.perf_counter() - started
    checkpoint = expert_dir / "conditioned_5Hz.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint, elapsed


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path = ROOT / spec["result_file"]
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    base_path = ROOT / spec["base_checkpoint"]
    if sha256(base_path) != spec["base_checkpoint_sha256"]:
        raise ValueError("Registered base checkpoint changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, base_payload = load_conditioned_checkpoint(base_path, device)
    manifest = base_payload.get("training_record", {}).get("receiver_observations", [])
    if len(manifest) != 33:
        raise ValueError(f"Expected 33 registered training records, got {len(manifest)}")

    training = []
    for row in manifest:
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Training record changed: {path}")
        _, _, _, source = load_record(path)
        training.append({"path": path, "sha256": row["sha256"], "source": source})
    cluster_scale = np.asarray(spec["cluster_scale_m"], dtype=np.float64)
    points = np.asarray([row["source"][:3] for row in training]) / cluster_scale
    labels, centres = deterministic_kmeans(points, int(spec["expert_count"]),
                                           int(spec["cluster_seed"]))
    clusters = [[training[i] for i in range(len(training)) if labels[i] == index]
                for index in range(int(spec["expert_count"]))]

    output_root = ROOT / spec["output_directory"]
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    training_seconds = []
    for index, records in enumerate(clusters):
        print(f"Training expert {index + 1}/{len(clusters)} on {len(records)} records",
              flush=True)
        checkpoint, elapsed = train_expert(spec, index, records, output_root)
        checkpoints.append(checkpoint)
        training_seconds.append(elapsed)

    expert_models = [load_conditioned_checkpoint(path, device)[0]
                     for path in checkpoints]
    scale = np.asarray([base_payload["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    targets = []
    for item in spec["historical_results"]:
        historical = json.loads((ROOT / item["path"]).read_text(encoding="utf-8"))
        for event in historical["events"]:
            path = Path(event.get("observation", ""))
            if not path.is_absolute() or not path.exists():
                path = ROOT / item["data_directory"] / event["event"]
            observed, times, receivers, source = load_record(path)
            targets.append({"event": event["event"], "path": path,
                            "observed": observed, "times": times,
                            "receivers": receivers, "source": source})
    if len(targets) != 16:
        raise ValueError(f"Expected 16 historical targets, got {len(targets)}")

    events = []
    for target in targets:
        normalized = target["source"][:3] / cluster_scale
        expert_index = int(np.argmin(np.sum((centres - normalized) ** 2, axis=1)))
        base_prediction = predict(base_model, target["source"], target["observed"].shape,
                                  target["times"], target["receivers"], scale, device)
        expert_prediction = predict(expert_models[expert_index], target["source"],
                                    target["observed"].shape, target["times"],
                                    target["receivers"], scale, device)
        base_metrics = metrics(target["observed"], base_prediction)
        expert_metrics = metrics(target["observed"], expert_prediction)
        events.append({"event": target["event"], "observation": str(target["path"]),
                       "sha256": sha256(target["path"]),
                       "source_m_s": target["source"].tolist(),
                       "selected_expert": expert_index + 1,
                       "base": base_metrics, "expert": expert_metrics,
                       "expert_improves": expert_metrics["full_l2"] < base_metrics["full_l2"]})

    def mean(branch: str, metric: str) -> float:
        return float(np.mean([row[branch][metric] for row in events]))

    result = {
        "role": spec["role"], "protocol_sha256": sha256(PROTOCOL),
        "base_checkpoint_sha256": spec["base_checkpoint_sha256"],
        "summary": {
            "event_count": len(events),
            "base_mean_full_l2": mean("base", "full_l2"),
            "expert_mean_full_l2": mean("expert", "full_l2"),
            "base_mean_active_l2": mean("base", "active_l2"),
            "expert_mean_active_l2": mean("expert", "active_l2"),
            "expert_improves_count": int(sum(row["expert_improves"] for row in events)),
            "total_training_seconds": float(sum(training_seconds)),
        },
        "experts": [{
            "expert": index + 1,
            "checkpoint": str(checkpoints[index]),
            "checkpoint_sha256": sha256(checkpoints[index]),
            "training_seconds_this_run": training_seconds[index],
            "centre_scaled": centres[index].tolist(),
            "centre_m": (centres[index] * cluster_scale).tolist(),
            "training_record_count": len(records),
            "training_records": [{"path": str(row["path"]), "sha256": row["sha256"],
                                  "source_m_s": row["source"].tolist()}
                                 for row in records],
        } for index, records in enumerate(clusters)],
        "events": events, "limits": spec["limits"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
