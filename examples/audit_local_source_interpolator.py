"""Audit a time-aligned local interpolation baseline for source waveforms."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/local_source_interpolator_protocol.json"


def load_record(path: Path):
    with np.load(path) as archive:
        return (np.asarray(archive["data"], dtype=np.float64),
                np.asarray(archive["times"], dtype=np.float64),
                np.asarray(archive["receiver_coords"], dtype=np.float64),
                np.asarray(archive["source_truth"], dtype=np.float64))


def shift_origin(data: np.ndarray, times: np.ndarray, training_origin_s: float,
                 target_origin_s: float) -> np.ndarray:
    """Express a training response at a new origin time using zero padding."""
    sample_times = times + training_origin_s - target_origin_s
    shifted = np.empty_like(data)
    for receiver, trace in enumerate(data):
        shifted[receiver] = np.interp(sample_times, times, trace, left=0.0, right=0.0)
    return shifted


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    energy = float(np.sum(observed ** 2))
    active = np.abs(observed) >= 0.05 * np.max(np.abs(observed))
    active_energy = float(np.sum(observed[active] ** 2))
    return {
        "full_l2": float(np.sqrt(np.sum((predicted - observed) ** 2) / energy)),
        "active_l2": float(np.sqrt(np.sum((predicted[active] - observed[active]) ** 2)
                                    / active_energy)),
        "zero_lag_correlation": float(np.sum(predicted * observed)
                                       / (np.linalg.norm(predicted) * np.linalg.norm(observed))),
    }


def predict(training: list[dict], target: np.ndarray, times: np.ndarray,
            xy_scale_m: float, z_scale_m: float, neighbour_count: int,
            power: float) -> tuple[np.ndarray, list[dict]]:
    sources = np.asarray([row["source"] for row in training])
    scale = np.asarray([xy_scale_m, xy_scale_m, z_scale_m])
    distances = np.linalg.norm((sources[:, :3] - target[:3]) / scale, axis=1)
    order = np.argsort(distances, kind="stable")[:neighbour_count]
    if distances[order[0]] <= 1e-12:
        order = order[:1]
        weights = np.ones(1)
    else:
        weights = distances[order] ** (-power)
        weights /= np.sum(weights)
    prediction = np.zeros_like(training[0]["data"])
    neighbours = []
    for index, weight in zip(order, weights):
        row = training[int(index)]
        shifted = shift_origin(row["data"], times, row["source"][3], target[3])
        prediction += float(weight) * shifted
        neighbours.append({"source_m_s": row["source"].tolist(),
                           "scaled_distance": float(distances[index]),
                           "weight": float(weight),
                           "sha256": row["sha256"]})
    return prediction, neighbours


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["result_file"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoint = ROOT / spec["training_checkpoint"]
    depth_result = ROOT / spec["depth_results"]
    if (hash_file(checkpoint) != spec["training_checkpoint_sha256"]
            or hash_file(depth_result) != spec["depth_results_sha256"]):
        raise ValueError("Registered input changed")
    _, metadata = load_conditioned_checkpoint(checkpoint, torch.device("cpu"))
    records = metadata["training_record"]["receiver_observations"]
    if len(records) != 33:
        raise ValueError("Expected the registered 33-source training manifest")
    training = []
    reference_times = reference_receivers = None
    for item in records:
        path = Path(item["path"])
        if hash_file(path) != item["sha256"]:
            raise ValueError(f"Training record changed: {path}")
        data, times, receivers, source = load_record(path)
        if reference_times is None:
            reference_times, reference_receivers = times, receivers
        elif not (np.array_equal(times, reference_times)
                  and np.array_equal(receivers, reference_receivers)):
            raise ValueError("Training observation geometry differs")
        training.append({"data": data, "source": source, "sha256": item["sha256"]})
    depth = json.loads(depth_result.read_text(encoding="utf-8"))
    allowed = set(spec["evaluation_offset_m"])
    targets = [row for row in depth["events"] if row["offset_m"] in allowed]
    if len(targets) != 8:
        raise ValueError("Expected eight nontraining depth development records")
    target_rows = []
    root = ROOT / "data/observations/eikonal_depth_sweep"
    for row in targets:
        path = root / row["file"]
        if hash_file(path) != row["sha256"]:
            raise ValueError(f"Depth record changed: {path}")
        data, times, receivers, source = load_record(path)
        if not (np.array_equal(times, reference_times)
                and np.array_equal(receivers, reference_receivers)):
            raise ValueError(f"Target geometry differs: {path}")
        target_rows.append({"event": row["file"], "offset_m": row["offset_m"],
                            "anchor": row["anchor"], "source": source,
                            "data": data, "sha256": row["sha256"]})
    candidates = []
    combinations = itertools.product(spec["z_scale_m_candidates"],
                                     spec["neighbour_count_candidates"],
                                     spec["inverse_distance_power_candidates"])
    for z_scale, count, power in combinations:
        event_metrics = []
        for target in target_rows:
            prediction, _ = predict(training, target["source"], reference_times,
                                    spec["xy_scale_m"], z_scale, count, power)
            event_metrics.append(metrics(target["data"], prediction))
        candidates.append({"z_scale_m": z_scale, "neighbour_count": count,
                           "inverse_distance_power": power,
                           "mean_full_l2": float(np.mean([m["full_l2"] for m in event_metrics])),
                           "mean_active_l2": float(np.mean([m["active_l2"] for m in event_metrics]))})
    candidates.sort(key=lambda row: (row["mean_full_l2"], row["mean_active_l2"]))
    best = candidates[0]
    events = []
    for target in target_rows:
        prediction, neighbours = predict(
            training, target["source"], reference_times, spec["xy_scale_m"],
            best["z_scale_m"], best["neighbour_count"],
            best["inverse_distance_power"])
        events.append({"event": target["event"], "anchor": target["anchor"],
                       "offset_m": target["offset_m"], "sha256": target["sha256"],
                       "source_m_s": target["source"].tolist(),
                       "metrics": metrics(target["data"], prediction),
                       "neighbours": neighbours})
    neural = json.loads((ROOT / spec["neural_reference"]).read_text())
    neural_events = {Path(row["observation"]).name: row
                     for row in neural["models"]["control"]["events"]}
    selected_neural = [neural_events[row["event"]] for row in events]
    result = {
        "role": spec["role"], "protocol_sha256": hash_file(PROTOCOL),
        "checkpoint_sha256": spec["training_checkpoint_sha256"],
        "candidate_count": len(candidates), "best_configuration": best,
        "top_configurations": candidates[:10],
        "summary": {
            "event_count": len(events),
            "interpolator_mean_full_l2": float(np.mean([r["metrics"]["full_l2"] for r in events])),
            "interpolator_mean_active_l2": float(np.mean([r["metrics"]["active_l2"] for r in events])),
            "neural_control_mean_full_l2": float(np.mean([
                r["full_record_relative_l2"] for r in selected_neural])),
            "neural_control_mean_active_l2": float(np.mean([
                r["active_sample_relative_l2"] for r in selected_neural])),
        },
        "events": events, "limits": spec["limits"],
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"best={best}")
    print(f"summary={result['summary']}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
