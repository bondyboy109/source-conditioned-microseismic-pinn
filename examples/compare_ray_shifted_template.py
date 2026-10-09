"""Development-only receiver-template propagation diagnostic at 5 Hz."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "source_distance_development.json"
OBSERVATIONS = ROOT / "data" / "observations"
VELOCITY = ROOT / "data" / "velocity" / "vp_model.npy"


def interpolate_velocity(model: np.ndarray, coordinates: np.ndarray,
                         spacing_m: float = 40.0) -> np.ndarray:
    """Trilinearly sample the FD velocity grid at physical coordinates."""
    position = np.asarray(coordinates, dtype=np.float64) / spacing_m
    if position.shape[-1] != 3 or not np.isfinite(position).all():
        raise ValueError("Coordinates must have three finite spatial components")
    if np.any(position < 0) or np.any(position > np.asarray(model.shape) - 1):
        raise ValueError("Velocity query is outside the grid")
    lower = np.minimum(np.floor(position).astype(np.intp), np.asarray(model.shape) - 2)
    weight = position - lower
    result = np.zeros(position.shape[:-1], dtype=np.float64)
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                factors = (weight[..., 0] if ix else 1 - weight[..., 0]) * (
                    weight[..., 1] if iy else 1 - weight[..., 1]) * (
                    weight[..., 2] if iz else 1 - weight[..., 2])
                result += factors * model[lower[..., 0] + ix,
                                          lower[..., 1] + iy,
                                          lower[..., 2] + iz]
    return result


def ray_travel_times(model: np.ndarray, source: np.ndarray,
                     receivers: np.ndarray, samples: int = 64) -> np.ndarray:
    """Midpoint slowness quadrature along straight source-receiver segments."""
    if samples < 1:
        raise ValueError("samples must be positive")
    offsets = receivers - source[:3]
    distance = np.linalg.norm(offsets, axis=1)
    fractions = (np.arange(samples) + 0.5) / samples
    points = source[:3] + fractions[None, :, None] * offsets[:, None, :]
    velocity = interpolate_velocity(model, points)
    if np.any(velocity <= 0):
        raise ValueError("Velocity must be positive")
    return distance * np.mean(1.0 / velocity, axis=1)


def shifted_template(data: np.ndarray, times: np.ndarray,
                     old_source: np.ndarray, new_source: np.ndarray,
                     receivers: np.ndarray, old_travel_s: np.ndarray,
                     new_travel_s: np.ndarray) -> np.ndarray:
    old_distance = np.linalg.norm(receivers - old_source[:3], axis=1)
    new_distance = np.linalg.norm(receivers - new_source[:3], axis=1)
    if np.any(old_distance == 0) or np.any(new_distance == 0):
        raise ValueError("Receiver coincides with a source")
    arrival_change = (new_source[3] - old_source[3]
                      + new_travel_s - old_travel_s)
    return np.asarray([
        (old_distance[i] / new_distance[i]) * np.interp(
            times - arrival_change[i], times, data[i], left=0.0, right=0.0)
        for i in range(len(receivers))
    ])


def quality(prediction: np.ndarray, reference: np.ndarray) -> dict:
    reference_energy = float(np.sum(reference ** 2))
    predicted_energy = float(np.sum(prediction ** 2))
    if reference_energy <= 0 or predicted_energy <= 0:
        raise ValueError("Cannot compare zero-energy waveforms")
    return {
        "relative_l2": float(np.sqrt(np.sum((prediction - reference) ** 2)
                                     / reference_energy)),
        "cosine": float(np.sum(prediction * reference)
                        / np.sqrt(reference_energy * predicted_energy)),
        "predicted_to_fd_peak": float(np.max(np.abs(prediction))
                                      / np.max(np.abs(reference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "dev_ray_template_5hz.json")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(OBSERVATIONS / protocol["reference_training_source"]) as archive:
        template = np.asarray(archive["data"], dtype=np.float64)
        times = np.asarray(archive["times"], dtype=np.float64)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
        source = np.asarray(archive["source_truth"], dtype=np.float64)
    if not np.allclose(source, protocol["reference_source_m_s"], atol=1e-5, rtol=0):
        raise ValueError("Template source differs from registered experiment")
    if template.shape != (25, 4501) or receivers.shape != (25, 3):
        raise ValueError("Unexpected template receiver geometry")
    model = np.load(VELOCITY, mmap_mode="r")
    if model.shape != (401, 401, 94):
        raise ValueError("Unexpected velocity grid")
    old_ray = ray_travel_times(model, source, receivers)
    reference_speed = 3500.0
    old_constant = np.linalg.norm(receivers - source[:3], axis=1) / reference_speed
    records = []

    for event in protocol["events"]:
        path = OBSERVATIONS / event["file"]
        with np.load(path) as archive:
            observed = np.asarray(archive["data"], dtype=np.float64)
            event_times = np.asarray(archive["times"], dtype=np.float64)
            event_receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
            new_source = np.asarray(archive["source_truth"], dtype=np.float64)
        if not (observed.shape == template.shape
                and np.array_equal(event_times, times)
                and np.array_equal(event_receivers, receivers)):
            raise ValueError(f"Receiver data mismatch: {path}")
        if not np.allclose(new_source, event["source_m_s"], atol=1e-5, rtol=0):
            raise ValueError(f"Source mismatch: {path}")
        distance = np.linalg.norm(new_source[:3] - source[:3])
        if not np.isclose(distance, event["distance_m"], atol=1e-5):
            raise ValueError(f"Registered distance mismatch: {path}")
        new_ray = ray_travel_times(model, new_source, receivers)
        new_constant = np.linalg.norm(receivers - new_source[:3], axis=1) / reference_speed
        constant_waveform = shifted_template(
            template, times, source, new_source, receivers,
            old_constant, new_constant)
        integrated_waveform = shifted_template(
            template, times, source, new_source, receivers,
            old_ray, new_ray)
        baseline = quality(template, observed)
        constant = quality(constant_waveform, observed)
        integrated = quality(integrated_waveform, observed)
        energy_total = float(np.sum(observed ** 2))
        windows = {}
        for end_s in (2.0, 4.0, 9.0):
            mask = times <= end_s + 1e-6
            reference_window = observed[:, mask]
            windows[f"0_to_{end_s:g}_s"] = {
                "fd_energy_fraction": float(np.sum(reference_window ** 2)
                                            / energy_total),
                "constant_velocity": quality(constant_waveform[:, mask],
                                             reference_window),
                "integrated_slowness": quality(integrated_waveform[:, mask],
                                               reference_window),
            }
        network_report = json.loads((ROOT / "outputs"
                                     / "dev_relative_carrier_5hz_train8_5000"
                                     / f"audit_{path.stem}.json").read_text(encoding="utf-8"))
        if network_report["file_in_receiver_training_set"] or network_report["source_in_receiver_training_set"]:
            raise ValueError("Development source was used to train the network")
        record = {"event": event["file"], "distance_m": distance,
                  "fd_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "network_relative_l2": network_report["relative_l2"],
                  "unchanged_template": baseline,
                  "constant_velocity_shift": constant,
                  "integrated_slowness_shift": integrated,
                  "time_windows": windows,
                  "median_absolute_travel_difference_constant_s": float(
                      np.median(np.abs(new_constant - old_constant))),
                  "median_absolute_travel_difference_integrated_s": float(
                      np.median(np.abs(new_ray - old_ray)))}
        records.append(record)
        print(f"{event['file']}: network={record['network_relative_l2']:.3f}, "
              f"unchanged={baseline['relative_l2']:.3f}, "
              f"constant={constant['relative_l2']:.3f}, "
              f"integrated={integrated['relative_l2']:.3f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "purpose": "development-only FD receiver-template propagation diagnostic",
        "template_event": protocol["reference_training_source"],
        "uses_template_fd_waveforms": True,
        "ray_integration_samples": 64,
        "reference_velocity_m_s": reference_speed,
        "amplitude_rule": "old_distance/new_distance, no fitted gain",
        "results": records,
    }, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
