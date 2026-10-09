"""Export frozen FD/PINN traces for the four registered high-error stations."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_forward_error_decomposition import predict
from source_conditioned_pinn import load_conditioned_checkpoint


CHECKPOINT = ROOT / "outputs/depth_coverage_aug24_hybrid_1000/conditioned_5Hz.pth"
DATA = ROOT / "data/observations/depth_coverage"
RECORDS = (DATA / "train_09_z860.npz", DATA / "dev_train_09_z940.npz")
STATIONS = (5, 10, 6, 11)  # Fixed by the preceding depth_inverse_diagnostic.json.
TRACE_OUTPUT = ROOT / "outputs/depth_trace_overlay_data.npz"
REPORT_OUTPUT = ROOT / "experiments/depth_trace_overlay_metrics.json"
WINDOWS = ((0., .5), (.5, 1.), (1., 1.5), (1.5, 2.001))


def relative_l2(reference: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / np.linalg.norm(reference))


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sum(first * second) / (np.linalg.norm(first) * np.linalg.norm(second)))


def best_shift_gain(fd: np.ndarray, pinn: np.ndarray, dt: float) -> dict:
    best = (float("inf"), 0, 0.)
    for lag in range(-20, 21):
        shifted = np.zeros_like(pinn)
        if lag > 0:
            shifted[lag:] = pinn[:-lag]
        elif lag < 0:
            shifted[:lag] = pinn[-lag:]
        else:
            shifted[:] = pinn
        norm = float(np.dot(shifted, shifted))
        gain = float(np.dot(fd, shifted)) / norm if norm else 0.
        score = float(np.linalg.norm(fd - gain * shifted) / np.linalg.norm(fd))
        if score < best[0]:
            best = (score, lag, gain)
    return {"adjusted_relative_l2_diagnostic_only": best[0],
            "lag_s_diagnostic_only": best[1] * dt,
            "gain_diagnostic_only": best[2]}


def main() -> None:
    _, payload = load_conditioned_checkpoint(CHECKPOINT, torch.device("cpu"))
    hashes = {row["sha256"] for row in payload["training_record"]["receiver_observations"]}
    if (hashlib.sha256(RECORDS[0].read_bytes()).hexdigest() not in hashes
            or hashlib.sha256(RECORDS[1].read_bytes()).hexdigest() in hashes):
        raise ValueError("The 860 m source must be training-only and 940 m development-only")
    pairs = [predict(CHECKPOINT, path, 8192) for path in RECORDS]
    (fd860, pin860, times860, source860), (fd940, pin940, times940, source940) = pairs
    if not np.array_equal(times860, times940):
        raise ValueError("Time axes do not match")
    with np.load(RECORDS[0]) as first, np.load(RECORDS[1]) as second:
        receivers = np.asarray(first["receiver_coords"], dtype=np.float32)
        if not np.array_equal(receivers, second["receiver_coords"]):
            raise ValueError("Receiver geometry differs")
    if not (np.allclose(source860, [6500., 3500., 860., .2], atol=1e-5)
            and np.allclose(source940, [6500., 3500., 940., .2], atol=1e-5)):
        raise ValueError("Unexpected FD source geometry")
    if any(np.array_equal(fd940[i], pin940[i]) for i in STATIONS):
        raise ValueError("FD and PINN traces must be independently calculated")
    dt = float(np.median(np.diff(times860)))
    records = []
    for index in STATIONS:
        fd_change = fd940[index] - fd860[index]
        pinn_change = pin940[index] - pin860[index]
        energy = float(np.sum(fd940[index] ** 2))
        windows = []
        for low, high in WINDOWS:
            segment = (times940 >= low) & (times940 < high)
            windows.append({"time_s": [low, min(high, 2.)],
                            "fd_energy_fraction": float(np.sum(fd940[index, segment] ** 2) / energy),
                            "residual_squared_over_full_station_fd_energy": float(
                                np.sum((fd940[index, segment] - pin940[index, segment]) ** 2) / energy)})
        row = {"index": index, "xyz_m": receivers[index].tolist(),
               "training_depth_trace_relative_l2": relative_l2(fd860[index], pin860[index]),
               "development_depth_trace_relative_l2": relative_l2(fd940[index], pin940[index]),
               "depth_80m_change_relative_l2": relative_l2(fd_change, pinn_change),
               "depth_80m_change_cosine": cosine(fd_change, pinn_change),
               "development_peak_ratio": float(np.max(np.abs(pin940[index])) /
                                               np.max(np.abs(fd940[index]))),
               "development_window_breakdown": windows,
               **best_shift_gain(fd940[index], pin940[index], dt)}
        records.append(row)
        print(f"station {index} {row['xyz_m']}: z940 L2="
              f"{row['development_depth_trace_relative_l2']:.3f}, "
              f"z860->940 response L2={row['depth_80m_change_relative_l2']:.3f}, "
              f"cos={row['depth_80m_change_cosine']:.3f}", flush=True)
    TRACE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TRACE_OUTPUT, times_s=times940.astype(np.float32),
                        receiver_indices=np.asarray(STATIONS), receivers_m=receivers[list(STATIONS)],
                        fd_860=fd860[list(STATIONS)].astype(np.float32),
                        pinn_860=pin860[list(STATIONS)].astype(np.float32),
                        fd_940=fd940[list(STATIONS)].astype(np.float32),
                        pinn_940=pin940[list(STATIONS)].astype(np.float32))
    report = {"purpose": "development-only raw waveform overlay and depth-change diagnosis",
              "checkpoint": str(CHECKPOINT.resolve()),
              "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              "training_record": str(RECORDS[0].resolve()),
              "development_record": str(RECORDS[1].resolve()),
              "trace_archive": str(TRACE_OUTPUT.resolve()),
              "no_shifts_or_gain_in_primary_metrics": True,
              "stations": records}
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {REPORT_OUTPUT} and {TRACE_OUTPUT}")


if __name__ == "__main__":
    main()
