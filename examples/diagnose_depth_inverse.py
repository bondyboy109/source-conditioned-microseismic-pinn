"""Diagnose frozen depth waveform error and truth-centred local inversion."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from invert_source_conditioned import DEFAULT_BOUNDS, waveform_mse
from source_conditioned_pinn import conditioned_prediction, load_conditioned_checkpoint

from audit_forward_error_decomposition import predict
from audit_spatial_source_response import response_metrics


CHECKPOINT = ROOT / "outputs" / "depth_coverage_aug24_hybrid_1000" / "conditioned_5Hz.pth"
OBS = ROOT / "data" / "observations" / "depth_coverage"
EVENTS = ("dev_train_01_z260", "dev_train_01_z740",
          "dev_train_09_z460", "dev_train_09_z940")
OUTPUT = ROOT / "experiments" / "depth_inverse_diagnostic.json"
WINDOWS = ((0, .5), (.5, 1), (1, 1.5), (1.5, 2.001))


def station_shift_gain(fd: np.ndarray, pinn: np.ndarray, dt: float) -> dict:
    """Diagnostic integer-lag fit; never used in the raw error or inverse."""
    best = (float("inf"), 0, 0.0)
    for lag in range(-20, 21):
        shifted = np.zeros_like(pinn)
        if lag < 0:
            shifted[:lag] = pinn[-lag:]
        elif lag > 0:
            shifted[lag:] = pinn[:-lag]
        else:
            shifted[:] = pinn
        denom = float(np.dot(shifted, shifted))
        gain = float(np.dot(fd, shifted)) / denom if denom else 0.0
        error = float(np.sum((fd - gain * shifted) ** 2))
        if error < best[0]:
            best = error, lag, gain
    return {"best_lag_s_diagnostic_only": best[1] * dt,
            "best_gain_diagnostic_only": best[2]}


def decompose(fd: np.ndarray, pinn: np.ndarray, times: np.ndarray,
              receivers: np.ndarray) -> dict:
    energy = float(np.sum(fd ** 2))
    error = (fd - pinn) ** 2
    receiver_energy = np.sum(fd ** 2, axis=1)
    station_rows = []
    for i, xyz in enumerate(receivers):
        row = {"receiver_index": i, "receiver_xyz_m": xyz.tolist(),
               "fd_energy_fraction": float(receiver_energy[i] / energy),
               "squared_error_fraction": float(np.sum(error[i]) / energy),
               "raw_relative_l2": (float(np.sqrt(np.sum(error[i]) / receiver_energy[i]))
                                    if receiver_energy[i] > energy * 1e-4 else None)}
        if receiver_energy[i] >= receiver_energy.max() * .01:
            row.update(station_shift_gain(fd[i], pinn[i], float(np.median(np.diff(times)))))
        station_rows.append(row)
    windows = []
    for start, stop in WINDOWS:
        selected = (times >= start) & (times < stop)
        windows.append({"time_s": [start, min(stop, 2.0)],
                        "fd_energy_fraction": float(np.sum(fd[:, selected] ** 2) / energy),
                        "squared_error_fraction": float(np.sum(error[:, selected]) / energy)})
    return {"raw_relative_l2": float(np.sqrt(np.sum(error) / energy)),
            "station_rows": station_rows, "windows": windows}


def loss_at(model, xyz, times, observed, physical_source, scale, rms):
    source = torch.as_tensor(np.asarray(physical_source, dtype=np.float32) / scale,
                             device=xyz.device)
    with torch.no_grad():
        return float(waveform_mse(model, xyz, times, observed, source, rms))


def depth_scan(model, xyz, times, observed, truth, scale, rms):
    zs = np.arange(max(100., float(truth[2]) - 160.),
                   min(1500., float(truth[2]) + 160.) + 1., 20.)
    zs = np.unique(np.r_[zs, float(truth[2])])
    losses = []
    for z in zs:
        source = truth.copy()
        source[2] = z
        losses.append(loss_at(model, xyz, times, observed, source, scale, rms))
    minimum = int(np.argmin(losses))
    true_index = int(np.argmin(np.abs(zs - truth[2])))
    return {"fixed_xyz_time_except_z": True, "z_m": zs.tolist(),
            "normalized_mse": losses, "best_z_m": float(zs[minimum]),
            "best_loss": losses[minimum], "true_z_m": float(truth[2]),
            "true_loss": losses[true_index],
            "off_truth_has_lower_loss": bool(minimum != true_index
                                             and losses[minimum] < losses[true_index])}


def local_search(model, xyz, times, observed, truth, scale, rms):
    low = np.maximum(truth - np.array([160., 160., 160., .04]), DEFAULT_BOUNDS[:, 0])
    high = np.minimum(truth + np.array([160., 160., 160., .04]), DEFAULT_BOUNDS[:, 1])
    low_t = torch.as_tensor(low / scale, dtype=torch.float32, device=xyz.device)
    high_t = torch.as_tensor(high / scale, dtype=torch.float32, device=xyz.device)
    records = []
    for offset in ((80., 0., -80., .02), (-80., 80., 80., -.02)):
        initial = truth + np.asarray(offset, dtype=np.float32)
        fraction = torch.nn.Parameter(torch.as_tensor(
            (initial - low) / (high - low), dtype=torch.float32, device=xyz.device))
        optimizer = torch.optim.Adam((fraction,), lr=.015)
        best = (float("inf"), initial.copy())
        for _ in range(120):
            optimizer.zero_grad()
            source = low_t + (high_t - low_t) * fraction
            loss = waveform_mse(model, xyz, times, observed, source, rms)
            if float(loss.detach()) < best[0]:
                best = (float(loss.detach()),
                        (source.detach().cpu().numpy() * scale).copy())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                fraction.clamp_(0., 1.)
        final = (low_t + (high_t - low_t) * fraction).detach().cpu().numpy() * scale
        final_loss = loss_at(model, xyz, times, observed, final, scale, rms)
        if final_loss < best[0]:
            best = final_loss, final.copy()
        estimate = best[1]
        records.append({"initial_source_m_s": initial.tolist(),
                        "selected_source_m_s": estimate.tolist(),
                        "selected_normalized_mse": best[0],
                        "location_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
                        "origin_time_error_s": float(abs(estimate[3] - truth[3])),
                        "box_boundary_hit": bool(np.any(np.isclose(
                            estimate, low, atol=np.array([1., 1., 1., .001]))
                            | np.isclose(estimate, high,
                                         atol=np.array([1., 1., 1., .001]))))})
    return {"bounds_m_s": np.stack([low, high], axis=1).tolist(),
            "iterations": 120, "learning_rate": .015,
            "truth_centered_search_not_blind": True, "starts": records}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(CHECKPOINT, device)
    model.eval().requires_grad_(False)
    cfg = metadata["config"]
    scale = np.array([cfg["x_length_m"], cfg["y_length_m"],
                      cfg["z_length_m"], cfg["time_length_s"]], dtype=np.float32)
    hashes = {row["sha256"] for row in metadata["training_record"]["receiver_observations"]}
    known_sources = [row["source_truth"] for row in metadata["training_record"]["receiver_observations"]]
    events = []
    cached = {}
    for stem in EVENTS:
        path = OBS / f"{stem}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() in hashes:
            raise ValueError(f"Development file appears in receiver training set: {path}")
        fd, pinn, time_axis, truth = predict(CHECKPOINT, path, 8192)
        if any(np.allclose(truth, source, atol=1e-3, rtol=0) for source in known_sources):
            raise ValueError(f"Development source appears in receiver training set: {path}")
        with np.load(path) as archive:
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        row = {"event": stem, "truth_source_m_s": truth.tolist(),
               "waveform": decompose(fd, pinn, time_axis, receivers)}
        xyz = torch.as_tensor(receivers / scale[:3], device=device)
        times = torch.as_tensor(time_axis[::4].astype(np.float32) / scale[3], device=device)
        observed = torch.as_tensor(fd[:, ::4].astype(np.float32), device=device)
        rms = float(np.sqrt(np.mean(fd ** 2)))
        row["depth_scan"] = depth_scan(model, xyz, times, observed, truth, scale, rms)
        if stem.endswith(("z740", "z940")):
            row["local_four_parameter_search"] = local_search(
                model, xyz, times, observed, truth, scale, rms)
        events.append(row)
        cached[stem] = fd, pinn, time_axis, receivers
        print(f"{stem}: raw L2={row['waveform']['raw_relative_l2']:.4f}, "
              f"fixed-truth depth minimum={row['depth_scan']['best_z_m']:.0f} m", flush=True)
    near_path = OBS / "train_09_z860.npz"
    if hashlib.sha256(near_path.read_bytes()).hexdigest() not in hashes:
        raise ValueError("The depth-860 reference must be in the training set")
    near_fd, near_pinn, near_times, near_source = predict(CHECKPOINT, near_path, 8192)
    far_fd, far_pinn, far_times, far_receivers = cached["dev_train_09_z940"]
    if not np.array_equal(near_times, far_times):
        raise ValueError("Near and far event time axes differ")
    with np.load(near_path) as archive:
        near_receivers = np.asarray(archive["receiver_coords"])
    if not np.array_equal(near_receivers, far_receivers):
        raise ValueError("Near and far receiver geometries differ")
    if not np.isclose(near_source[2], 860., atol=1e-3):
        raise ValueError("Incorrect training-depth reference")
    near_response = response_metrics(near_fd, near_pinn, far_fd-near_fd,
                                     far_pinn-near_pinn)
    report = {"purpose": "development-only frozen depth-inverse diagnostic",
              "checkpoint": str(CHECKPOINT.resolve()),
              "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              "time_stride_for_inverse": 4, "receiver_stride_for_inverse": 1,
              "normalization": "full unstrided observation RMS, fixed over all candidates",
              "uncorrected_primary_loss": True,
              "events": events,
              "train09_depth_860_to_940_response": near_response}
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"z860->940: response L2={near_response['change_relative_l2']:.4f}, "
          f"cosine={near_response['change_cosine']:.4f}", flush=True)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
