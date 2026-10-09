"""Batched second-order 3-D acoustic FD receiver simulation utilities."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from examples.generate_fd_observation import receiver_weights


def _source_kernel(source_xyz: np.ndarray, spacing: np.ndarray,
                   shape: tuple[int, int, int], device: torch.device):
    sigma = 50.0
    radius = 4 * sigma
    bounds = [
        (max(1, math.floor((value - radius) / step)),
         min(count - 2, math.ceil((value + radius) / step)))
        for value, step, count in zip(source_xyz, spacing, shape)
    ]
    slices = tuple(slice(low, high + 1) for low, high in bounds)
    axes = [torch.arange(low, high + 1, device=device, dtype=torch.float32)
            * float(step)
            for (low, high), step in zip(bounds, spacing)]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    kernel = torch.exp(-((gx - float(source_xyz[0])).square()
                         + (gy - float(source_xyz[1])).square()
                         + (gz - float(source_xyz[2])).square())
                       / (2 * sigma**2))
    return slices, kernel


def simulate_candidates(config_path: Path, source_xyz: np.ndarray, duration_s: float,
                        origin_time_s: float = 0.05, batch_size: int = 4,
                        device: torch.device | None = None) -> tuple[np.ndarray, dict]:
    """Simulate candidate receiver records with the registered FD discretization."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model = np.load(cfg["velocity_file"], mmap_mode="r")
    receivers = np.load(cfg["receivers_file"])
    shape = tuple(cfg["velocity_shape"])
    spacing = np.asarray(cfg["grid_spacing_m"], dtype=np.float64)
    sources = np.asarray(source_xyz, dtype=np.float64)
    if sources.ndim != 2 or sources.shape[1] != 3 or len(sources) == 0:
        raise ValueError("source_xyz must have shape [candidate, 3]")
    if batch_size < 1 or duration_s <= 0 or duration_s > cfg["record_length_s"]:
        raise ValueError("Invalid batch size or duration")
    if model.shape != shape or not np.allclose(spacing, spacing[0]):
        raise ValueError("Expected the registered cubic FD grid")
    low, high = np.asarray(cfg["source_location_bounds_m"], dtype=np.float64)
    if np.any(sources < low) or np.any(sources > high):
        raise ValueError("Candidate source is outside the registered bounds")
    t_low, t_high = cfg["origin_time_bounds_s"]
    if not t_low <= origin_time_s <= t_high:
        raise ValueError("Reference origin time is outside the registered bounds")

    dt = float(cfg["time_step_s"])
    n_time = int(round(duration_s / dt)) + 1
    cfl = float(np.max(model) * dt * np.linalg.norm(1.0 / spacing))
    if cfl >= 1:
        raise ValueError(f"Unstable FD time step: CFL={cfl:.3f}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    velocity = torch.as_tensor(np.array(model, dtype=np.float32, copy=True), device=device)
    coefficient = velocity.square() * (dt / float(spacing[0]))**2
    del velocity
    indices, weights = receiver_weights(receivers, shape, spacing, device)
    traces = np.empty((len(sources), len(receivers), n_time), dtype=np.float32)
    batch_seconds = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for start in range(0, len(sources), batch_size):
        selected = sources[start:start + batch_size]
        count = len(selected)
        kernels = [_source_kernel(source, spacing, shape, device)
                   for source in selected]
        u_prev = torch.zeros((count, *shape), device=device, dtype=torch.float32)
        u_curr = torch.zeros_like(u_prev)
        u_next = torch.zeros_like(u_prev)
        trace = torch.zeros((count, len(receivers), n_time),
                            device=device, dtype=torch.float32)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            for step in range(n_time):
                sampled = torch.zeros((count, len(receivers)), device=device)
                for corner in range(8):
                    ix, iy, iz = indices[corner].unbind(dim=1)
                    sampled += weights[corner][None, :] * u_curr[:, ix, iy, iz]
                trace[:, :, step] = sampled
                if step == n_time - 1:
                    break
                center = u_curr[:, 1:-1, 1:-1, 1:-1]
                laplacian = (
                    u_curr[:, 2:, 1:-1, 1:-1] + u_curr[:, :-2, 1:-1, 1:-1]
                    + u_curr[:, 1:-1, 2:, 1:-1] + u_curr[:, 1:-1, :-2, 1:-1]
                    + u_curr[:, 1:-1, 1:-1, 2:] + u_curr[:, 1:-1, 1:-1, :-2]
                    - 6 * center
                )
                u_next[:, 1:-1, 1:-1, 1:-1] = (
                    2 * center - u_prev[:, 1:-1, 1:-1, 1:-1]
                    + coefficient[None, 1:-1, 1:-1, 1:-1] * laplacian
                )
                tau = math.pi * 5.0 * (step * dt - origin_time_s)
                pulse = (1 - 2 * tau * tau) * math.exp(-tau * tau)
                scale = (0.5 if step == 0 else 1.0) * dt * dt * pulse
                for batch_index, (slices, kernel) in enumerate(kernels):
                    u_next[(batch_index, *slices)] += scale * kernel
                u_prev, u_curr, u_next = u_curr, u_next, u_prev
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_seconds.append(time.perf_counter() - started)
        traces[start:start + count] = trace.cpu().numpy()
        del trace, u_prev, u_curr, u_next, kernels

    report = {
        "method": "batched_second_order_3d_acoustic_FD",
        "candidate_count": int(len(sources)),
        "batch_size": int(batch_size),
        "batch_seconds": batch_seconds,
        "runtime_seconds": float(sum(batch_seconds)),
        "duration_s": float(duration_s),
        "time_step_s": dt,
        "reference_origin_time_s": float(origin_time_s),
        "device": str(device),
        "cfl": cfl,
        "peak_cuda_memory_bytes": (int(torch.cuda.max_memory_allocated(device))
                                   if device.type == "cuda" else None),
    }
    return traces, report


def best_origin_time(template: np.ndarray, observed: np.ndarray, dt: float,
                     reference_origin_s: float, bounds_s: tuple[float, float]) -> dict:
    """Find the zero-padded discrete delay with minimum unscaled waveform MSE."""
    template = np.asarray(template, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if (template.ndim != 2 or observed.ndim != 2
            or template.shape[0] != observed.shape[0]
            or template.shape[1] < observed.shape[1]):
        raise ValueError("Template must cover the observed [receiver, time] window")
    first = int(math.ceil((bounds_s[0] - reference_origin_s) / dt - 1e-10))
    last = int(math.floor((bounds_s[1] - reference_origin_s) / dt + 1e-10))
    if first > last:
        raise ValueError("Invalid origin-time bounds")
    observed_energy = float(np.sum(observed**2))
    if observed_energy <= 0:
        raise ValueError("Observed waveform has zero energy")
    best = None
    observed_count = observed.shape[1]
    for lag in range(first, last + 1):
        prediction = np.zeros_like(observed)
        if lag >= 0:
            usable = observed_count - lag
            if usable <= 0:
                continue
            prediction[:, lag:] = template[:, :usable]
        else:
            start = -lag
            stop = start + observed_count
            if stop > template.shape[1]:
                continue
            prediction[:] = template[:, start:stop]
        squared_error = float(np.sum((prediction - observed)**2))
        normalized_mse = squared_error / observed.size / (observed_energy / observed.size)
        if best is None or normalized_mse < best["normalized_mse"]:
            best = {"origin_time_s": reference_origin_s + lag * dt,
                    "lag_samples": lag, "normalized_mse": normalized_mse,
                    "relative_l2": math.sqrt(max(squared_error / observed_energy, 0))}
    return best
