"""Generate independent 3-D acoustic FD receiver records at 5 Hz.

Second-order centered differences solve u_tt = v(x)^2 Laplacian(u) + q(x,t).
By default all six faces have zero-Dirichlet boundaries, matching the PINN.
An optional lateral/bottom sponge is for boundary-sensitivity diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "data" / "observations" / "experiment_draft.json"
DEFAULT_OUTPUT = ROOT / "data" / "observations" / "event_01.npz"


def receiver_weights(coords: np.ndarray, shape: tuple[int, ...], spacing: np.ndarray,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pos = coords / spacing
    if np.any(pos < 0) or np.any(pos >= np.array(shape) - 1):
        raise ValueError("Receivers must lie inside the FD grid")
    lower = np.floor(pos).astype(np.int64)
    fraction = pos - lower
    indices, weights = [], []
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                indices.append(lower + np.array([ox, oy, oz]))
                weights.append(np.prod(np.where([ox, oy, oz], fraction, 1 - fraction), axis=1))
    return (
        torch.as_tensor(np.stack(indices), device=device, dtype=torch.long),
        torch.as_tensor(np.stack(weights), device=device, dtype=torch.float32),
    )


def sponge_damping(shape: tuple[int, int, int], spacing: np.ndarray,
                   width_m: float, max_damping_s: float,
                   device: torch.device) -> torch.Tensor:
    """Damp x/y sides and the bottom; leave the top free of sponge damping."""
    axes = []
    for axis, (count, step) in enumerate(zip(shape, spacing)):
        indices = torch.arange(count, dtype=torch.float32, device=device)
        depth = (count - 1 - indices) * float(step)
        if axis < 2:
            depth = torch.minimum(indices * float(step), depth)
        axes.append((1.0 - depth / width_m).clamp(min=0).square())
    return max_damping_s * (axes[0][:, None, None]
                            + axes[1][None, :, None]
                            + axes[2][None, None, :])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=float, nargs=4,
                        default=[7000.0, 8000.0, 500.0, 0.2],
                        metavar=("X", "Y", "Z", "T"))
    parser.add_argument("--duration", type=float, default=None,
                        help="Shorten only for code smoke tests; full record is 9 s")
    parser.add_argument("--sponge-width-m", type=float, default=0.0,
                        help="Optional lateral and bottom sponge width; 0 uses six zero boundaries")
    parser.add_argument("--sponge-max-damping-s", type=float, default=60.0,
                        help="Maximum damping rate per face in 1/s")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    model = np.load(cfg["velocity_file"], mmap_mode="r")
    shape = tuple(cfg["velocity_shape"])
    spacing = np.asarray(cfg["grid_spacing_m"], dtype=np.float64)
    domain = np.asarray(cfg["domain_extent_m"], dtype=np.float64)
    if (not np.isfinite(args.sponge_width_m) or args.sponge_width_m < 0
            or args.sponge_width_m > domain.min() / 4):
        raise ValueError("Sponge width must be 0 or at most one quarter of the smallest domain length")
    if not np.isfinite(args.sponge_max_damping_s) or args.sponge_max_damping_s <= 0:
        raise ValueError("Sponge maximum damping rate must be positive and finite")
    receivers = np.load(cfg["receivers_file"]).astype(np.float32)
    source = np.asarray(args.source, dtype=np.float64)
    if model.shape != shape or not np.allclose((np.array(shape) - 1) * spacing, domain):
        raise ValueError("Velocity shape and physical domain do not match the configuration")
    low, high = np.asarray(cfg["source_location_bounds_m"], dtype=np.float64)
    t_low, t_high = cfg["origin_time_bounds_s"]
    if np.any(source[:3] < low) or np.any(source[:3] > high) or not t_low <= source[3] <= t_high:
        raise ValueError("Source is outside the source-conditioned training bounds")
    if not np.all(np.isfinite(model)) or np.min(model) <= 0:
        raise ValueError("Velocity contains invalid values")

    frequency_hz = 5.0
    dt = float(cfg["time_step_s"])
    duration = float(cfg["record_length_s"] if args.duration is None else args.duration)
    if duration <= 0 or duration > cfg["record_length_s"]:
        raise ValueError("Duration must be within the configured record length")
    n_time = int(round(duration / dt)) + 1
    cfl = float(np.max(model) * dt * np.linalg.norm(1.0 / spacing))
    if cfl >= 1.0:
        raise ValueError(f"Unstable 3-D FD time step (CFL={cfl:.3f})")
    if float(np.min(model)) / (frequency_hz * max(spacing)) < 10:
        raise ValueError("Fewer than ten grid cells per minimum wavelength")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"FD: {shape}, {n_time} samples, dt={dt:g} s, CFL={cfl:.3f}, {device}", flush=True)
    velocity = torch.as_tensor(np.array(model, dtype=np.float32), device=device)
    coeff = velocity.square() * (dt / float(spacing[0])) ** 2
    del velocity
    damping_dt = (sponge_damping(shape, spacing, args.sponge_width_m,
                                 args.sponge_max_damping_s, device) * dt
                  if args.sponge_width_m > 0 else None)
    if damping_dt is not None and damping_dt.max().item() >= 1:
        raise ValueError("Maximum sponge damping times dt must remain below 1")
    indices, weights = receiver_weights(receivers, shape, spacing, device)
    trace = torch.zeros((len(receivers), n_time), device=device, dtype=torch.float32)
    u_prev = torch.zeros(shape, device=device, dtype=torch.float32)
    u_curr = torch.zeros_like(u_prev)
    u_next = torch.zeros_like(u_prev)

    sigma = 50.0
    radius = 4 * sigma
    bounds = [
        (max(1, math.floor((s - radius) / h)),
         min(n - 2, math.ceil((s + radius) / h)))
        for s, h, n in zip(source[:3], spacing, shape)
    ]
    slices = tuple(slice(a, b + 1) for a, b in bounds)
    axes = [torch.arange(a, b + 1, device=device, dtype=torch.float32) * float(h)
            for (a, b), h in zip(bounds, spacing)]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    gaussian = torch.exp(-((gx - source[0]).square() + (gy - source[1]).square()
                           + (gz - source[2]).square()) / (2 * sigma ** 2))
    del gx, gy, gz

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(n_time):
            sampled = torch.zeros(len(receivers), device=device)
            for corner in range(8):
                ix, iy, iz = indices[corner].unbind(dim=1)
                sampled += weights[corner] * u_curr[ix, iy, iz]
            trace[:, step] = sampled
            if step == n_time - 1:
                break

            center = u_curr[1:-1, 1:-1, 1:-1]
            lap = (
                u_curr[2:, 1:-1, 1:-1] + u_curr[:-2, 1:-1, 1:-1]
                + u_curr[1:-1, 2:, 1:-1] + u_curr[1:-1, :-2, 1:-1]
                + u_curr[1:-1, 1:-1, 2:] + u_curr[1:-1, 1:-1, :-2]
                - 6.0 * center
            )
            previous = u_prev[1:-1, 1:-1, 1:-1]
            forcing = coeff[1:-1, 1:-1, 1:-1] * lap
            if damping_dt is None:
                u_next[1:-1, 1:-1, 1:-1] = 2.0 * center - previous + forcing
            else:
                gamma_dt = damping_dt[1:-1, 1:-1, 1:-1]
                u_next[1:-1, 1:-1, 1:-1] = (
                    2.0 * center - (1.0 - gamma_dt) * previous + forcing
                ) / (1.0 + gamma_dt)
            tau = math.pi * frequency_hz * (step * dt - source[3])
            pulse = (1 - 2 * tau * tau) * math.exp(-tau * tau)
            u_next[slices] += (0.5 if step == 0 else 1.0) * dt * dt * pulse * gaussian
            u_prev, u_curr, u_next = u_curr, u_next, u_prev
            if (step + 1) % 500 == 0:
                print(f"FD step {step + 1}/{n_time - 1}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    data = trace.cpu().numpy()
    if not np.all(np.isfinite(data)):
        raise RuntimeError("FD waveforms contain NaN or infinity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        data=data,
        times=np.arange(n_time, dtype=np.float32) * dt,
        receiver_coords=receivers,
        source_truth=source.astype(np.float32),
    )
    report = {
        "method": "second_order_3d_acoustic_FD",
        "frequency_hz": frequency_hz,
        "boundary": ("zero Dirichlet on all six faces" if damping_dt is None
                     else "zero top; lateral and bottom damped sponge ending at zero faces"),
        "sponge_width_m": args.sponge_width_m,
        "sponge_max_damping_s": (args.sponge_max_damping_s
                                 if damping_dt is not None else None),
        "source": "unit Ricker times Gaussian, sigma=50 m",
        "receiver_sampling": "trilinear",
        "device": str(device),
        "cfl": cfl,
        "duration_s": duration,
        "runtime_seconds": elapsed,
        "waveform_peak_abs": float(np.max(np.abs(data))),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else None,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output} ({data.shape}); peak={report['waveform_peak_abs']:.3e}; "
          f"runtime={elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
