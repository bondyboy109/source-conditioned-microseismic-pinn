"""Invert source parameters with a trained source-conditioned PINN.

The waveform NPZ must contain data [receiver, time], times [time], and
receiver_coords [receiver, 3]. Coordinates are metres and times are seconds.
An optional source_truth vector enables error reporting.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from source_conditioned_pinn import (
    SourceConditionedConfig,
    conditioned_prediction,
    load_conditioned_checkpoint,
)


ROOT = Path(__file__).resolve().parent
DOMAIN_SCALE = np.array([16000.0, 16000.0, 3720.0, 9.0], dtype=np.float32)
DEFAULT_BOUNDS = np.array(
    [[2000.0, 14000.0], [2000.0, 14000.0],
     [100.0, 1500.0], [0.05, 0.5]],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("waveforms", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "source_conditioned" /
                        "inversion.json")
    parser.add_argument("--grid", type=int, nargs=4, default=[12, 12, 8, 8])
    parser.add_argument("--candidate-batch", type=int, default=16)
    parser.add_argument("--receiver-stride", type=int, default=2)
    parser.add_argument("--time-stride", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--restarts", type=int, default=1,
                        help="Number of lowest-misfit coarse candidates refined independently")
    return parser.parse_args()


def load_observations(path: Path):
    archive = np.load(path)
    missing = {"data", "times"}.difference(archive.files)
    if missing:
        raise KeyError(f"Missing NPZ fields: {sorted(missing)}")
    data = np.asarray(archive["data"], dtype=np.float32)
    times = np.asarray(archive["times"], dtype=np.float32)
    if "receiver_coords" not in archive.files:
        raise KeyError(
            "receiver_coords is required; the legacy synthetic fallback is "
            "not valid for source-conditioned evaluation."
        )
    receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
    truth = (
        np.asarray(archive["source_truth"], dtype=np.float32)
        if "source_truth" in archive.files else None
    )
    if data.shape != (receivers.shape[0], times.shape[0]):
        raise ValueError("data must have shape [n_receiver, n_time]")
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("Observation RMS must be finite and positive")
    return data, times, receivers, truth, rms


def normalize_geometry(receivers, times):
    return receivers / DOMAIN_SCALE[:3], times / DOMAIN_SCALE[3]


def candidate_grid(shape):
    axes = [
        np.linspace(low, high, count, dtype=np.float32)
        for (low, high), count in zip(DEFAULT_BOUNDS, shape)
    ]
    physical = np.array(list(itertools.product(*axes)), dtype=np.float32)
    return physical, physical / DOMAIN_SCALE


def waveform_mse(model, receiver_xyz, times, observed, source,
                 observation_rms):
    prediction = conditioned_prediction(model, receiver_xyz, times, source)
    return ((prediction - observed) / observation_rms).square().mean()


@torch.no_grad()
def grid_search(model, receiver_xyz, times, observed, shape,
                candidate_batch, observation_rms, restarts=1):
    physical, normalized = candidate_grid(shape)
    if restarts < 1 or restarts > len(normalized):
        raise ValueError("Restarts must be between 1 and the number of candidates")
    all_losses = []
    for start in range(0, len(normalized), candidate_batch):
        source = torch.as_tensor(
            normalized[start:start + candidate_batch],
            dtype=torch.float32, device=receiver_xyz.device
        )
        losses = []
        for candidate in source:
            losses.append(waveform_mse(
                model, receiver_xyz, times, observed, candidate,
                observation_rms))
        losses = torch.stack(losses)
        all_losses.extend(losses.cpu().tolist())
    indices = np.argsort(np.asarray(all_losses), kind="stable")[:restarts]
    starts = [(physical[index], normalized[index], float(all_losses[index]))
              for index in indices]
    return starts


def bounded_local_inversion(model, receiver_xyz, times, observed,
                            initial_normalized, iterations, learning_rate,
                            observation_rms):
    lower = torch.as_tensor(
        DEFAULT_BOUNDS[:, 0] / DOMAIN_SCALE,
        dtype=torch.float32, device=receiver_xyz.device
    )
    upper = torch.as_tensor(
        DEFAULT_BOUNDS[:, 1] / DOMAIN_SCALE,
        dtype=torch.float32, device=receiver_xyz.device
    )
    initial = torch.as_tensor(
        initial_normalized, dtype=torch.float32, device=receiver_xyz.device
    )
    fraction = ((initial - lower) / (upper - lower)).clamp(0, 1)
    parameter = torch.nn.Parameter(fraction.clone())
    optimizer = torch.optim.Adam((parameter,), lr=learning_rate)
    history = []
    best = None

    for _ in range(iterations):
        optimizer.zero_grad()
        source = lower + (upper - lower) * parameter
        loss = waveform_mse(model, receiver_xyz, times, observed, source,
                            observation_rms)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(0, 1)
        value = float(loss.detach())
        history.append(value)
        if best is None or value < best[0]:
            best = (value, source.detach().clone())
    return best[1].cpu().numpy(), best[0], history


def source_gradient(model, receiver_xyz, times, observed,
                    source_normalized, observation_rms):
    """Return dL/d(xs,ys,zs,ts) to audit all inversion paths."""
    source = torch.tensor(
        source_normalized, dtype=torch.float32,
        device=receiver_xyz.device, requires_grad=True
    )
    loss = waveform_mse(model, receiver_xyz, times, observed, source,
                        observation_rms)
    gradient = torch.autograd.grad(loss, source)[0]
    return gradient.detach().cpu().numpy()


def main():
    args = parse_args()
    if (any(count < 2 for count in args.grid) or args.restarts < 1
            or args.restarts > math.prod(args.grid)
            or args.iterations < 1 or args.learning_rate <= 0
            or args.receiver_stride < 1 or args.time_stride < 1):
        raise ValueError("Invalid grid, restart, stride, or optimization settings")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(args.checkpoint, device)
    architecture = metadata.get("architecture", {})
    if architecture.get("kind") == "velocity_conditioned_travel_time_carrier":
        velocity_path = ROOT / "data" / "velocity" / "vp_model.npy"
        model.set_velocity_grid(torch.as_tensor(
            np.array(np.load(velocity_path, mmap_mode="r"), dtype=np.float32),
            device=device))
    config = metadata.get("config", {})
    expected = SourceConditionedConfig()
    for key in ("x_length_m", "y_length_m", "z_length_m", "time_length_s"):
        if config.get(key) != getattr(expected, key):
            raise ValueError(
                f"Checkpoint {key}={config.get(key)} does not match current "
                f"domain {getattr(expected, key)}; retrain with the corrected scale."
            )
    model.eval()
    model.requires_grad_(False)
    data, times, receivers, truth, observation_rms = load_observations(args.waveforms)
    receiver_norm, time_norm = normalize_geometry(receivers, times)

    receiver_slice = slice(None, None, args.receiver_stride)
    time_slice = slice(None, None, args.time_stride)
    receiver_t = torch.as_tensor(
        receiver_norm[receiver_slice], device=device
    )
    time_t = torch.as_tensor(time_norm[time_slice], device=device)
    observed_t = torch.as_tensor(
        data[receiver_slice, time_slice], device=device
    )

    started = time.perf_counter()
    starts = grid_search(
        model, receiver_t, time_t, observed_t, args.grid,
        args.candidate_batch, observation_rms, args.restarts
    )
    coarse_real, coarse_norm, coarse_loss = starts[0]
    coarse_gradient = source_gradient(
        model, receiver_t, time_t, observed_t, coarse_norm, observation_rms
    )
    refinements = []
    for start_real, start_norm, start_loss in starts:
        estimate, loss, trajectory = bounded_local_inversion(
            model, receiver_t, time_t, observed_t, start_norm,
            args.iterations, args.learning_rate, observation_rms)
        refinements.append((estimate, loss, trajectory, start_real, start_loss))
    final_norm, final_loss, history, selected_start, _ = min(
        refinements, key=lambda record: record[1])
    final_real = final_norm * DOMAIN_SCALE
    result = {
        "method": "source_conditioned_pinn",
        "checkpoint": str(args.checkpoint),
        "checkpoint_frequency_hz": metadata["frequency_hz"],
        "coarse_source": coarse_real.tolist(),
        "selected_refinement_start": selected_start.tolist(),
        "refinements": [{"start": row[3].tolist(), "coarse_loss": row[4],
                         "final_loss": row[1]} for row in refinements],
        "estimated_source": final_real.tolist(),
        "coarse_loss": coarse_loss,
        "coarse_normalized_source_gradient": coarse_gradient.tolist(),
        "all_source_gradient_paths_finite": bool(
            np.isfinite(coarse_gradient).all()
        ),
        "all_source_gradient_paths_nonzero": bool(
            (np.abs(coarse_gradient) > 1e-12).all()
        ),
        "final_loss": final_loss,
        "observation_rms": observation_rms,
        "misfit_scale": "prediction and observation divided by full-record observation RMS",
        "loss_history": history,
        "runtime_seconds": time.perf_counter() - started,
        "grid_shape": args.grid,
    }
    if truth is not None:
        with torch.no_grad():
            truth_source = torch.as_tensor(truth / DOMAIN_SCALE, device=device)
            result["truth_source_normalized_mse"] = float(waveform_mse(
                model, receiver_t, time_t, observed_t, truth_source,
                observation_rms))
        delta = final_real - truth
        result["source_truth"] = truth.tolist()
        result["component_error"] = delta.tolist()
        result["location_error_m"] = float(np.linalg.norm(delta[:3]))
        result["origin_time_error_s"] = float(abs(delta[3]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
