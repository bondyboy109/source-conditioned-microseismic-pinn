"""Check whether the network can fit one FD event's receiver traces.

This is a supervised representation diagnostic using the test event itself.
Neither the fitted weights nor its held-out time points validate source
generalization, physics-informed training, or localization performance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source_conditioned_pinn import (SineLayer, SourceConditionedConfig,
                                     SourceConditionedWavePINN)


class HarmonicCarrier(nn.Module):
    """Diagnostic only: learn slowly varying 5-Hz sine/cosine coefficients."""

    def __init__(self, frequency_hz: float, time_length_s: float):
        super().__init__()
        self.frequency_hz = frequency_hz
        self.time_length_s = time_length_s
        self.net = nn.Sequential(
            SineLayer(8, 128, is_first=True),
            *(SineLayer(128, 128) for _ in range(4)),
            nn.Linear(128, 2),
        )

    def forward(self, space_time: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        coefficients = self.net(torch.cat((space_time, source), dim=-1))
        phase = (2 * math.pi * self.frequency_hz * self.time_length_s
                 * (space_time[:, 3:4] - source[:, 3:4]))
        return (coefficients[:, :1] * torch.cos(phase)
                + coefficients[:, 1:] * torch.sin(phase))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("observation", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--harmonic-carrier", action="store_true",
                        help="Test a 5 Hz sine/cosine output representation")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "waveform_capacity.json")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2:
        parser.error("steps must be positive and batch size must be at least 2")
    with np.load(args.observation) as archive:
        waveform = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        source = np.asarray(archive["source_truth"], dtype=np.float32)
    if waveform.shape != (len(receivers), len(times)) or source.shape != (4,):
        raise ValueError("Invalid waveform geometry or missing source truth")
    if not np.isfinite(waveform).all() or not np.isfinite(times).all():
        raise ValueError("Invalid waveform or times")
    scale = float(np.max(np.abs(waveform)))
    if scale <= 0:
        raise ValueError("Waveform is all zero")
    cfg = SourceConditionedConfig()
    domain = np.array([cfg.x_length_m, cfg.y_length_m,
                       cfg.z_length_m, cfg.time_length_s], dtype=np.float32)
    held_out = np.arange(len(times)) % 5 == 0
    training = ~held_out
    n_receiver, n_time = waveform.shape
    space_time = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (n_receiver, n_time, 3)),
        np.broadcast_to(times[None, :, None], (n_receiver, n_time, 1)),
    ), axis=-1) / domain
    x_train = space_time[:, training].reshape(-1, 4)
    y_train = (waveform[:, training] / scale).reshape(-1)
    x_test = space_time[:, held_out].reshape(-1, 4)
    y_test = (waveform[:, held_out] / scale).reshape(-1)
    active = np.flatnonzero(np.abs(y_train) >= 0.05)
    if len(active) == 0:
        raise ValueError("No active waveform samples above 5% of the global peak")

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (HarmonicCarrier(5.0, cfg.time_length_s)
             if args.harmonic_carrier else SourceConditionedWavePINN()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    src = torch.as_tensor(source / domain, device=device)
    half = args.batch_size // 2
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        # Balance active waveforms and quiet samples to expose short arrivals.
        indices = np.concatenate((rng.choice(active, half, replace=True),
                                  rng.integers(len(y_train), size=args.batch_size - half)))
        x = torch.as_tensor(x_train[indices], device=device)
        y = torch.as_tensor(y_train[indices], device=device)
        optimizer.zero_grad()
        prediction = model(x, src.expand(len(x), 4)).squeeze(-1)
        loss = (prediction - y).square().mean()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            print(f"Step {step}/{args.steps}: balanced batch MSE={loss.item():.5g}", flush=True)

    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x_test), 8192):
            x = torch.as_tensor(x_test[start:start + 8192], device=device)
            chunks.append(model(x, src.expand(len(x), 4)).squeeze(-1).cpu().numpy())
    pred = np.concatenate(chunks).astype(np.float64)
    truth = y_test.astype(np.float64)
    relative_l2 = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
    report = {
        "purpose": "single-event supervised architecture capacity test only",
        "architecture": "harmonic_carrier_5Hz" if args.harmonic_carrier else "original_8_input",
        "observation": str(args.observation),
        "seed": 42,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": str(device),
        "training_samples": len(y_train),
        "held_out_time_samples": len(y_test),
        "training_active_samples": len(active),
        "amplitude_scale": scale,
        "held_out_relative_l2": relative_l2,
        "zero_prediction_relative_l2": 1.0,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Held-out time relative L2={relative_l2:.4g}; zero baseline=1.0")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
