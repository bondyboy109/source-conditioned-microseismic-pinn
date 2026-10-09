"""Train a source-conditioned eikonal traveltime network."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from eikonal_traveltime import FactorizedEikonalNet, eikonal_loss, save_eikonal_checkpoint
from source_conditioned_pinn import SourceConditionedConfig, sample_source_condition
from train_source_conditioned_pinn import GridVelocity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--core-m", type=float, default=20.0)
    parser.add_argument("--exclude-source-radius-m", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.save_every < 1:
        raise ValueError("Training counts must be positive")
    if args.exclude_source_radius_m <= args.core_m:
        raise ValueError("Excluded source radius must exceed the smooth core")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    array = np.load(args.velocity, mmap_mode="r")
    vmin, vmax = float(array.min()), float(array.max())
    velocity = GridVelocity(args.velocity, device)
    cfg = SourceConditionedConfig()
    model = FactorizedEikonalNet(
        velocity_min=0.95 * vmin, velocity_max=1.05 * vmax,
        domain_m=(cfg.x_length_m, cfg.y_length_m, cfg.z_length_m),
        core_m=args.core_m).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    lengths = torch.tensor(model.domain_m, device=device)
    metadata = {
        "velocity": str(args.velocity.resolve()),
        "velocity_sha256": hashlib.sha256(args.velocity.read_bytes()).hexdigest(),
        "velocity_min_max": [vmin, vmax], "seed": args.seed,
        "batch_size": args.batch_size,
        "exclude_source_radius_m": args.exclude_source_radius_m,
        "device": str(device),
    }
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        source = sample_source_condition(args.batch_size, device, cfg)[:, :3]
        xyz = torch.rand(args.batch_size, 3, device=device)
        distance = torch.linalg.vector_norm((xyz - source) * lengths, dim=-1)
        near = distance < args.exclude_source_radius_m
        while near.any():
            xyz[near] = torch.rand(int(near.sum()), 3, device=device)
            distance = torch.linalg.vector_norm((xyz - source) * lengths, dim=-1)
            near = distance < args.exclude_source_radius_m
        speed = velocity(xyz)
        loss = eikonal_loss(model, xyz, source, speed)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
        if step == 1 or step % 100 == 0:
            print(f"step {step}/{args.steps}: relative eikonal MSE={history[-1]:.6e}",
                  flush=True)
        if step % args.save_every == 0:
            save_eikonal_checkpoint(args.output / f"eikonal_step_{step}.pth",
                                    model, optimizer, step, history, metadata)
    elapsed = time.perf_counter() - started
    final = args.output / "eikonal.pth"
    metadata.update(runtime_seconds=elapsed)
    save_eikonal_checkpoint(final, model, optimizer, args.steps, history, metadata)
    (args.output / "training_summary.json").write_text(json.dumps({
        **metadata, "steps": args.steps, "final_loss": history[-1],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(final),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
