"""Train the source-conditioned PINN with frequency continuation.

Example:
    python train_source_conditioned_pinn.py --epochs 2000

Use --epochs 2 --n-phys 64 --n-ic 32 --n-bc-per-face 8 for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from source_conditioned_pinn import (
    SourceConditionedConfig,
    SourceConditionedWavePINN,
    save_conditioned_checkpoint,
    training_step,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_VELOCITY = ROOT / "data" / "velocity" / "vp_model.npy"
DEFAULT_OUTPUT = ROOT / "outputs" / "source_conditioned"


class GridVelocity:
    def __init__(self, path: Path, device: torch.device):
        array = np.load(path, mmap_mode="r")
        self.shape = array.shape
        # mmap arrays are read-only; copy before creating a writable tensor.
        self.values = torch.as_tensor(
            np.array(array, dtype=np.float32, copy=True), device=device
        )

    def __call__(self, xyz: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(
            [self.shape[0] - 1, self.shape[1] - 1, self.shape[2] - 1],
            device=xyz.device,
        )
        index = (xyz.clamp(0.0, 1.0) * scale).long()
        return self.values[index[:, 0], index[:, 1], index[:, 2]].unsqueeze(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity", type=Path, default=DEFAULT_VELOCITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequencies", type=float, nargs="+",
                        default=[5, 10, 15, 20, 30, 40, 60])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--n-phys", type=int, default=4096)
    parser.add_argument("--n-ic", type=int, default=2048)
    parser.add_argument("--n-bc-per-face", type=int, default=341)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output.mkdir(parents=True, exist_ok=True)
    velocity = GridVelocity(args.velocity, device)
    cfg = SourceConditionedConfig()
    model = SourceConditionedWavePINN().to(device)
    summary = {
        "run": {
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "seed": args.seed,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "n_phys": args.n_phys,
            "n_ic": args.n_ic,
            "n_bc_per_face": args.n_bc_per_face,
            "learning_rate": args.learning_rate,
        }
    }

    for frequency in args.frequencies:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate
        )
        history = []
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            metrics = training_step(
                model=model,
                optimizer=optimizer,
                velocity_lookup=velocity,
                frequency=frequency,
                n_phys=args.n_phys,
                n_ic=args.n_ic,
                n_bc_per_face=args.n_bc_per_face,
                device=device,
                cfg=cfg,
            )
            history.append(metrics)
            if epoch == 1 or epoch % 100 == 0:
                print(
                    f"{frequency:g} Hz epoch {epoch}/{args.epochs}: "
                    f"total={metrics['total']:.6e}"
                )
            if epoch % args.save_every == 0:
                save_conditioned_checkpoint(
                    args.output / f"conditioned_{frequency:g}Hz_epoch_{epoch}.pth",
                    model, optimizer, frequency, epoch, cfg, history
                )

        elapsed = time.perf_counter() - started
        final_path = args.output / f"conditioned_{frequency:g}Hz.pth"
        save_conditioned_checkpoint(
            final_path, model, optimizer, frequency, args.epochs, cfg, history
        )
        summary[f"{frequency:g}Hz"] = {
            "final": history[-1],
            "seconds": elapsed,
            "checkpoint": str(final_path),
            "source_conditioned": True,
            "input_dim": 8,
        }
        if device.type == "cuda":
            summary[f"{frequency:g}Hz"]["peak_cuda_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
        with (args.output / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
