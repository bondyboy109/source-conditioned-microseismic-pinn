"""Small, leakage-controlled velocity pretraining audit (not a paper benchmark).

An independent analytic 1-D velocity model supplies source-domain labels.
All arms use the same target pool; source and continued-target arms share
the final target batches, and the 500-step scratch arm uses both target batches.
The test set holds out an entire x-coordinate slab from target fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class VelocityNet(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 1))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def source_velocity(points: np.ndarray) -> np.ndarray:
    """Independent layered/gradient model; never reads the target volume."""
    depth = points[:, 2]
    return 3000.0 + 1700.0 * depth + 300.0 * np.tanh(8.0 * (depth - 0.55))


def draw_target(volume: np.ndarray, rng: np.random.Generator,
                count: int, x_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = rng.choice(x_indices, count)
    iy = rng.integers(0, volume.shape[1], count)
    iz = rng.integers(0, volume.shape[2], count)
    points = np.stack((ix / (volume.shape[0] - 1),
                       iy / (volume.shape[1] - 1),
                       iz / (volume.shape[2] - 1)), axis=-1).astype(np.float32)
    values = np.asarray(volume[ix, iy, iz], dtype=np.float32)
    return 2.0 * points - 1.0, (values - 4000.0) / 2000.0


def to_tensors(points: np.ndarray, labels: np.ndarray, device: torch.device):
    return (torch.as_tensor(points, device=device),
            torch.as_tensor(labels[:, None], device=device))


def rmse_mae(model: nn.Module, data, device: torch.device) -> dict:
    model.eval()
    with torch.inference_mode():
        error = ((model(data[0]).squeeze(-1) - data[1].squeeze(-1)) * 2000.0)
        return {"rmse_m_s": float(torch.sqrt(error.square().mean()).cpu()),
                "mae_m_s": float(error.abs().mean().cpu())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True,
                        help="Reference vp_true.npy, not the already reconstructed vp_model.npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Optional directory for per-seed, per-arm velocity weights")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41, 53, 67])
    parser.add_argument("--source-steps", type=int, default=250)
    parser.add_argument("--target-steps", type=int, default=250)
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--eval-count", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    if (args.source_steps < 1 or args.target_steps < 1 or args.train_count < 1
            or args.eval_count < 1 or args.batch_size < 1 or args.width < 8
            or not args.seeds):
        parser.error("Steps, sample counts, batch size, width, and seeds must be positive")
    if not args.target.is_file():
        parser.error(f"Missing target reference volume: {args.target}")
    volume = np.load(args.target, mmap_mode="r")
    if volume.ndim != 3 or min(volume.shape) < 10:
        raise ValueError("Expected a 3-D reference velocity volume")
    # Validation and test are entire x slabs; no target training sample can
    # share their x coordinate. This is intentionally harder than random voxels.
    ix = np.arange(volume.shape[0])
    fraction = ix / (volume.shape[0] - 1)
    val_ix = ix[(fraction >= 0.20) & (fraction <= 0.28)]
    test_ix = ix[(fraction >= 0.60) & (fraction <= 0.72)]
    train_ix = ix[~np.isin(ix, np.r_[val_ix, test_ix])]
    rng = np.random.default_rng(20260924)
    target_train_np = draw_target(volume, rng, args.train_count, train_ix)
    target_val_np = draw_target(volume, rng, args.eval_count, val_ix)
    target_test_np = draw_target(volume, rng, args.eval_count, test_ix)
    source_points = rng.uniform(-1.0, 1.0, (args.train_count, 3)).astype(np.float32)
    source_labels = ((source_velocity((source_points + 1.0) / 2.0) - 4000.0)
                     / 2000.0).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_train = to_tensors(*target_train_np, device)
    target_val = to_tensors(*target_val_np, device)
    target_test = to_tensors(*target_test_np, device)
    source_train = to_tensors(source_points, source_labels, device)
    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        initial = VelocityNet(args.width).to(device).state_dict()
        # Copy tensors because target-only pretraining mutates its model in-place.
        initial = {key: value.clone() for key, value in initial.items()}
        seq_rng = np.random.default_rng(seed)
        source_batches = seq_rng.integers(0, len(source_points),
                                         (args.source_steps, args.batch_size))
        target_pre_batches = seq_rng.integers(0, len(target_train_np[0]),
                                             (args.source_steps, args.batch_size))
        target_batches = seq_rng.integers(0, len(target_train_np[0]),
                                         (args.target_steps, args.batch_size))
        arms = {}
        for name in ("scratch", "scratch_matched_compute", "independent_source",
                     "target_continuation"):
            model = VelocityNet(args.width).to(device)
            model.load_state_dict(initial)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            before = None
            if name in ("independent_source", "target_continuation"):
                data = source_train if name == "independent_source" else target_train
                batches = source_batches if name == "independent_source" else target_pre_batches
                optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
                model.train()
                for batch in batches:
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(data[0][batch]) - data[1][batch]).square().mean()
                    loss.backward()
                    optimizer.step()
                before = rmse_mae(model, target_val, device)
            # Start a fresh optimizer after pretraining, as in the legacy code.
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
            model.train()
            # Equal total update count for scratch_matched_compute, with one
            # constant target learning rate; target_continuation is the tighter
            # control for pretraining data because its two-stage schedule matches.
            batches = (np.concatenate((target_pre_batches, target_batches))
                       if name == "scratch_matched_compute" else target_batches)
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = (model(target_train[0][batch])
                        - target_train[1][batch]).square().mean()
                loss.backward()
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            arms[name] = {"validation_rmse_before_target_m_s":
                          before["rmse_m_s"] if before else None,
                          "validation": rmse_mae(model, target_val, device),
                          "spatial_test": rmse_mae(model, target_test, device),
                          "runtime_s": time.perf_counter() - start}
            if args.checkpoint_dir is not None:
                args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = args.checkpoint_dir / f"seed_{seed}_{name}.pth"
                torch.save({"model_state_dict": model.cpu().state_dict(),
                            "width": args.width, "seed": seed, "arm": name,
                            "velocity_offset_m_s": 4000.0,
                            "velocity_scale_m_s": 2000.0,
                            "target_shape": tuple(volume.shape)}, checkpoint)
                arms[name]["checkpoint"] = str(checkpoint.resolve())
        results.append({"seed": seed, "arms": arms})
        print(f"seed {seed}: test RMSE (m/s) "
              + ", ".join(f"{name}={result['spatial_test']['rmse_m_s']:.1f}"
                          for name, result in arms.items()), flush=True)
    digest = hashlib.sha256()
    with args.target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    names = results[0]["arms"]
    summary = {}
    for name in names:
        summary[name] = {}
        for metric in ("rmse_m_s", "mae_m_s"):
            samples = np.array([run["arms"][name]["spatial_test"][metric]
                                for run in results])
            summary[name][metric] = {"mean": float(samples.mean()),
                                     "std": float(samples.std(ddof=1)) if len(samples) > 1 else None}
        runtimes = np.array([run["arms"][name]["runtime_s"] for run in results])
        summary[name]["runtime_s"] = {"mean": float(runtimes.mean()),
                                      "std": float(runtimes.std(ddof=1)) if len(runtimes) > 1 else None}
    for control in ("scratch", "scratch_matched_compute", "target_continuation"):
        differences = np.array([
            run["arms"]["independent_source"]["spatial_test"]["rmse_m_s"]
            - run["arms"][control]["spatial_test"]["rmse_m_s"] for run in results])
        summary[f"independent_source_minus_{control}_rmse_m_s"] = {
            "mean": float(differences.mean()),
            "std": float(differences.std(ddof=1)) if len(differences) > 1 else None,
            "paired_values": differences.tolist()}
    report = {
        "purpose": "small independent-source transfer feasibility audit; not historical reproduction",
        "target_reference": str(args.target.resolve()),
        "target_sha256": digest.hexdigest(), "source": "analytic independent 1-D layered gradient",
        "source_formula_m_s": "3000 + 1700*z + 300*tanh(8*(z-0.55)); z in [0,1]",
        "split": {"train_x_indices": len(train_ix), "validation_x_indices": len(val_ix),
                  "test_x_indices": len(test_ix), "sample_seed": 20260924},
        "configuration": {"seeds": args.seeds, "source_steps": args.source_steps,
                          "target_steps": args.target_steps, "train_count": args.train_count,
                          "eval_count": args.eval_count, "batch_size": args.batch_size,
                          "width": args.width, "device": str(device),
                          "target_learning_rate": 2e-5,
                          "pretrain_learning_rate": 2e-4,
                          "scratch_matched_compute_learning_rate": 2e-5,
                          "scratch_matched_compute_steps": args.source_steps + args.target_steps},
        "results": results, "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
