"""Fine-tune gated residual experts with SNR=10 dB receiver augmentation."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixture_of_experts import GatedResidualMixtureOfExperts
from receiver_training import ReceiverTrainingData
from source_conditioned_pinn import SourceConditionedConfig, load_conditioned_checkpoint

BASE = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth"
OUT_DIR = ROOT / "outputs/joint_moe_noise10_development"
CHECKPOINT = OUT_DIR / "joint_moe_noise10.pth"
RESULT = ROOT / "experiments/joint_moe_noise10_protocol.json"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if CHECKPOINT.exists() or RESULT.exists():
        raise FileExistsError("Refusing to overwrite noise-augmented outputs")
    seed = 20260927
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, payload = load_conditioned_checkpoint(BASE, device)
    base.eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    cfg = SourceConditionedConfig()
    manifest = payload.get("training_record", {}).get("receiver_observations", [])
    paths = [Path(row["path"]) for row in manifest]
    if len(paths) != 33: raise ValueError("Expected 33 registered observations")
    for row, path in zip(manifest, paths):
        if sha256(path) != row["sha256"]: raise ValueError(f"Training record changed: {path}")
    data = ReceiverTrainingData(paths, device, cfg, 5.0, seed=seed)
    model = GatedResidualMixtureOfExperts(base, expert_count=3, hidden_dim=128,
        hidden_layers=3, feature_scale=1e6).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    noise_std = float(10.0 ** (-10.0 / 20.0))
    history = []
    started = time.perf_counter()
    for update in range(1, 1001):
        model.train(); optimizer.zero_grad(set_to_none=True)
        space, target, source, rms = data.batch(2048, 8)
        noisy_target = target + noise_std * torch.randn_like(target)
        predicted, gates = model(space, source, freeze_base=True)
        loss_receiver = ((predicted.squeeze(-1) / rms - noisy_target) ** 2).mean()
        gate_mean = gates.mean(dim=0)
        loss_balance = ((gate_mean - 1.0 / model.expert_count) ** 2).mean()
        total = loss_receiver + 0.01 * loss_balance
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if update == 1 or update % 100 == 0:
            row = {"update": update, "total": float(total.detach()),
                   "receiver": float(loss_receiver.detach()), "gate_balance": float(loss_balance.detach()),
                   "mean_gate": gate_mean.detach().cpu().tolist()}
            history.append(row); print(row, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "spec": {"expert_count": 3, "expert_hidden_dim": 128,
                         "expert_hidden_layers": 3, "feature_scale": 1e6,
                         "noise_snr_db": 10.0, "seed": seed},
                "base_checkpoint_sha256": sha256(BASE),
                "training_history": history}, CHECKPOINT)
    result = {"role": "noise-augmented receiver-only MoE fine-tuning",
              "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
              "base_checkpoint_sha256": sha256(BASE), "noise_snr_db": 10.0,
              "updates": 1000, "training_seconds": time.perf_counter() - started,
              "history": history,
              "limits": "One seed, clean training records with synthetic SNR=10 dB augmentation; independent holdout evaluation is separate."}
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("checkpoint", "training_seconds", "noise_snr_db")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
