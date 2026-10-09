"""Evaluate the SNR=10 dB noise-augmented MoE on clean and noisy holdouts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixture_of_experts import GatedResidualMixtureOfExperts
from source_conditioned_pinn import load_conditioned_checkpoint

BASE = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth"
NOISE_MOE = ROOT / "outputs/joint_moe_noise10_development/joint_moe_noise10.pth"
OBS = ROOT / "data/observations/holdout_20260927"
OUT = ROOT / "experiments/holdout_20260927_noise_moe_forward_results_v2.json"
DOMAIN = np.asarray([16000., 16000., 3720., 9.], dtype=np.float32)
EVENTS = ["event_01.npz", "event_03.npz", "event_06.npz", "event_08.npz"]
SNR_DB = 10.0
SEEDS = [20260927, 20260928]


def predict(model, source, receivers, times, device, is_moe=False):
    receiver_norm = receivers / DOMAIN[:3]
    time_norm = times / DOMAIN[3]
    points = np.concatenate((np.broadcast_to(receiver_norm[:, None, :], (len(receivers), len(times), 3)),
                             np.broadcast_to(time_norm[None, :, None], (len(receivers), len(times), 1))), axis=-1).reshape(-1, 4)
    src = torch.as_tensor(source / DOMAIN, dtype=torch.float32, device=device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(points), 8192):
            x = torch.as_tensor(points[start:start + 8192], dtype=torch.float32, device=device)
            if is_moe:
                outputs.append(model(x, src.expand(x.shape[0], -1), freeze_base=True)[0])
            else:
                outputs.append(model(x, src.expand(x.shape[0], -1)))
    return torch.cat(outputs).reshape(len(receivers), len(times)).cpu().numpy()


def l2(reference, estimate):
    return float(np.sqrt(np.sum((estimate - reference) ** 2) / max(np.sum(reference ** 2), 1e-30)))


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _ = load_conditioned_checkpoint(BASE, device)
    for parameter in base.parameters(): parameter.requires_grad_(False)
    payload = torch.load(NOISE_MOE, map_location=device, weights_only=False)
    model = GatedResidualMixtureOfExperts(base, expert_count=3, hidden_dim=128,
        hidden_layers=3, feature_scale=1e6).to(device)
    model.load_state_dict(payload["state_dict"]); model.eval()
    rows = []
    for event_index, name in enumerate(EVENTS):
        with np.load(OBS / name) as archive:
            clean = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            source = np.asarray(archive["source_truth"], dtype=np.float32)
        base_prediction = predict(base, source, receivers, times, device, is_moe=False)
        noise_prediction = predict(model, source, receivers, times, device, is_moe=True)
        signal_rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))
        for seed in SEEDS:
            rng = np.random.default_rng(seed + int(SNR_DB * 100) + event_index * 1000)
            noisy = clean + rng.normal(0.0, signal_rms / (10.0 ** (SNR_DB / 20.0)), size=clean.shape).astype(np.float32)
            rows.append({"event": name, "seed": seed,
                         "base_vs_clean_l2": l2(clean, base_prediction),
                         "noise_moe_vs_clean_l2": l2(clean, noise_prediction),
                         "base_vs_noisy_l2": l2(noisy, base_prediction),
                         "noise_moe_vs_noisy_l2": l2(noisy, noise_prediction)})
    summary = {key: float(np.mean([row[key] for row in rows]))
               for key in ("base_vs_clean_l2", "noise_moe_vs_clean_l2", "base_vs_noisy_l2", "noise_moe_vs_noisy_l2")}
    result = {"role": "noise-augmented MoE forward holdout audit", "snr_db": SNR_DB,
              "events": EVENTS, "seeds": SEEDS, "summary": summary, "records": rows,
              "checkpoint": str(NOISE_MOE),
              "limits": "Four independent events, two noisy realizations, one seed and receiver-only noise augmentation; no localization or UQ claim."}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
