"""Decompose one holdout MoE objective into time and spatial source effects."""

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
MOE = ROOT / "outputs/joint_moe_physics_development/joint_moe_physics.pth"
OBS = ROOT / "data/observations/holdout_20260927/event_01.npz"
OUT = ROOT / "experiments/holdout_20260927_moe_objective_diagnostic.json"
DOMAIN = np.asarray([16000., 16000., 3720., 9.], dtype=np.float32)
BOUNDS = np.asarray([[2000., 14000.], [2000., 14000.], [100., 1500.], [0.05, 0.5]], dtype=np.float32)


def predict(model, source, receiver_norm, time_norm, device):
    points = np.concatenate((
        np.broadcast_to(receiver_norm[:, None, :], (len(receiver_norm), len(time_norm), 3)),
        np.broadcast_to(time_norm[None, :, None], (len(receiver_norm), len(time_norm), 1)),
    ), axis=-1).reshape(-1, 4)
    outputs = []
    for start in range(0, len(points), 8192):
        x = torch.as_tensor(points[start:start + 8192], device=device)
        s = source.expand(x.shape[0], -1)
        outputs.append(model(x, s, freeze_base=True)[0])
    return torch.cat(outputs).reshape(len(receiver_norm), len(time_norm))


def loss_at(model, physical_source, receiver_norm, time_norm, observed, rms, device):
    source = torch.as_tensor(np.asarray(physical_source, dtype=np.float32) / DOMAIN,
                             dtype=torch.float32, device=device)
    estimate = predict(model, source, receiver_norm, time_norm, device)
    return float(((estimate - observed) / rms).square().mean().detach())


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _ = load_conditioned_checkpoint(BASE, device)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    spec = torch.load(MOE, map_location=device, weights_only=False)
    model = GatedResidualMixtureOfExperts(
        base, expert_count=int(spec["spec"]["expert_count"]),
        hidden_dim=int(spec["spec"]["expert_hidden_dim"]),
        hidden_layers=int(spec["spec"]["expert_hidden_layers"]),
        feature_scale=float(spec["spec"]["feature_scale"]),
    ).to(device)
    model.load_state_dict(spec["state_dict"])
    model.eval()
    with np.load(OBS) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        truth = np.asarray(archive["source_truth"], dtype=np.float32)
    receiver_norm = receivers[::2] / DOMAIN[:3]
    time_norm = times[::8] / DOMAIN[3]
    observed = torch.as_tensor(data[::2, ::8], device=device)
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))

    time_values = np.linspace(BOUNDS[3, 0], BOUNDS[3, 1], 19, dtype=np.float32)
    time_curve = [{"time_s": float(t), "loss": loss_at(model, truth[:3].tolist() + [float(t)], receiver_norm, time_norm, observed, rms, device)} for t in time_values]

    xyz_step = np.asarray([1200., 1200., 140.], dtype=np.float32)
    spatial_rows = []
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            for dz in (-2, -1, 0, 1, 2):
                xyz = truth[:3] + np.asarray([dx, dy, dz], dtype=np.float32) * xyz_step
                if np.all(xyz >= BOUNDS[:3, 0]) and np.all(xyz <= BOUNDS[:3, 1]):
                    spatial_rows.append({"source_m": xyz.tolist(), "loss": loss_at(model, xyz.tolist() + [float(truth[3])], receiver_norm, time_norm, observed, rms, device)})
    spatial_rows.sort(key=lambda row: row["loss"])
    result = {
        "role": "objective-landscape diagnostic; no model fitting",
        "observation": str(OBS),
        "truth_source_m_s": truth.tolist(),
        "truth_loss": loss_at(model, truth.tolist(), receiver_norm, time_norm, observed, rms, device),
        "time_curve_at_true_xyz": time_curve,
        "best_time_at_true_xyz": min(time_curve, key=lambda row: row["loss"]),
        "spatial_curve_at_true_time_best_five": spatial_rows[:5],
        "spatial_loss_at_truth": next(row["loss"] for row in spatial_rows if np.allclose(row["source_m"], truth[:3])),
        "limits": "One held-out event and one checkpoint; diagnostic only.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"truth_loss": result["truth_loss"], "best_time_at_true_xyz": result["best_time_at_true_xyz"], "best_spatial": result["spatial_curve_at_true_time_best_five"][0]}, indent=2))


if __name__ == "__main__":
    main()
