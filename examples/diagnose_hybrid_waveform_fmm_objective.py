"""Compare waveform and FMM-arrival objective landscapes on one holdout event."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import skfmm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from audit_fmm_traveltime import first_arrival, trilinear
from mixture_of_experts import GatedResidualMixtureOfExperts
from source_conditioned_pinn import load_conditioned_checkpoint

BASE = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth"
MOE = ROOT / "outputs/joint_moe_physics_development/joint_moe_physics.pth"
OBS = ROOT / "data/observations/holdout_20260927/event_01.npz"
OUT = ROOT / "experiments/holdout_20260927_hybrid_objective_diagnostic.json"
DOMAIN = np.asarray([16000., 16000., 3720., 9.], dtype=np.float32)
BOUNDS = np.asarray([[2000., 14000.], [2000., 14000.], [100., 1500.], [0.05, 0.5]], dtype=np.float32)
FMM_SPACING = 160.0


def waveform_prediction(model, source_m_s, receiver_norm, time_norm, device):
    points = np.concatenate((
        np.broadcast_to(receiver_norm[:, None, :], (len(receiver_norm), len(time_norm), 3)),
        np.broadcast_to(time_norm[None, :, None], (len(receiver_norm), len(time_norm), 1)),
    ), axis=-1).reshape(-1, 4)
    source = torch.as_tensor(source_m_s / DOMAIN, dtype=torch.float32, device=device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(points), 8192):
            x = torch.as_tensor(points[start:start + 8192], dtype=torch.float32, device=device)
            outputs.append(model(x, source.expand(x.shape[0], -1), freeze_base=True)[0])
    return torch.cat(outputs).reshape(len(receiver_norm), len(time_norm))


def fmm_arrivals(velocity, source_xyz, receivers):
    index = np.rint(source_xyz / FMM_SPACING).astype(int)
    index = np.clip(index, 0, np.asarray(velocity.shape) - 1)
    level = np.ones(velocity.shape, dtype=np.float64)
    level[tuple(index)] = -1.0
    field = np.asarray(skfmm.travel_time(level, speed=velocity, dx=FMM_SPACING), dtype=np.float64)
    return trilinear(field, receivers, FMM_SPACING)


def main() -> None:
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
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
        truth = np.asarray(archive["source_truth"], dtype=np.float64)
    receiver_norm = (receivers / DOMAIN[:3]).astype(np.float32)[::2]
    time_norm = (times / DOMAIN[3]).astype(np.float32)[::8]
    observed = torch.as_tensor(data[::2, ::8], dtype=torch.float32, device=device)
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    picks = np.asarray([first_arrival(trace, times) for trace in data], dtype=np.float64)
    valid = np.isfinite(picks)
    velocity = np.asarray(np.load(ROOT / "data/velocity/vp_model.npy"), dtype=np.float64)[::4, ::4, ::4]
    true_travel = fmm_arrivals(velocity, truth[:3], receivers)
    true_predicted = truth[3] + true_travel
    true_arrival_rmse = float(np.sqrt(np.mean((true_predicted[valid] - picks[valid]) ** 2)))

    axes = [np.linspace(low, high, count, dtype=np.float32)
            for (low, high), count in zip(BOUNDS[:3], (6, 6, 4))]
    candidates_xyz = np.asarray(np.meshgrid(*axes, indexing="ij"), dtype=np.float32).reshape(3, -1).T
    candidates = []
    for xyz in candidates_xyz:
        travel = fmm_arrivals(velocity, xyz, receivers)
        origin = float(np.median(picks[valid] - travel[valid]))
        origin = float(np.clip(origin, *BOUNDS[3]))
        arrival_residual = origin + travel[valid] - picks[valid]
        arrival_rmse = float(np.sqrt(np.mean(arrival_residual ** 2)))
        source = np.r_[xyz, origin].astype(np.float32)
        prediction = waveform_prediction(model, source, receiver_norm, time_norm, device)
        waveform_loss = float(((prediction - observed) / rms).square().mean().detach())
        candidates.append({"source_m_s": source.tolist(), "waveform_loss": waveform_loss,
                           "arrival_rmse_s": arrival_rmse, "origin_time_s": origin})

    candidates.sort(key=lambda row: row["waveform_loss"])
    top_wave = candidates[0]
    for row in candidates:
        row["arrival_normalized_loss"] = (row["arrival_rmse_s"] / 0.10) ** 2
        row["hybrid_loss_alpha_0.25"] = row["waveform_loss"] + 0.25 * row["arrival_normalized_loss"]
        row["hybrid_loss_alpha_1.0"] = row["waveform_loss"] + row["arrival_normalized_loss"]
    top_arrival = min(candidates, key=lambda row: row["arrival_rmse_s"])
    top_hybrid_025 = min(candidates, key=lambda row: row["hybrid_loss_alpha_0.25"])
    top_hybrid_1 = min(candidates, key=lambda row: row["hybrid_loss_alpha_1.0"])
    result = {
        "role": "waveform plus FMM-arrival objective diagnostic; no retraining",
        "observation": str(OBS),
        "truth_source_m_s": truth.tolist(),
        "fmm_spacing_m": FMM_SPACING,
        "valid_pick_count": int(valid.sum()),
        "true_source_fmm_arrival_rmse_s": true_arrival_rmse,
        "candidate_count": len(candidates),
        "best_waveform": top_wave,
        "best_arrival": top_arrival,
        "best_hybrid_alpha_0.25": top_hybrid_025,
        "best_hybrid_alpha_1.0": top_hybrid_1,
        "top_waveform_candidates": sorted(candidates, key=lambda row: row["waveform_loss"])[:5],
        "top_hybrid_candidates": sorted(candidates, key=lambda row: row["hybrid_loss_alpha_0.25"])[:5],
        "limits": "Threshold picks are not independent physical arrival labels; one event, one seed, coarse FMM grid, and no localization claim.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("true_source_fmm_arrival_rmse_s", "best_waveform", "best_arrival",
                       "best_hybrid_alpha_0.25", "best_hybrid_alpha_1.0")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
