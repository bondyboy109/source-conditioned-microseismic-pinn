"""Robust-picker FMM-seeded MoE refinement at one noisy SNR."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import skfmm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from audit_fmm_traveltime import first_arrival, trilinear
from run_holdout_noise_picker_comparison import robust_arrival
sys.path.insert(0, str(ROOT))
from mixture_of_experts import GatedResidualMixtureOfExperts
from source_conditioned_pinn import load_conditioned_checkpoint

BASE = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth"
MOE = ROOT / "outputs/joint_moe_physics_development/joint_moe_physics.pth"
OUT = ROOT / "experiments/holdout_20260927_noise_hybrid_results.json"
VELOCITY = ROOT / "data/velocity/vp_model.npy"
OBSERVATIONS = ROOT / "data/observations/holdout_20260927"
DOMAIN = np.asarray([16000., 16000., 3720., 9.], dtype=np.float32)
BOUNDS = np.asarray([[2000., 14000.], [2000., 14000.], [100., 1500.], [0.05, 0.5]], dtype=np.float32)
SPACING = 160.0
EVENTS = ["event_01.npz", "event_03.npz", "event_06.npz", "event_08.npz"]
SNR_DB = 10.0
SEEDS = [20260927, 20260928]


def fields_for(velocity, candidates, receivers):
    output = []
    for xyz in candidates:
        idx = np.clip(np.rint(xyz / SPACING).astype(int), 0, np.asarray(velocity.shape) - 1)
        level = np.ones(velocity.shape, dtype=np.float64)
        level[tuple(idx)] = -1.0
        field = np.asarray(skfmm.travel_time(level, speed=velocity, dx=SPACING), dtype=np.float64)
        output.append(trilinear(field, receivers, SPACING))
    return np.asarray(output)


def waveform(model, source, receiver_norm, time_norm, device):
    points = np.concatenate((np.broadcast_to(receiver_norm[:, None, :], (len(receiver_norm), len(time_norm), 3)),
                             np.broadcast_to(time_norm[None, :, None], (len(receiver_norm), len(time_norm), 1))), axis=-1).reshape(-1, 4)
    src = torch.as_tensor(source / DOMAIN, dtype=torch.float32, device=device)
    values = []
    with torch.inference_mode():
        for start in range(0, len(points), 8192):
            x = torch.as_tensor(points[start:start + 8192], dtype=torch.float32, device=device)
            values.append(model(x, src.expand(x.shape[0], -1), freeze_base=True)[0])
    return torch.cat(values).reshape(len(receiver_norm), len(time_norm))


def choose(picks, candidates, travel):
    valid = np.isfinite(picks)
    best = None
    for xyz, tt in zip(candidates, travel):
        origin = float(np.clip(np.median(picks[valid] - tt[valid]), 0.05, 0.5))
        rmse = float(np.sqrt(np.mean((origin + tt[valid] - picks[valid]) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, np.r_[xyz, origin])
    return best


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _ = load_conditioned_checkpoint(BASE, device)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    spec = torch.load(MOE, map_location=device, weights_only=False)
    model = GatedResidualMixtureOfExperts(base, expert_count=int(spec["spec"]["expert_count"]),
        hidden_dim=int(spec["spec"]["expert_hidden_dim"]), hidden_layers=int(spec["spec"]["expert_hidden_layers"]),
        feature_scale=float(spec["spec"]["feature_scale"])).to(device)
    model.load_state_dict(spec["state_dict"]); model.eval()
    velocity = np.asarray(np.load(VELOCITY), dtype=np.float64)[::4, ::4, ::4]
    axes = [np.linspace(low, high, count, dtype=np.float64) for (low, high), count in zip(BOUNDS[:3], (6, 6, 4))]
    coarse_candidates = np.asarray(np.meshgrid(*axes, indexing="ij"), dtype=np.float64).reshape(3, -1).T
    rows = []
    started = time.perf_counter()
    for event_index, event_name in enumerate(EVENTS):
        with np.load(OBSERVATIONS / event_name) as archive:
            clean = np.asarray(archive["data"], dtype=np.float64)
            times = np.asarray(archive["times"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        travel_coarse = fields_for(velocity, coarse_candidates, receivers)
        signal_rms = float(np.sqrt(np.mean(clean ** 2)))
        for seed in SEEDS:
            rng = np.random.default_rng(seed + int(SNR_DB * 100) + event_index * 1000)
            noisy = clean + rng.normal(0.0, signal_rms / (10.0 ** (SNR_DB / 20.0)), size=clean.shape)
            picks = np.asarray([robust_arrival(trace, times) for trace in noisy])
            fmm_best = choose(picks, coarse_candidates, travel_coarse)
            center = fmm_best[1][:3]
            local_candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        local_candidates.append(np.clip(center + np.asarray([dx, dy, dz]) * [2400., 2400., 466.6667], BOUNDS[:3, 0], BOUNDS[:3, 1]))
            local_candidates = np.unique(np.asarray(local_candidates), axis=0)
            local_travel = fields_for(velocity, local_candidates, receivers)
            receiver_norm = (receivers / DOMAIN[:3]).astype(np.float32)[::2]
            time_norm = (times / DOMAIN[3]).astype(np.float32)[::8]
            observed = torch.as_tensor(noisy[::2, ::8], dtype=torch.float32, device=device)
            rms = float(np.sqrt(np.mean(noisy ** 2)))
            scored = []
            for xyz, tt in zip(local_candidates, local_travel):
                origin = float(np.clip(np.median(picks - tt), 0.05, 0.5))
                source = np.r_[xyz, origin].astype(np.float32)
                prediction = waveform(model, source, receiver_norm, time_norm, device)
                wave_loss = float(((prediction - observed) / rms).square().mean().detach())
                arrival_rmse = float(np.sqrt(np.mean((origin + tt - picks) ** 2)))
                scored.append({"source_m_s": source.tolist(), "waveform_loss": wave_loss,
                               "arrival_rmse_s": arrival_rmse,
                               "hybrid_loss": wave_loss + 0.25 * (arrival_rmse / 0.10) ** 2})
            fmm_source = np.asarray(fmm_best[1])
            hybrid = min(scored, key=lambda row: row["hybrid_loss"])
            wave = min(scored, key=lambda row: row["waveform_loss"])
            row = {"event": event_name, "seed": seed, "snr_db": SNR_DB,
                   "truth_source_m_s": truth.tolist(), "fmm_source_m_s": fmm_source.tolist(),
                   "hybrid_source_m_s": hybrid["source_m_s"], "waveform_source_m_s": wave["source_m_s"],
                   "fmm_location_error_m": float(np.linalg.norm(fmm_source[:3] - truth[:3])),
                   "hybrid_location_error_m": float(np.linalg.norm(np.asarray(hybrid["source_m_s"])[:3] - truth[:3])),
                   "waveform_location_error_m": float(np.linalg.norm(np.asarray(wave["source_m_s"])[:3] - truth[:3])),
                   "fmm_origin_time_error_s": float(abs(fmm_source[3] - truth[3])),
                   "hybrid_origin_time_error_s": float(abs(hybrid["source_m_s"][3] - truth[3])),
                   "waveform_origin_time_error_s": float(abs(wave["source_m_s"][3] - truth[3]))}
            rows.append(row); print(event_name, seed, row["fmm_location_error_m"], row["hybrid_location_error_m"], flush=True)
    summary = {}
    for key in ("fmm", "hybrid", "waveform"):
        summary[key] = {"mean_location_error_m": float(np.mean([row[f"{key}_location_error_m"] for row in rows])),
                        "mean_origin_time_error_s": float(np.mean([row[f"{key}_origin_time_error_s"] for row in rows]))}
    result = {"role": "SNR=10 dB robust-picker FMM-seeded MoE refinement",
              "events": EVENTS, "seeds": SEEDS, "snr_db": SNR_DB, "summary": summary,
              "records": rows, "runtime_seconds": time.perf_counter() - started,
              "limits": "Eight noisy realizations, one MoE checkpoint, coarse FMM/local grids, and one hybrid weight. Diagnostic only; no final localization claim."}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
