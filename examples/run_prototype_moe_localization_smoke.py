"""Bounded source inversion smoke test for the joint MoE on one new holdout."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixture_of_experts import PrototypeGatedResidualMixtureOfExperts
from source_conditioned_pinn import load_conditioned_checkpoint

BASE = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth"
MOE = ROOT / "outputs/prototype_moe_development/joint_moe_physics.pth"
OBS = ROOT / "data/observations/holdout_20260927/event_01.npz"
OUT = ROOT / "experiments/prototype_moe_localization_smoke.json"
DOMAIN = np.asarray([16000.0, 16000.0, 3720.0, 9.0], dtype=np.float32)
BOUNDS = np.asarray([[2000.0, 14000.0], [2000.0, 14000.0],
                     [100.0, 1500.0], [0.05, 0.5]], dtype=np.float32)


def predict(model, source, receiver_norm, time_norm, device, stride=1):
    points = np.concatenate((
        np.broadcast_to(receiver_norm[:, None, :],
                        (len(receiver_norm), len(time_norm), 3)),
        np.broadcast_to(time_norm[None, :, None],
                        (len(receiver_norm), len(time_norm), 1)),
    ), axis=-1).reshape(-1, 4)
    output = []
    with torch.set_grad_enabled(True):
        for start in range(0, len(points), 8192):
            x = torch.as_tensor(points[start:start + 8192], device=device)
            s = source.expand(x.shape[0], -1)
            y, _ = model(x, s, freeze_base=True)
            output.append(y)
    return torch.cat(output, dim=0).reshape(len(receiver_norm), len(time_norm))


def objective(model, source_parameter, receiver_norm, time_norm, observed, rms, device):
    lower = torch.as_tensor(BOUNDS[:, 0] / DOMAIN, device=device)
    width = torch.as_tensor((BOUNDS[:, 1] - BOUNDS[:, 0]) / DOMAIN, device=device)
    source = lower + width * source_parameter
    prediction = predict(model, source, receiver_norm, time_norm, device)
    return ((prediction - observed) / rms).square().mean(), source


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, payload = load_conditioned_checkpoint(BASE, device)
    for p in base.parameters():
        p.requires_grad_(False)
    spec = torch.load(MOE, map_location=device, weights_only=False)
    model = PrototypeGatedResidualMixtureOfExperts(
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
    receiver_norm = receivers / DOMAIN[:3]
    time_norm = times / DOMAIN[3]
    receiver_norm = receiver_norm[::2]
    time_norm = time_norm[::8]
    observed = torch.as_tensor(data[::2, ::8], device=device)
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))

    # Coarse screening uses a small fixed grid and does not use the truth.
    starts = []
    for x in np.linspace(0.0, 1.0, 6, dtype=np.float32):
        for y in np.linspace(0.0, 1.0, 6, dtype=np.float32):
            for z in np.linspace(0.0, 1.0, 4, dtype=np.float32):
                for t in np.linspace(0.0, 1.0, 4, dtype=np.float32):
                    starts.append(np.asarray([x, y, z, t], dtype=np.float32))
    scored = []
    for start in starts:
        with torch.no_grad():
            loss, source = objective(model, torch.as_tensor(start, device=device),
                                     receiver_norm, time_norm, observed, rms, device)
        scored.append((float(loss), start, source.detach().cpu().numpy()))
    scored.sort(key=lambda row: row[0])

    refined = []
    started = time.perf_counter()
    for coarse_loss, start, coarse_source in scored[:3]:
        parameter = torch.nn.Parameter(torch.as_tensor(start, device=device))
        optimizer = torch.optim.Adam([parameter], lr=0.02)
        best = None
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss, source = objective(model, parameter.clamp(0, 1), receiver_norm,
                                    time_norm, observed, rms, device)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                parameter.clamp_(0, 1)
            value = float(loss.detach())
            if best is None or value < best[0]:
                best = (value, source.detach().cpu().numpy().copy())
        refined.append({"coarse_source_m_s": (coarse_source * DOMAIN).tolist(),
                        "coarse_loss": coarse_loss,
                        "estimated_source_m_s": (best[1] * DOMAIN).tolist(),
                        "final_loss": best[0]})
    final = min(refined, key=lambda row: row["final_loss"])
    estimate = np.asarray(final["estimated_source_m_s"], dtype=np.float32)
    gradient_parameter = torch.tensor(scored[0][1], device=device, requires_grad=True)
    gradient_loss, _ = objective(model, gradient_parameter, receiver_norm, time_norm,
                                 observed, rms, device)
    gradient = torch.autograd.grad(gradient_loss, gradient_parameter)[0].detach().cpu().numpy()
    result = {
        "role": "one-event localization smoke test after independent forward holdout",
        "observation": str(OBS),
        "source_truth_m_s": truth.tolist(),
        "estimated_source_m_s": estimate.tolist(),
        "component_error": (estimate - truth).tolist(),
        "location_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
        "origin_time_error_s": float(abs(estimate[3] - truth[3])),
        "selected_coarse_loss": float(scored[0][0]),
        "selected_final_loss": float(final["final_loss"]),
        "truth_loss": float(objective(model, torch.as_tensor((truth - BOUNDS[:, 0]) /
                                                               (BOUNDS[:, 1] - BOUNDS[:, 0]), device=device),
                                       receiver_norm, time_norm, observed, rms, device)[0].detach()),
        "coarse_source_gradient": gradient.tolist(),
        "all_gradient_paths_finite": bool(np.isfinite(gradient).all()),
        "refinements": refined,
        "limits": "One holdout event, one checkpoint and one random seed; this is a smoke test, not evidence of publication-level localization accuracy or UQ calibration.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"runtime_seconds={time.perf_counter() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
