"""Evaluate the frozen source-conditioned base and joint MoE on new FD holdouts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixture_of_experts import GatedResidualMixtureOfExperts
from source_conditioned_pinn import load_conditioned_checkpoint


BASE_CHECKPOINT = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth"
MOE_CHECKPOINT = ROOT / "outputs/joint_moe_physics_development/joint_moe_physics.pth"
OBSERVATION_DIR = ROOT / "data/observations/holdout_20260927"
RESULT_PATH = ROOT / "experiments/holdout_20260927_moe_results_v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_prediction(model, source, shape, receivers, times, domain, device):
    coords = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*shape, 3)),
        np.broadcast_to(times[None, :, None], (*shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    src = np.broadcast_to(source / domain, (len(coords), 4)).astype(np.float32)
    outputs, gates = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coords), 8192):
            x = torch.as_tensor(coords[start:start + 8192], device=device)
            s = torch.as_tensor(src[start:start + 8192], device=device)
            y, gate = model(x, s, freeze_base=True)
            outputs.append(y.squeeze(-1).cpu().numpy())
            gates.append(gate.cpu().numpy())
    return np.concatenate(outputs).reshape(shape), np.concatenate(gates, axis=0)


def base_prediction(base_model, source, shape, receivers, times, domain, device):
    coords = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*shape, 3)),
        np.broadcast_to(times[None, :, None], (*shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    src = np.broadcast_to(source / domain, (len(coords), 4)).astype(np.float32)
    outputs = []
    base_model.eval()
    with torch.inference_mode():
        for start in range(0, len(coords), 8192):
            x = torch.as_tensor(coords[start:start + 8192], device=device)
            s = torch.as_tensor(src[start:start + 8192], device=device)
            outputs.append(base_model(x, s).squeeze(-1).cpu().numpy())
    return np.concatenate(outputs).reshape(shape)


def metrics(observed, predicted):
    reference = observed.astype(np.float64)
    estimate = predicted.astype(np.float64)
    energy = max(float(np.sum(reference ** 2)), 1e-30)
    active = np.abs(reference) >= 0.05 * np.max(np.abs(reference))
    active_energy = max(float(np.sum(reference[active] ** 2)), 1e-30)
    return {
        "full_l2": float(np.sqrt(np.sum((estimate - reference) ** 2) / energy)),
        "active_l2": float(np.sqrt(np.sum((estimate[active] - reference[active]) ** 2)
                                   / active_energy)),
        "zero_lag_correlation": float(np.sum(estimate * reference)
                                       / max(np.linalg.norm(estimate) * np.linalg.norm(reference), 1e-30)),
    }


def main() -> None:
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite {RESULT_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, payload = load_conditioned_checkpoint(BASE_CHECKPOINT, device)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    spec = torch.load(MOE_CHECKPOINT, map_location=device, weights_only=False)
    model = GatedResidualMixtureOfExperts(
        base_model, expert_count=int(spec["spec"]["expert_count"]),
        hidden_dim=int(spec["spec"]["expert_hidden_dim"]),
        hidden_layers=int(spec["spec"]["expert_hidden_layers"]),
        feature_scale=float(spec["spec"]["feature_scale"]),
    ).to(device)
    model.load_state_dict(spec["state_dict"])
    domain = np.asarray([payload["config"][key] for key in
                         ("x_length_m", "y_length_m", "z_length_m", "time_length_s")], dtype=np.float32)

    events = []
    for path in sorted(OBSERVATION_DIR.glob("event_*.npz")):
        with np.load(path) as archive:
            observed = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            source = np.asarray(archive["source_truth"], dtype=np.float32)
        base = base_prediction(base_model, source, observed.shape, receivers, times, domain, device)
        moe, gate = make_prediction(model, source, observed.shape, receivers, times, domain, device)
        events.append({
            "event": path.name,
            "observation": str(path),
            "sha256": sha256(path),
            "source_m_s": source.tolist(),
            "base": metrics(observed, base),
            "moe": metrics(observed, moe),
            "mean_gate": gate.mean(axis=0).tolist(),
            "moe_improves": bool(metrics(observed, moe)["full_l2"] < metrics(observed, base)["full_l2"]),
        })

    def mean(branch, metric):
        return float(np.mean([event[branch][metric] for event in events]))

    base_mean = mean("base", "full_l2")
    moe_mean = mean("moe", "full_l2")
    gate_usage = np.mean([event["mean_gate"] for event in events], axis=0)
    summary = {
        "event_count": len(events),
        "base_mean_full_l2": base_mean,
        "moe_mean_full_l2": moe_mean,
        "base_mean_active_l2": mean("base", "active_l2"),
        "moe_mean_active_l2": mean("moe", "active_l2"),
        "relative_improvement_over_base": float((base_mean - moe_mean) / base_mean),
        "moe_improves_count": int(sum(event["moe_improves"] for event in events)),
        "mean_gate_usage": gate_usage.tolist(),
        "gate_collapsed": bool(np.max(gate_usage) > 0.90),
    }
    result = {
        "role": "independent holdout forward audit; no localization",
        "base_checkpoint_sha256": sha256(BASE_CHECKPOINT),
        "moe_checkpoint_sha256": sha256(MOE_CHECKPOINT),
        "observation_directory": str(OBSERVATION_DIR),
        "summary": summary,
        "events": events,
        "limits": "These four records were generated after the development audit and were not used for MoE training. This is a forward-only, one-seed holdout and does not establish localization or calibrated UQ.",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
