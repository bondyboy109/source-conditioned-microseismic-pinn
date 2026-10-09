"""Evaluate seeded joint-MoE checkpoints on the independent eight-event holdout."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixture_of_experts import PrototypeGatedResidualMixtureOfExperts
from source_conditioned_pinn import load_conditioned_checkpoint

BASE = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth"
SEED_CHECKPOINTS = {
    2027: ROOT / "outputs/prototype_moe_development/joint_moe_physics.pth",
}
OBSERVATION_DIR = ROOT / "data/observations/holdout_20260927"
OUTPUT = ROOT / "experiments/prototype_moe_holdout_v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    reference = observed.astype(np.float64)
    estimate = predicted.astype(np.float64)
    energy = max(float(np.sum(reference ** 2)), 1e-30)
    active = np.abs(reference) >= 0.05 * np.max(np.abs(reference))
    active_energy = max(float(np.sum(reference[active] ** 2)), 1e-30)
    return {
        "full_l2": float(np.sqrt(np.sum((estimate - reference) ** 2) / energy)),
        "active_l2": float(np.sqrt(np.sum((estimate[active] - reference[active]) ** 2) / active_energy)),
        "zero_lag_correlation": float(np.sum(estimate * reference) /
                                         max(np.linalg.norm(estimate) * np.linalg.norm(reference), 1e-30)),
    }


def predict_base(model, source, shape, receivers, times, domain, device):
    coords = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*shape, 3)),
        np.broadcast_to(times[None, :, None], (*shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    src = np.broadcast_to(source / domain, (len(coords), 4)).astype(np.float32)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(coords), 8192):
            x = torch.as_tensor(coords[start:start + 8192], device=device)
            s = torch.as_tensor(src[start:start + 8192], device=device)
            chunks.append(model(x, s).squeeze(-1).cpu().numpy())
    return np.concatenate(chunks).reshape(shape)


def predict_moe(model, source, shape, receivers, times, domain, device):
    coords = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*shape, 3)),
        np.broadcast_to(times[None, :, None], (*shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    src = np.broadcast_to(source / domain, (len(coords), 4)).astype(np.float32)
    outputs, gates = [], []
    with torch.inference_mode():
        for start in range(0, len(coords), 8192):
            x = torch.as_tensor(coords[start:start + 8192], device=device)
            s = torch.as_tensor(src[start:start + 8192], device=device)
            y, gate = model(x, s, freeze_base=True)
            outputs.append(y.squeeze(-1).cpu().numpy())
            gates.append(gate.cpu().numpy())
    return np.concatenate(outputs).reshape(shape), np.concatenate(gates, axis=0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, payload = load_conditioned_checkpoint(BASE, device)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    domain = np.asarray([payload["config"][key] for key in
                         ("x_length_m", "y_length_m", "z_length_m", "time_length_s")], dtype=np.float32)
    models = {}
    for seed, checkpoint in SEED_CHECKPOINTS.items():
        model = PrototypeGatedResidualMixtureOfExperts(base, expert_count=3, hidden_dim=128,
                                                       hidden_layers=3, feature_scale=1e6).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        model.eval()
        models[seed] = model

    events = []
    for path in sorted(OBSERVATION_DIR.glob("event_*.npz")):
        with np.load(path) as archive:
            observed = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            source = np.asarray(archive["source_truth"], dtype=np.float32)
        base_prediction = predict_base(base, source, observed.shape, receivers, times, domain, device)
        event = {"event": path.name, "sha256": sha256(path),
                 "source_m_s": source.tolist(), "base": metrics(observed, base_prediction),
                 "seeds": {}}
        for seed, model in models.items():
            prediction, gate = predict_moe(model, source, observed.shape, receivers, times, domain, device)
            event["seeds"][str(seed)] = {
                "metrics": metrics(observed, prediction),
                "mean_gate": gate.mean(axis=0).tolist(),
            }
        events.append(event)

    base_mean = float(np.mean([row["base"]["full_l2"] for row in events]))
    summary = {"event_count": len(events), "base_mean_full_l2": base_mean, "seeds": {}}
    for seed in SEED_CHECKPOINTS:
        values = [row["seeds"][str(seed)]["metrics"]["full_l2"] for row in events]
        gate = np.mean([row["seeds"][str(seed)]["mean_gate"] for row in events], axis=0)
        mean = float(np.mean(values))
        summary["seeds"][str(seed)] = {
            "mean_full_l2": mean,
            "std_full_l2_across_events": float(np.std(values, ddof=1)),
            "relative_improvement_over_base": float((base_mean - mean) / base_mean),
            "improves_count": int(sum(value < base_value for value, base_value in
                                       zip(values, [row["base"]["full_l2"] for row in events]))),
            "mean_gate": gate.tolist(),
        }
    seed_means = [summary["seeds"][str(seed)]["mean_full_l2"] for seed in SEED_CHECKPOINTS]
    summary["seed_mean_full_l2"] = float(np.mean(seed_means))
    summary["seed_std_full_l2"] = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
    result = {
        "role": "independent eight-event holdout forward evaluation of seeded joint MoE",
        "base_checkpoint_sha256": sha256(BASE),
        "observation_directory": str(OBSERVATION_DIR),
        "summary": summary,
        "events": events,
        "limits": "Forward-only evaluation at the known source parameters; it does not establish source localization or calibrated uncertainty.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
