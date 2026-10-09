"""Jointly train a differentiable gated MoE with PDE and receiver losses."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixture_of_experts import PrototypeGatedResidualMixtureOfExperts
from receiver_training import ReceiverTrainingData
from source_conditioned_pinn import (
    SourceConditionedConfig,
    load_conditioned_checkpoint,
    sample_source_condition,
    source_term,
)

PROTOCOL = ROOT / "experiments/prototype_moe_seed2028_protocol.json"


class GridVelocity:
    def __init__(self, path: Path, device: torch.device):
        array = np.load(path, mmap_mode="r")
        self.shape = array.shape
        self.values = torch.as_tensor(np.array(array, dtype=np.float32, copy=True),
                                      device=device)

    def __call__(self, xyz: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor([self.shape[0] - 1, self.shape[1] - 1,
                              self.shape[2] - 1], device=xyz.device)
        index = (xyz.clamp(0.0, 1.0) * scale).long()
        return self.values[index[:, 0], index[:, 1], index[:, 2]].unsqueeze(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physics_loss(model, space_time, source, velocity, frequency, cfg):
    space_time = space_time.clone().detach().requires_grad_(True)
    predicted, _ = model(space_time, source, freeze_base=False)
    grads = torch.autograd.grad(predicted, space_time, torch.ones_like(predicted),
                                create_graph=True)[0]
    second = []
    for axis in range(4):
        first = grads[:, axis:axis + 1]
        second.append(torch.autograd.grad(first, space_time,
                                          torch.ones_like(first),
                                          create_graph=True, retain_graph=True)[0][:, axis:axis + 1])
    u_xx, u_yy, u_zz, u_tt = second
    laplacian = (u_xx / cfg.x_length_m ** 2 + u_yy / cfg.y_length_m ** 2
                 + u_zz / cfg.z_length_m ** 2)
    residual = u_tt / cfg.time_length_s ** 2 - velocity.square() * laplacian
    return (residual - source_term(space_time, source, frequency, cfg)).square().mean()


def initial_loss(model, xyz, source, frequency, cfg, output_scale):
    time = torch.zeros(xyz.shape[0], 1, device=xyz.device)
    space_time = torch.cat((xyz, time), dim=-1).requires_grad_(True)
    predicted, _ = model(space_time, source, freeze_base=False)
    normalized = predicted / output_scale
    velocity = torch.autograd.grad(normalized, space_time, torch.ones_like(normalized),
                                   create_graph=True)[0][:, 3:4]
    normalized_velocity = velocity / (2.0 * np.pi * frequency * cfg.time_length_s)
    return normalized.square().mean() + normalized_velocity.square().mean()


def boundary_loss(model, count, device, cfg, output_scale):
    losses = []
    for axis in range(3):
        for value in (0.0, 1.0):
            space_time = torch.rand(count, 4, device=device)
            space_time[:, axis] = value
            source = sample_source_condition(count, device, cfg)
            predicted, _ = model(space_time, source, freeze_base=False)
            losses.append((predicted / output_scale).square().mean())
    return torch.stack(losses).mean()


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


def metrics(observed, predicted):
    reference = observed.astype(np.float64)
    estimate = predicted.astype(np.float64)
    energy = float(np.sum(reference ** 2))
    active = np.abs(reference) >= 0.05 * np.max(np.abs(reference))
    active_energy = float(np.sum(reference[active] ** 2))
    return {
        "full_l2": float(np.sqrt(np.sum((estimate - reference) ** 2) / energy)),
        "active_l2": float(np.sqrt(np.sum((estimate[active] - reference[active]) ** 2)
                                   / max(active_energy, 1e-30))),
        "zero_lag_correlation": float(np.sum(estimate * reference)
                                       / max(np.linalg.norm(estimate) * np.linalg.norm(reference), 1e-30)),
    }


def main():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path = ROOT / spec["result_file"]
    output_dir = ROOT / spec["output_directory"]
    checkpoint_path = output_dir / "joint_moe_physics.pth"
    if result_path.exists() or checkpoint_path.exists():
        raise FileExistsError("Refusing to overwrite existing physics-MoE outputs")
    base_path = ROOT / spec["base_checkpoint"]
    if sha256(base_path) != spec["base_checkpoint_sha256"]:
        raise ValueError("Registered base checkpoint changed")
    seed = int(spec["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, payload = load_conditioned_checkpoint(base_path, device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    cfg = SourceConditionedConfig()
    domain = np.asarray([payload["config"][key] for key in
                         ("x_length_m", "y_length_m", "z_length_m", "time_length_s")], dtype=np.float32)
    manifest = payload.get("training_record", {}).get("receiver_observations", [])
    observation_paths = [Path(row["path"]) for row in manifest]
    if len(observation_paths) != 33:
        raise ValueError("Expected 33 registered receiver observations")
    for row, path in zip(manifest, observation_paths):
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Training record changed: {path}")

    receiver_data = ReceiverTrainingData(observation_paths, device, cfg, 5.0, seed=seed)
    velocity = GridVelocity(ROOT / "data/velocity/vp_model.npy", device)
    model = PrototypeGatedResidualMixtureOfExperts(
        base_model, expert_count=int(spec["expert_count"]),
        hidden_dim=int(spec["expert_hidden_dim"]),
        hidden_layers=int(spec["expert_hidden_layers"]),
        feature_scale=float(spec["feature_scale"]),
    ).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=float(spec["learning_rate"]))
    history = []
    started = time.perf_counter()
    for update in range(1, int(spec["training_updates"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        n_phys = int(spec["physics_points"])
        space_time = torch.rand(n_phys, 4, device=device)
        source_phys = sample_source_condition(n_phys, device, cfg)
        loss_phys = physics_loss(model, space_time, source_phys,
                                 velocity(space_time[:, :3]), 5.0, cfg)
        source_ic = sample_source_condition(int(spec["initial_points"]), device, cfg)
        loss_ic = initial_loss(model, torch.rand(int(spec["initial_points"]), 3, device=device),
                               source_ic, 5.0, cfg, base_model.output_scale)
        loss_bc = boundary_loss(model, int(spec["boundary_points_per_face"]), device,
                                cfg, base_model.output_scale)
        receiver_batch = receiver_data.batch(int(spec["receiver_batch_size"]),
                                             int(spec["receiver_events_per_batch"]))
        space_r, target_r, source_r, rms = receiver_batch
        predicted_r, gate_r = model(space_r, source_r, freeze_base=True)
        loss_receiver = ((predicted_r.squeeze(-1) / rms - target_r) ** 2).mean()
        gate_mean = gate_r.mean(dim=0)
        gate_balance = ((gate_mean - 1.0 / model.expert_count) ** 2).mean()
        total = (loss_phys + cfg.lambda_ic * loss_ic + cfg.lambda_bc * loss_bc
                 + float(spec["receiver_loss_weight"]) * loss_receiver
                 + float(spec["gate_balance_weight"]) * gate_balance)
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if update == 1 or update % 100 == 0:
            row = {"update": update, "total": float(total.detach()),
                   "physics": float(loss_phys.detach()), "initial": float(loss_ic.detach()),
                   "boundary": float(loss_bc.detach()), "receivers": float(loss_receiver.detach()),
                   "gate_balance": float(gate_balance.detach()),
                   "mean_gate": gate_mean.detach().cpu().tolist()}
            history.append(row)
            print(row, flush=True)
    training_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "spec": spec,
                "domain": domain.tolist(), "training_history": history}, checkpoint_path)

    targets = []
    for item in spec["historical_results"]:
        historical = json.loads((ROOT / item["path"]).read_text(encoding="utf-8"))
        for event in historical["events"]:
            path = Path(event.get("observation", ""))
            if not path.is_absolute() or not path.exists():
                path = ROOT / item["data_directory"] / event["event"]
            with np.load(path) as archive:
                targets.append({"event": event["event"], "path": path,
                                "data": np.asarray(archive["data"], dtype=np.float32),
                                "times": np.asarray(archive["times"], dtype=np.float32),
                                "receivers": np.asarray(archive["receiver_coords"], dtype=np.float32),
                                "source": np.asarray(archive["source_truth"], dtype=np.float32)})
    if len(targets) != 16:
        raise ValueError(f"Expected 16 historical targets, got {len(targets)}")
    events = []
    for target in targets:
        prediction, gate = make_prediction(model, target["source"], target["data"].shape,
                                            target["receivers"], target["times"], domain, device)
        base_prediction, _ = make_prediction(model, target["source"], target["data"].shape,
                                             target["receivers"], target["times"], domain, device)
        # Remove the trained correction for the paired frozen-base comparison.
        base_outputs = []
        coords = np.concatenate((
            np.broadcast_to(target["receivers"][:, None, :], (*target["data"].shape, 3)),
            np.broadcast_to(target["times"][None, :, None], (*target["data"].shape, 1)),
        ), axis=-1).reshape(-1, 4) / domain
        src = np.broadcast_to(target["source"] / domain, (len(coords), 4)).astype(np.float32)
        with torch.inference_mode():
            for start in range(0, len(coords), 8192):
                x = torch.as_tensor(coords[start:start + 8192], device=device)
                s = torch.as_tensor(src[start:start + 8192], device=device)
                base_outputs.append(base_model(x, s).squeeze(-1).cpu().numpy())
        base_prediction = np.concatenate(base_outputs).reshape(target["data"].shape)
        events.append({"event": target["event"], "observation": str(target["path"]),
                       "sha256": sha256(target["path"]), "source_m_s": target["source"].tolist(),
                       "mean_gate": gate.mean(axis=0).tolist(),
                       "base": metrics(target["data"], base_prediction),
                       "moe": metrics(target["data"], prediction)})
        events[-1]["moe_improves"] = events[-1]["moe"]["full_l2"] < events[-1]["base"]["full_l2"]

    def mean(branch, metric):
        return float(np.mean([row[branch][metric] for row in events]))
    base_mean = mean("base", "full_l2")
    moe_mean = mean("moe", "full_l2")
    gate_usage = np.mean([row["mean_gate"] for row in events], axis=0)
    improvement = (base_mean - moe_mean) / base_mean
    improved_count = int(sum(row["moe_improves"] for row in events))
    result = {"role": spec["role"], "protocol_sha256": sha256(PROTOCOL),
              "base_checkpoint_sha256": spec["base_checkpoint_sha256"],
              "moe_checkpoint": str(checkpoint_path), "moe_checkpoint_sha256": sha256(checkpoint_path),
              "summary": {"event_count": len(events), "training_seconds": training_seconds,
                          "base_mean_full_l2": base_mean, "moe_mean_full_l2": moe_mean,
                          "base_mean_active_l2": mean("base", "active_l2"),
                          "moe_mean_active_l2": mean("moe", "active_l2"),
                          "relative_improvement_over_base": improvement,
                          "moe_improves_count": improved_count,
                          "mean_gate_usage": gate_usage.tolist(),
                          "gate_collapsed": bool(np.max(gate_usage) > 0.90),
                          "decision_gate_passed": bool(improvement >= 0.05 and improved_count >= 9 and np.max(gate_usage) <= 0.90)},
              "training_history": history, "events": events,
              "decision_rule": spec["decision_rule"], "limits": spec["limits"]}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
