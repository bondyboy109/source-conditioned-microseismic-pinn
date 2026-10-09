"""Train and audit a differentiable source-gated residual mixture-of-experts."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixture_of_experts import GatedResidualMixtureOfExperts
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/joint_moe_development_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_record(path: Path):
    with np.load(path) as archive:
        return (np.asarray(archive["data"], dtype=np.float32),
                np.asarray(archive["times"], dtype=np.float32),
                np.asarray(archive["receiver_coords"], dtype=np.float32),
                np.asarray(archive["source_truth"], dtype=np.float32))


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
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
                                       / max(np.linalg.norm(estimate)
                                             * np.linalg.norm(reference), 1e-30)),
    }


def make_features(data_shape: tuple[int, int], receivers: np.ndarray,
                  times: np.ndarray, source: np.ndarray,
                  domain: np.ndarray) -> np.ndarray:
    coords = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*data_shape, 3)),
        np.broadcast_to(times[None, :, None], (*data_shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    source_values = np.broadcast_to(source / domain, (len(coords), 4))
    return np.concatenate((coords, source_values), axis=1).astype(np.float32)


def predict(model: GatedResidualMixtureOfExperts, source: np.ndarray,
            data_shape: tuple[int, int], receivers: np.ndarray, times: np.ndarray,
            domain: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    base_features = make_features(data_shape, receivers, times, source, domain)
    parts, gates = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(base_features), 8192):
            x = torch.as_tensor(base_features[start:start + 8192, :4], device=device)
            s = torch.as_tensor(base_features[start:start + 8192, 4:8], device=device)
            y, g = model(x, s, freeze_base=True)
            parts.append(y.squeeze(-1).cpu().numpy())
            gates.append(g.cpu().numpy())
    return (np.concatenate(parts).reshape(data_shape).astype(np.float32),
            np.concatenate(gates, axis=0))


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path = ROOT / spec["result_file"]
    output_dir = ROOT / spec["output_directory"]
    checkpoint_path = output_dir / "joint_moe.pth"
    if result_path.exists() or checkpoint_path.exists():
        raise FileExistsError("Refusing to overwrite existing MoE outputs")
    base_path = ROOT / spec["base_checkpoint"]
    if sha256(base_path) != spec["base_checkpoint_sha256"]:
        raise ValueError("Registered base checkpoint changed")
    seed = int(spec["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, payload = load_conditioned_checkpoint(base_path, device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    domain = np.asarray([payload["config"][key] for key in
                         ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                        dtype=np.float32)
    manifest = payload.get("training_record", {}).get("receiver_observations", [])
    if len(manifest) != 33:
        raise ValueError(f"Expected 33 training records, got {len(manifest)}")

    records = []
    reference_times = reference_receivers = None
    for row in manifest:
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Training record changed: {path}")
        data, times, receivers, source = load_record(path)
        if reference_times is None:
            reference_times, reference_receivers = times, receivers
        elif not (np.array_equal(times, reference_times)
                  and np.array_equal(receivers, reference_receivers)):
            raise ValueError("Training geometry differs")
        records.append({"path": path, "data": data, "times": times,
                        "receivers": receivers, "source": source})

    # Build a shared feature/target table from all registered receiver records.
    feature_blocks, target_blocks, active_blocks, source_blocks = [], [], [], []
    with torch.inference_mode():
        for row in records:
            x_np = make_features(row["data"].shape, row["receivers"], row["times"],
                                 row["source"], domain)
            source_np = np.broadcast_to(row["source"] / domain, (len(x_np), 4)).astype(np.float32)
            base_values = []
            for start in range(0, len(x_np), 8192):
                x = torch.as_tensor(x_np[start:start + 8192, :4], device=device)
                s = torch.as_tensor(source_np[start:start + 8192], device=device)
                base_values.append(base_model(x, s).squeeze(-1).cpu().numpy())
            base_values = np.concatenate(base_values).astype(np.float32)
            target = row["data"].reshape(-1) - base_values
            feature_blocks.append(np.concatenate((x_np, base_values[:, None] * float(spec["feature_scale"])), axis=1))
            target_blocks.append((target * float(spec["feature_scale"])).astype(np.float32))
            active_blocks.append(np.abs(row["data"].reshape(-1)) >= 0.05 * np.max(np.abs(row["data"])))
            source_blocks.append(source_np)
    features = np.concatenate(feature_blocks)
    targets = np.concatenate(target_blocks)
    active = np.flatnonzero(np.concatenate(active_blocks))
    sources = np.concatenate(source_blocks)

    model = GatedResidualMixtureOfExperts(
        base_model, expert_count=int(spec["expert_count"]),
        hidden_dim=int(spec["expert_hidden_dim"]),
        hidden_layers=int(spec["expert_hidden_layers"]),
        feature_scale=float(spec["feature_scale"]),
    ).to(device)
    # The base is a shared frozen trunk; only all experts and the gate are trained jointly.
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(spec["learning_rate"]))
    rng = np.random.default_rng(seed)
    batch_size = int(spec["batch_size"])
    history = []
    started = time.perf_counter()
    for update in range(1, int(spec["training_updates"]) + 1):
        half = batch_size // 2
        indices = np.concatenate((rng.choice(active, half, replace=True),
                                  rng.integers(0, len(features), batch_size - half)))
        x = torch.as_tensor(features[indices, :4], device=device)
        source = torch.as_tensor(features[indices, 4:8], device=device)
        target = torch.as_tensor(targets[indices], device=device).unsqueeze(1)
        # The module recomputes the frozen base; the explicit base column is part of
        # the expert input and is reconstructed by the model for consistency.
        optimizer.zero_grad(set_to_none=True)
        prediction, gate = model(x, source, freeze_base=True)
        residual_scaled = (prediction - model.base_model(x, source)) * float(spec["feature_scale"])
        loss_data = torch.mean((residual_scaled - target) ** 2)
        mean_gate = gate.mean(dim=0)
        balance = torch.mean((mean_gate - 1.0 / model.expert_count) ** 2)
        loss = loss_data + float(spec["gate_balance_weight"]) * balance
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if update == 1 or update % 300 == 0:
            record = {"update": update, "loss": float(loss.detach().cpu()),
                      "data_loss": float(loss_data.detach().cpu()),
                      "gate_balance": float(balance.detach().cpu()),
                      "mean_gate": mean_gate.detach().cpu().tolist()}
            history.append(record)
            print(record, flush=True)
    training_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "spec": spec,
                "domain": domain.tolist(), "training_history": history}, checkpoint_path)

    targets_eval = []
    for item in spec["historical_results"]:
        historical = json.loads((ROOT / item["path"]).read_text(encoding="utf-8"))
        for event in historical["events"]:
            path = Path(event.get("observation", ""))
            if not path.is_absolute() or not path.exists():
                path = ROOT / item["data_directory"] / event["event"]
            data, times, receivers, source = load_record(path)
            targets_eval.append({"event": event["event"], "path": path,
                                 "data": data, "times": times,
                                 "receivers": receivers, "source": source})
    if len(targets_eval) != 16:
        raise ValueError(f"Expected 16 historical targets, got {len(targets_eval)}")

    events = []
    for target in targets_eval:
        prediction, gate = predict(model, target["source"], target["data"].shape,
                                   target["receivers"], target["times"], domain, device)
        base_prediction = neural_prediction = None
        # Re-evaluate the frozen base directly for a paired comparison.
        base_parts = []
        x_np = make_features(target["data"].shape, target["receivers"], target["times"],
                             target["source"], domain)
        source_np = np.broadcast_to(target["source"] / domain, (len(x_np), 4)).astype(np.float32)
        with torch.inference_mode():
            for start in range(0, len(x_np), 8192):
                x = torch.as_tensor(x_np[start:start + 8192, :4], device=device)
                s = torch.as_tensor(source_np[start:start + 8192], device=device)
                base_parts.append(base_model(x, s).squeeze(-1).cpu().numpy())
        base_prediction = np.concatenate(base_parts).reshape(target["data"].shape).astype(np.float32)
        event = {"event": target["event"], "observation": str(target["path"]),
                 "sha256": sha256(target["path"]),
                 "source_m_s": target["source"].tolist(),
                 "mean_gate": gate.mean(axis=0).tolist(),
                 "base": metrics(target["data"], base_prediction),
                 "moe": metrics(target["data"], prediction)}
        event["moe_improves"] = event["moe"]["full_l2"] < event["base"]["full_l2"]
        events.append(event)

    def mean(branch: str, metric: str) -> float:
        return float(np.mean([row[branch][metric] for row in events]))

    mean_base = mean("base", "full_l2")
    mean_moe = mean("moe", "full_l2")
    improvement = (mean_base - mean_moe) / mean_base
    count = int(sum(row["moe_improves"] for row in events))
    gate_usage = np.mean([row["mean_gate"] for row in events], axis=0)
    collapse = bool(np.max(gate_usage) > 0.90)
    result = {
        "role": spec["role"], "protocol_sha256": sha256(PROTOCOL),
        "base_checkpoint_sha256": spec["base_checkpoint_sha256"],
        "moe_checkpoint": str(checkpoint_path),
        "moe_checkpoint_sha256": sha256(checkpoint_path),
        "summary": {
            "event_count": len(events), "training_seconds": training_seconds,
            "base_mean_full_l2": mean_base, "moe_mean_full_l2": mean_moe,
            "base_mean_active_l2": mean("base", "active_l2"),
            "moe_mean_active_l2": mean("moe", "active_l2"),
            "relative_improvement_over_base": improvement,
            "moe_improves_count": count, "mean_gate_usage": gate_usage.tolist(),
            "gate_collapsed": collapse,
            "decision_gate_passed": bool(improvement >= 0.05 and count >= 9 and not collapse),
        },
        "training_history": history, "events": events,
        "decision_rule": spec["decision_rule"], "limits": spec["limits"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
