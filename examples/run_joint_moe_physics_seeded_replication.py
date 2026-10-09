"""Replicate the physics-and-receiver MoE with two independent seeds."""

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
from receiver_training import ReceiverTrainingData
from source_conditioned_pinn import SourceConditionedConfig, load_conditioned_checkpoint, sample_source_condition, source_term

BASE = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth"
OUT_DIR = ROOT / "outputs/joint_moe_physics_replications"
RESULT = ROOT / "experiments/joint_moe_physics_seeded_replications.json"
SEEDS = [7, 123]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physics_loss(model, space_time, source, velocity, cfg):
    space_time = space_time.clone().detach().requires_grad_(True)
    predicted, _ = model(space_time, source, freeze_base=False)
    grads = torch.autograd.grad(predicted, space_time, torch.ones_like(predicted), create_graph=True)[0]
    second = []
    for axis in range(4):
        first = grads[:, axis:axis + 1]
        second.append(torch.autograd.grad(first, space_time, torch.ones_like(first), create_graph=True,
                                          retain_graph=True)[0][:, axis:axis + 1])
    lap = second[0] / cfg.x_length_m ** 2 + second[1] / cfg.y_length_m ** 2 + second[2] / cfg.z_length_m ** 2
    residual = second[3] / cfg.time_length_s ** 2 - velocity.square() * lap
    return (residual - source_term(space_time, source, 5.0, cfg)).square().mean()


class GridVelocity:
    def __init__(self, path, device):
        array = np.load(path, mmap_mode="r")
        self.shape = array.shape
        self.values = torch.as_tensor(np.array(array, dtype=np.float32, copy=True), device=device)

    def __call__(self, xyz):
        scale = torch.tensor([self.shape[0] - 1, self.shape[1] - 1, self.shape[2] - 1], device=xyz.device)
        index = (xyz.clamp(0, 1) * scale).long()
        return self.values[index[:, 0], index[:, 1], index[:, 2]].unsqueeze(1)


def train(seed, base_model, payload, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    cfg = SourceConditionedConfig()
    paths = [Path(row["path"]) for row in payload["training_record"]["receiver_observations"]]
    data = ReceiverTrainingData(paths, device, cfg, 5.0, seed=seed)
    velocity = GridVelocity(ROOT / "data/velocity/vp_model.npy", device)
    model = GatedResidualMixtureOfExperts(base_model, expert_count=3, hidden_dim=128,
        hidden_layers=3, feature_scale=1e6).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    started = time.perf_counter()
    history = []
    for update in range(1, 1001):
        model.train(); optimizer.zero_grad(set_to_none=True)
        space = torch.rand(64, 4, device=device)
        source_phys = sample_source_condition(64, device, cfg)
        velocity_values = velocity(space[:, :3])
        loss_phys = physics_loss(model, space, source_phys, velocity_values, cfg)
        space_r, target_r, source_r, rms = data.batch(1024, 8)
        prediction, gate = model(space_r, source_r, freeze_base=True)
        loss_rec = ((prediction.squeeze(-1) / rms - target_r) ** 2).mean()
        gate_mean = gate.mean(dim=0)
        loss_gate = ((gate_mean - 1.0 / model.expert_count) ** 2).mean()
        total = loss_phys + 0.1 * loss_rec + 0.01 * loss_gate
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if update in (1, 500, 1000):
            history.append({"update": update, "total": float(total.detach()),
                            "physics": float(loss_phys.detach()), "receiver": float(loss_rec.detach()),
                            "gate": gate_mean.detach().cpu().tolist()})
    output = OUT_DIR / f"seed_{seed}.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": seed, "history": history,
                "base_checkpoint_sha256": sha256(BASE)}, output)
    return output, history, time.perf_counter() - started


def predict(model, source, receivers, times, domain, device):
    coords = np.concatenate((np.broadcast_to(receivers[:, None, :], (*receivers.shape[:1], len(times), 3)),
                             np.broadcast_to(times[None, :, None], (len(receivers), len(times), 1))), axis=-1).reshape(-1, 4) / domain
    src = np.broadcast_to(source / domain, (len(coords), 4)).astype(np.float32)
    values = []
    with torch.inference_mode():
        for start in range(0, len(coords), 8192):
            x = torch.as_tensor(coords[start:start + 8192], device=device)
            s = torch.as_tensor(src[start:start + 8192], device=device)
            values.append(model(x, s, freeze_base=True)[0].squeeze(-1).cpu().numpy())
    return np.concatenate(values).reshape(len(receivers), len(times))


def evaluate(checkpoint, base_model, payload, device):
    spec = {"expert_count": 3, "hidden_dim": 128, "hidden_layers": 3, "feature_scale": 1e6}
    model = GatedResidualMixtureOfExperts(base_model, **spec).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    domain = np.asarray([payload["config"][key] for key in ("x_length_m", "y_length_m", "z_length_m", "time_length_s")], dtype=np.float32)
    records = []
    paths = []
    for result_path, directory in ((ROOT / "experiments/eikonal_independent_results.json", ROOT / "data/observations/eikonal_independent"),
                                   (ROOT / "experiments/eikonal_wide_search_independent_results.json", ROOT / "data/observations/eikonal_wide_search_independent")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for event in result["events"]:
            observation = event.get("observation")
            path = Path(observation) if observation else directory / event["event"]
            if not path.is_file(): path = directory / event["event"]
            paths.append(path)
    for path in paths:
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            source = np.asarray(archive["source_truth"], dtype=np.float32)
        prediction = predict(model, source, receivers, times, domain, device)
        energy = max(float(np.sum(data.astype(np.float64) ** 2)), 1e-30)
        records.append(float(np.sqrt(np.sum((prediction - data) ** 2) / energy)))
    return {"event_count": len(records), "mean_full_l2": float(np.mean(records)), "records": records}


def main():
    if RESULT.exists(): raise FileExistsError(f"Refusing to overwrite {RESULT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, payload = load_conditioned_checkpoint(BASE, device)
    base.eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    results = []
    for seed in SEEDS:
        checkpoint, history, seconds = train(seed, base, payload, device)
        results.append({"seed": seed, "checkpoint": str(checkpoint), "training_seconds": seconds,
                        "summary": evaluate(checkpoint, base, payload, device), "history": history})
    output = {"role": "seeded replication of physics-and-receiver joint MoE",
              "seeds": SEEDS, "base_checkpoint_sha256": sha256(BASE), "results": results,
              "limits": "The same 16 previously opened events are used only to compare seed stability; independent eight-event holdout evaluation is separate."}
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps([{"seed": row["seed"], "summary": row["summary"]} for row in results], indent=2), flush=True)


if __name__ == "__main__":
    main()
