"""Fine-tune a source-conditioned model with FD midpoint pseudo-labels."""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_eikonal_training_generalization import evaluate
from examples.run_fmm_seeded_waveform_development import hash_file
from receiver_training import ReceiverTrainingData
from source_conditioned_pinn import SourceConditionedConfig, load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/depth_midpoint_teacher_finetune_protocol.json"


def pair_specs():
    return (
        ("a10", (10000., 6500., .25), ((840., 1000.), (1000., 1160.))),
        ("a14", (11000., 10000., .34), ((1040., 1200.), (1200., 1360.))),
    )


def check_inputs(protocol: dict):
    checkpoint = ROOT / protocol["base_checkpoint"]
    depth_path = ROOT / protocol["depth_results"]
    if (hash_file(checkpoint) != protocol["base_checkpoint_sha256"]
            or hash_file(depth_path) != protocol["depth_results_sha256"]):
        raise ValueError("Registered input changed")
    output = ROOT / protocol["output_directory"]
    result = ROOT / protocol["result_file"]
    if output.exists() or result.exists():
        raise FileExistsError("Teacher experiment already exists; refusing to overwrite")
    _, metadata = load_conditioned_checkpoint(checkpoint, torch.device("cpu"))
    records = metadata["training_record"]["receiver_observations"]
    if (len(records) != protocol["training_sources"] or metadata["frequency_hz"] != 5
            or metadata["epoch"] != 1000):
        raise ValueError("Unexpected base checkpoint")
    paths = [Path(item["path"]) for item in records]
    for item, path in zip(records, paths):
        if hash_file(path) != item["sha256"]:
            raise ValueError(f"Training record changed: {path}")
    source_map = {tuple(np.round(item["source_truth"], 4)): path
                  for path, item in zip(paths, records)}
    pairs = []
    for anchor, (x, y, origin), depths in pair_specs():
        for low_z, high_z in depths:
            low = (x, y, low_z, origin)
            high = (x, y, high_z, origin)
            if low not in source_map or high not in source_map:
                raise ValueError(f"Missing teacher endpoints: {low}, {high}")
            pairs.append({"anchor": anchor, "low_path": source_map[low],
                          "high_path": source_map[high], "low_z": low_z,
                          "high_z": high_z})
    depth = json.loads(depth_path.read_text(encoding="utf-8"))
    if len(depth["events"]) != 12:
        raise ValueError("Depth evaluation set changed")
    return checkpoint, metadata, records, paths, pairs, depth


def build_pseudo_events(data: ReceiverTrainingData, pairs: list[dict]):
    index = {Path(item["path"]).resolve(): i for i, item in enumerate(data.metadata)}
    events = []
    for pair in pairs:
        low = data.events[index[Path(pair["low_path"]).resolve()]]
        high = data.events[index[Path(pair["high_path"]).resolve()]]
        low_x, low_y, low_source, _, low_rms = low
        high_x, high_y, high_source, _, high_rms = high
        if not torch.equal(low_x, high_x):
            raise ValueError("Teacher endpoints have different receiver-time coordinates")
        raw_target = 0.5 * (low_y * low_rms + high_y * high_rms)
        pseudo_rms = torch.sqrt(torch.mean(raw_target.double().square())).float()
        normalized = raw_target / pseudo_rms
        active = torch.nonzero(torch.abs(raw_target) >= 0.05 * torch.max(torch.abs(raw_target)),
                               as_tuple=False).squeeze(1)
        midpoint = 0.5 * (low_source + high_source)
        events.append((low_x, normalized, midpoint, active, pseudo_rms))
    return events


def pseudo_batch(events, count: int, numpy_rng: np.random.Generator,
                 torch_rng: torch.Generator):
    event = events[int(numpy_rng.integers(len(events)))]
    points, target, source, active, rms = event
    half = count // 2
    active_ids = active[torch.randint(len(active), (half,), device=points.device,
                                      generator=torch_rng)]
    uniform_ids = torch.randint(len(points), (count - half,), device=points.device,
                                generator=torch_rng)
    ids = torch.cat((active_ids, uniform_ids))
    return (points[ids], target[ids], source.expand(count, 4),
            torch.full((count,), float(rms), device=points.device))


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checkpoint, metadata, records, paths, pairs, depth = check_inputs(protocol)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _ = load_conditioned_checkpoint(checkpoint, device)
    config = SourceConditionedConfig()
    control_data = ReceiverTrainingData(paths, device, config, 5., seed=protocol["seed"])
    candidate_data = ReceiverTrainingData(paths, device, config, 5., seed=protocol["seed"])
    teacher_events = build_pseudo_events(candidate_data, pairs)
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol["seed"])
    control = copy.deepcopy(base).to(device)
    candidate = copy.deepcopy(base).to(device)
    del base
    optimizers = {
        "control": torch.optim.Adam(control.parameters(), lr=protocol["learning_rate"]),
        "candidate": torch.optim.Adam(candidate.parameters(), lr=protocol["learning_rate"]),
    }
    pseudo_numpy = np.random.default_rng(protocol["seed"])
    pseudo_torch = torch.Generator(device=device).manual_seed(protocol["seed"])
    history = {"control": [], "candidate": []}
    started = time.perf_counter()
    for update in range(1, protocol["updates"] + 1):
        control_batch = control_data.batch(protocol["receiver_batch_size"],
                                           protocol["receiver_events_per_batch"])
        candidate_batch = candidate_data.batch(protocol["receiver_batch_size"],
                                                protocol["receiver_events_per_batch"])
        if any(not torch.equal(a, b) for a, b in zip(control_batch, candidate_batch)):
            raise ValueError("Paired receiver batches diverged")
        teacher_batch = pseudo_batch(teacher_events, protocol["teacher_batch_size"],
                                     pseudo_numpy, pseudo_torch)
        for name, model, batch in (("control", control, control_batch),
                                   ("candidate", candidate, candidate_batch)):
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad()
            points, target, source, rms = batch
            receiver = ((model(points, source).squeeze(-1) / rms - target) ** 2).mean()
            if name == "candidate":
                tp, ty, ts, tr = teacher_batch
                teacher = ((model(tp, ts).squeeze(-1) / tr - ty) ** 2).mean()
            else:
                teacher = torch.zeros((), device=device)
            total = (protocol["receiver_loss_weight"] * receiver
                     + protocol["teacher_loss_weight"] * teacher)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            history[name].append({"total": float(total.detach()),
                                  "receiver": float(receiver.detach()),
                                  "teacher": float(teacher.detach())})
        if update == 1 or update % 100 == 0:
            print(f"update {update}/{protocol['updates']}: "
                  f"control={history['control'][-1]['receiver']:.5f}, "
                  f"candidate={history['candidate'][-1]['receiver']:.5f}, "
                  f"teacher={history['candidate'][-1]['teacher']:.5f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    depth_events = [ROOT / "data/observations/eikonal_depth_sweep" / row["file"]
                    for row in depth["events"]]
    reports = {}
    for name, model in (("control", control), ("candidate", candidate)):
        rows = [evaluate(model, scale, path, hash_file(path)) for path in depth_events]
        reports[name] = {"mean_full_l2": float(np.mean([
            row["full_record_relative_l2"] for row in rows])),
            "mean_active_l2": float(np.mean([
                row["active_sample_relative_l2"] for row in rows])), "events": rows}
    output_dir = ROOT / protocol["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, model in (("control", control), ("candidate", candidate)):
        torch.save({"model_state_dict": {k: v.detach().cpu()
                                         for k, v in model.state_dict().items()},
                    "input": name, "updates": protocol["updates"],
                    "base_checkpoint_sha256": protocol["base_checkpoint_sha256"],
                    "protocol_sha256": hash_file(PROTOCOL)}, output_dir / f"{name}.pth")
    report = {"role": protocol["role"], "protocol_sha256": hash_file(PROTOCOL),
              "base_checkpoint_sha256": protocol["base_checkpoint_sha256"],
              "pairs": [{key: value for key, value in pair.items()
                         if key in ("anchor", "low_z", "high_z")} for pair in pairs],
              "elapsed_seconds": elapsed,
              "history_tail": {key: value[-1] for key, value in history.items()},
              "models": reports,
              "model_paths": {name: str((output_dir / f"{name}.pth").resolve())
                              for name in ("control", "candidate")},
              "limits": protocol["limits"]}
    result = ROOT / protocol["result_file"]
    with result.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
