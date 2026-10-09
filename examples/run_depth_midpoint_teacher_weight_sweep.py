"""Sweep midpoint FD pseudo-label weights with paired receiver batches."""

from __future__ import annotations

import argparse
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
from examples.run_depth_midpoint_teacher_finetune import (
    build_pseudo_events, pair_specs, pseudo_batch,
)
from examples.run_fmm_seeded_waveform_development import hash_file
from receiver_training import ReceiverTrainingData
from source_conditioned_pinn import SourceConditionedConfig, load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/depth_midpoint_teacher_weight_sweep_protocol.json"


def group_mean(rows: list[dict], offsets: set[int], metric: str,
               offset_by_file: dict[str, int]) -> float:
    selected = [row[metric] for row in rows
                if abs(offset_by_file[Path(row["observation"]).name]) in offsets]
    return float(np.mean(selected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    spec = json.loads(protocol_path.read_text(encoding="utf-8"))
    checkpoint = ROOT / spec["base_checkpoint"]
    depth_path = ROOT / spec["depth_results"]
    output_dir = ROOT / spec["output_directory"]
    result_path = ROOT / spec["result_file"]
    if output_dir.exists() or result_path.exists():
        raise FileExistsError("Weight sweep output exists; refusing to overwrite")
    if (hash_file(checkpoint) != spec["base_checkpoint_sha256"]
            or hash_file(depth_path) != spec["depth_results_sha256"]):
        raise ValueError("Registered input changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, metadata = load_conditioned_checkpoint(checkpoint, device)
    records = metadata["training_record"]["receiver_observations"]
    if len(records) != 33 or metadata["frequency_hz"] != 5 or metadata["epoch"] != 1000:
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
            low, high = (x, y, low_z, origin), (x, y, high_z, origin)
            if low not in source_map or high not in source_map:
                raise ValueError(f"Missing endpoints: {low}, {high}")
            pairs.append({"anchor": anchor, "low_path": source_map[low],
                          "high_path": source_map[high]})
    labels = ["control", *[f"teacher_{weight:g}" for weight in spec["teacher_loss_weights"]]]
    weights = dict(zip(labels, [0.0, *spec["teacher_loss_weights"]]))
    config = SourceConditionedConfig()
    datasets = {label: ReceiverTrainingData(paths, device, config, 5., seed=spec["seed"])
                for label in labels}
    teacher_events = build_pseudo_events(datasets[labels[1]], pairs)
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec["seed"])
    models = {label: copy.deepcopy(base).to(device) for label in labels}
    del base
    optimizers = {label: torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
                  for label, model in models.items()}
    pseudo_numpy = np.random.default_rng(spec["seed"])
    pseudo_torch = torch.Generator(device=device).manual_seed(spec["seed"])
    history = {label: [] for label in labels}
    started = time.perf_counter()
    for update in range(1, spec["updates"] + 1):
        batches = {label: datasets[label].batch(spec["receiver_batch_size"],
                                                spec["receiver_events_per_batch"])
                   for label in labels}
        reference = batches["control"]
        if any(any(not torch.equal(a, b) for a, b in zip(reference, batches[label]))
               for label in labels[1:]):
            raise ValueError("Paired receiver batches diverged")
        teacher = pseudo_batch(teacher_events, spec["teacher_batch_size"],
                               pseudo_numpy, pseudo_torch)
        for label in labels:
            model, optimizer = models[label], optimizers[label]
            optimizer.zero_grad()
            points, target, source, rms = batches[label]
            receiver_loss = ((model(points, source).squeeze(-1) / rms - target) ** 2).mean()
            if weights[label] > 0:
                tp, ty, ts, tr = teacher
                teacher_loss = ((model(tp, ts).squeeze(-1) / tr - ty) ** 2).mean()
            else:
                teacher_loss = torch.zeros((), device=device)
            total = spec["receiver_loss_weight"] * receiver_loss + weights[label] * teacher_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            history[label].append({"receiver": float(receiver_loss.detach()),
                                   "teacher": float(teacher_loss.detach()),
                                   "total": float(total.detach())})
        if update == 1 or update % 100 == 0:
            values = ", ".join(f"{label}={history[label][-1]['receiver']:.5f}"
                               for label in labels)
            print(f"update {update}/{spec['updates']}: {values}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    depth = json.loads(depth_path.read_text(encoding="utf-8"))
    offset_by_file = {row["file"]: row["offset_m"] for row in depth["events"]}
    paths_eval = [ROOT / "data/observations/eikonal_depth_sweep" / row["file"]
                  for row in depth["events"]]
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    reports = {}
    for label, model in models.items():
        rows = [evaluate(model, scale, path, hash_file(path)) for path in paths_eval]
        reports[label] = {
            "teacher_loss_weight": weights[label],
            "midpoint_80_full_l2": group_mean(rows, {80}, "full_record_relative_l2", offset_by_file),
            "endpoint_160_full_l2": group_mean(rows, {160}, "full_record_relative_l2", offset_by_file),
            "extrapolation_280_full_l2": group_mean(rows, {280}, "full_record_relative_l2", offset_by_file),
            "nonendpoint_full_l2": group_mean(rows, {80, 280}, "full_record_relative_l2", offset_by_file),
            "nonendpoint_active_l2": group_mean(rows, {80, 280}, "active_sample_relative_l2", offset_by_file),
            "events": rows,
        }
    control = reports["control"]
    passing = []
    for label in labels[1:]:
        row = reports[label]
        row["midpoint_improves"] = row["midpoint_80_full_l2"] < control["midpoint_80_full_l2"]
        row["endpoint_within_limit"] = row["endpoint_160_full_l2"] <= (
            control["endpoint_160_full_l2"]
            * (1 + spec["endpoint_degradation_limit_fraction"]))
        row["passes_selection_rule"] = row["midpoint_improves"] and row["endpoint_within_limit"]
        if row["passes_selection_rule"]:
            passing.append(label)
    selected = min(passing, key=lambda label: reports[label]["nonendpoint_full_l2"]) if passing else None
    output_dir.mkdir(parents=True, exist_ok=False)
    for label, model in models.items():
        torch.save({"model_state_dict": {key: value.detach().cpu()
                                         for key, value in model.state_dict().items()},
                    "label": label, "teacher_loss_weight": weights[label],
                    "updates": spec["updates"], "protocol_sha256": hash_file(protocol_path)},
                   output_dir / f"{label}.pth")
    result = {"role": spec["role"], "protocol_sha256": hash_file(protocol_path),
              "elapsed_seconds": elapsed, "selection_rule": spec["selection_rule"],
              "selected_label": selected, "reports": reports,
              "history_tail": {label: values[-1] for label, values in history.items()},
              "limits": spec["limits"]}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"selected={selected}")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
