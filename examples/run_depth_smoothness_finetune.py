"""Fine-tune a depth-covered source-conditioned model with midpoint smoothness."""

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

PROTOCOL = ROOT / "experiments/depth_smoothness_finetune_protocol.json"


def check_inputs(protocol: dict):
    checkpoint = ROOT / protocol["base_checkpoint"]
    if hash_file(checkpoint) != protocol["base_checkpoint_sha256"]:
        raise ValueError("Base checkpoint changed")
    depth_path = ROOT / protocol["depth_results"]
    if hash_file(depth_path) != protocol["depth_results_sha256"]:
        raise ValueError("Depth results changed")
    depth = json.loads(depth_path.read_text())
    if len(depth["events"]) != 12 or depth["protocol_sha256"] != hash_file(ROOT / protocol["depth_protocol"]):
        raise ValueError("Depth result provenance changed")
    output = ROOT / protocol["output_directory"]
    result = ROOT / protocol["result_file"]
    if output.exists() or result.exists():
        raise FileExistsError("Smoothness experiment already exists; refusing to overwrite")
    _, metadata = load_conditioned_checkpoint(checkpoint, torch.device("cpu"))
    records = metadata["training_record"]["receiver_observations"]
    if (len(records) != protocol["training_sources"] or metadata["frequency_hz"] != 5
            or metadata["epoch"] != 1000):
        raise ValueError("Base checkpoint is not the registered 33-source 5 Hz model")
    paths = [Path(item["path"]) for item in records]
    for item, path in zip(records, paths):
        if hash_file(path) != item["sha256"]:
            raise ValueError(f"Training record changed: {path}")
    source_map = {tuple(np.round(item["source_truth"], 4)): (path, item)
                  for path, item in zip(paths, records)}
    pairs = []
    pair_specs = (
        ("a10", (10000., 6500., .25), ((840., 1000.), (1000., 1160.))),
        ("a14", (11000., 10000., .34), ((1040., 1360.),)),
    )
    for anchor, (x, y, t), depths in pair_specs:
        for low_z, high_z in depths:
            low_key = (x, y, low_z, t)
            high_key = (x, y, high_z, t)
            if low_key not in source_map or high_key not in source_map:
                raise ValueError(f"Missing registered pair for {anchor}: {low_key}, {high_key}")
            pairs.append({"anchor": anchor, "low": source_map[low_key],
                          "high": source_map[high_key], "low_z": low_z,
                          "high_z": high_z})
    return checkpoint, paths, records, pairs, depth


def smooth_batch(data: ReceiverTrainingData, pairs: list[dict], event_index: dict[Path, int],
                 rng: np.random.Generator, torch_rng: torch.Generator, count: int):
    pair = pairs[int(rng.integers(len(pairs)))]
    low_path = Path(pair["low"][0]).resolve()
    high_path = Path(pair["high"][0]).resolve()
    try:
        low_index = event_index[low_path]
        high_index = event_index[high_path]
    except KeyError as exc:
        raise ValueError(f"Smoothness pair is not present in ReceiverTrainingData: {exc}") from exc
    low_x, _, low_source, _, low_rms = data.events[low_index]
    # Use the endpoint event's receiver-time coordinates for both source conditions.
    high_x, _, high_source, _, high_rms = data.events[high_index]
    if not torch.equal(low_x, high_x):
        raise ValueError("Paired records do not share receiver-time coordinates")
    ids = torch.randint(len(low_x), (count,), device=low_x.device, generator=torch_rng)
    points = low_x[ids]
    low = low_source.expand(count, 4)
    high = high_source.expand(count, 4)
    midpoint = .5 * (low_source + high_source).expand(count, 4)
    scale = torch.full((count,), .5 * (low_rms + high_rms), device=low_x.device)
    return points, low, high, midpoint, scale


def smooth_loss(model, sample):
    points, low, high, midpoint, scale = sample
    pred_low = model(points, low).squeeze(-1) / scale
    pred_high = model(points, high).squeeze(-1) / scale
    pred_mid = model(points, midpoint).squeeze(-1) / scale
    return ((pred_mid - .5 * (pred_low + pred_high)) ** 2).mean()


def main():
    protocol = json.loads(PROTOCOL.read_text())
    checkpoint, paths, records, pairs, depth = check_inputs(protocol)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, metadata = load_conditioned_checkpoint(checkpoint, device)
    config = SourceConditionedConfig()
    data_control = ReceiverTrainingData(paths, device, config, 5., seed=42)
    data_smooth = ReceiverTrainingData(paths, device, config, 5., seed=42)
    event_index = {
        Path(item["path"]).resolve(): index
        for index, item in enumerate(data_smooth.metadata)
    }
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol["seed"])
    control = copy.deepcopy(base).to(device)
    candidate = copy.deepcopy(base).to(device)
    del base
    opt_control = torch.optim.Adam(control.parameters(), lr=protocol["learning_rate"])
    opt_candidate = torch.optim.Adam(candidate.parameters(), lr=protocol["learning_rate"])
    np_control = np.random.default_rng(protocol["seed"])
    np_candidate = np.random.default_rng(protocol["seed"])
    torch_control = torch.Generator(device=device).manual_seed(protocol["seed"])
    torch_candidate = torch.Generator(device=device).manual_seed(protocol["seed"])
    history = {"control": [], "candidate": []}
    started = time.perf_counter()
    for epoch in range(1, protocol["updates"] + 1):
        batch_control = data_control.batch(protocol["receiver_batch_size"],
                                           protocol["receiver_events_per_batch"])
        batch_candidate = data_smooth.batch(protocol["receiver_batch_size"],
                                            protocol["receiver_events_per_batch"])
        if any(not torch.equal(a, b) for a, b in zip(batch_control, batch_candidate)):
            raise ValueError("Paired receiver batches diverged")
        smooth_sample = smooth_batch(data_smooth, pairs, event_index, np_candidate,
                                     torch_candidate, protocol["smooth_samples"])
        for name, model, optimizer in (("control", control, opt_control),
                                       ("candidate", candidate, opt_candidate)):
            model.train()
            optimizer.zero_grad()
            points, target, source, rms = batch_control
            receiver = ((model(points, source).squeeze(-1) / rms - target) ** 2).mean()
            regularizer = smooth_loss(model, smooth_sample) if name == "candidate" else torch.zeros((), device=device)
            total = (protocol["receiver_loss_weight"] * receiver
                     + protocol["smooth_loss_weight"] * regularizer)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            history[name].append({"total": float(total.detach()),
                                  "receiver": float(receiver.detach()),
                                  "smooth": float(regularizer.detach())})
        if epoch == 1 or epoch % 100 == 0:
            print(f"epoch {epoch}/{protocol['updates']}: "
                  f"control={history['control'][-1]['receiver']:.5f}, "
                  f"candidate={history['candidate'][-1]['receiver']:.5f}, "
                  f"smooth={history['candidate'][-1]['smooth']:.5f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")], dtype=np.float32)
    depth_events = [ROOT / "data/observations/eikonal_depth_sweep" / row["file"]
                    for row in depth["events"]]
    reports = {}
    for name, model in (("control", control), ("candidate", candidate)):
        rows = [evaluate(model, scale, path, hash_file(path)) for path in depth_events]
        reports[name] = {"mean_full_l2": float(np.mean([r["full_record_relative_l2"] for r in rows])),
                         "mean_active_l2": float(np.mean([r["active_sample_relative_l2"] for r in rows])),
                         "events": rows}
    output_dir = ROOT / protocol["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, model in (("control", control), ("candidate", candidate)):
        torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "input": name, "epoch": protocol["updates"],
                    "base_checkpoint_sha256": protocol["base_checkpoint_sha256"],
                    "protocol_sha256": hash_file(PROTOCOL)},
                   output_dir / f"{name}.pth")
    report = {"role": protocol["role"], "protocol_sha256": hash_file(PROTOCOL),
              "base_checkpoint_sha256": protocol["base_checkpoint_sha256"],
              "training_records_sha256": [item["sha256"] for item in records],
              "pairs": [{"anchor": p["anchor"], "low_z": p["low_z"], "high_z": p["high_z"],
                         "low_sha256": p["low"][1]["sha256"], "high_sha256": p["high"][1]["sha256"]}
                        for p in pairs],
              "elapsed_seconds": elapsed, "history_tail": {k: v[-1] for k, v in history.items()},
              "models": reports, "model_paths": {k: str((output_dir / f"{k}.pth").resolve())
                                                   for k in ("control", "candidate")},
              "limits": protocol["limits"]}
    result = ROOT / protocol["result_file"]
    with result.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
