"""Matched-budget ablation for explicit source-receiver geometric features."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.invert_fmm_receiver_fields import interpolate_fields
from examples.run_source_transfer_geometry_feature import PROTOCOL as PILOT_PROTOCOL
from examples.run_source_transfer_geometry_feature import load_observations, relative_l2, sample_training_batch
from examples.run_source_transfer_moe import digest, read_setup as read_parent_setup
from examples.run_source_transfer_geometry_feature import aligned_prediction
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator, TravelTimeAlignedPrototypeMoE

PROTOCOL = ROOT / "experiments/source_transfer_geometry_ablation_protocol.json"


def new_model(fields, receivers, setup, device, geometric):
    m = setup["model"]
    model = TravelTimeAlignedPrototypeMoE(
        ReceiverTravelTimeInterpolator(torch.from_numpy(np.array(fields, copy=True)), 80.0),
        torch.from_numpy(receivers), tuple(m["spatial_lengths_m"]),
        time_length_s=m["time_length_s"], frequency_hz=m["frequency_hz"],
        expert_count=m["expert_count"], hidden_dim=m["hidden_dim"],
        hidden_layers=m["hidden_layers"],
        prototype_centers=torch.tensor(m["prototype_centers"]),
        prototype_scales=tuple(m["prototype_scales"]),
        temperature=m["prototype_temperature"],
        geometric_attenuation=geometric)
    return model.to(device)


def phase(model, events, receivers, times, fields, setup, rng, updates, device):
    training = setup["training"]
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    arrivals = interpolate_fields(fields, np.stack([e["source"][:3] for e in events]), 80.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    for step in range(1, updates + 1):
        model.train()
        points, sources, receiver_ids, target = sample_training_batch(
            events, receivers, times, arrivals, setup, rng, domain, device)
        optimizer.zero_grad(set_to_none=True)
        prediction, gates, _ = model(points, sources, receiver_ids)
        balance = (gates.mean(0) - 1.0 / model.expert_count).square().mean()
        loss = ((prediction - target).square().mean()
                + training["gate_balance_weight"] * balance
                + training["prototype_separation_weight"] * model.prototype_separation_loss())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
        optimizer.step()
        with torch.no_grad():
            model.prototype_centers.clamp_(*setup["model"]["prototype_bounds_normalized"])
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=None,
                        help="Override both source and target update counts")
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--report-file", default=None)
    args = parser.parse_args()
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent, setup, _, target_paths = read_parent_setup()
    parent_spec = json.loads((ROOT / spec["parent_protocol"]).read_text(encoding="utf-8"))
    model_spec = json.loads((ROOT / spec["model_protocol"]).read_text(encoding="utf-8"))
    if spec["seeds"] != [4101, 4102, 4103]:
        raise ValueError("Unexpected seeds")
    updates = int(args.updates) if args.updates is not None else int(spec["source_updates"])
    if updates < 1:
        raise ValueError("updates must be positive")
    output_dir = ROOT / (args.output_directory or spec["output_directory"])
    result_path = ROOT / (args.result_json or spec["result_json"])
    report_path = ROOT / (args.report_file or spec["report_file"])
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite geometry ablation results")
    source_paths = [ROOT / parent_spec["source_observation_directory"] /
                    f"source_{i:02d}.npz" for i in parent_spec["source_training_indices"]]
    source_events, receivers, times = load_observations(source_paths)
    target_events, target_receivers, target_times = load_observations(target_paths)
    if not np.array_equal(receivers, target_receivers) or not np.array_equal(times, target_times):
        raise ValueError("Source and target acquisition differ")
    development_paths = [ROOT / directory / name
                         for directory in model_spec["development_sets"]
                         for name in sorted(p.name for p in (ROOT / directory).glob("*.npz"))]
    development, dev_receivers, dev_times = load_observations(development_paths)
    if not np.array_equal(receivers, dev_receivers) or not np.array_equal(times, dev_times):
        raise ValueError("Development acquisition differs")
    source_fields = np.load(ROOT / parent_spec["source_fmm_fields"], mmap_mode="r")
    target_fields = np.load(ROOT / parent_spec["target_fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, checkpoints = [], []
    arm_specs = [("source_to_target_no_geometry", "source_to_target", False),
                 ("source_to_target_geometry", "source_to_target", True),
                 ("target_to_target_no_geometry", "target_to_target", False),
                 ("target_to_target_geometry", "target_to_target", True)]
    for seed in spec["seeds"]:
        for name, base_arm, geometric in arm_specs:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            first_events = source_events if base_arm == "source_to_target" else [
                target_events[i] for i in parent_spec["source_training_indices"]]
            first_fields = source_fields if base_arm == "source_to_target" else target_fields
            model = new_model(first_fields, receivers, setup, device, geometric)
            source_loss = phase(model, first_events, receivers, times, first_fields,
                                setup, rng, updates, device)
            if base_arm == "source_to_target":
                weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                model = new_model(target_fields, receivers, setup, device, geometric)
                model.load_state_dict(weights)
            target_loss = phase(model, target_events, receivers, times, target_fields,
                                setup, rng, updates, device)
            path = output_dir / f"{name}_seed{seed}.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "protocol_sha256": digest(PROTOCOL), "arm": name, "seed": seed,
                        "source_loss": source_loss, "target_loss": target_loss}, path)
            checkpoints.append({"arm": name, "seed": seed,
                                "path": str(path.relative_to(ROOT)), "sha256": digest(path)})
            for event in development:
                prediction = aligned_prediction(model, event, receivers, times,
                                                np.asarray([*setup["model"]["spatial_lengths_m"],
                                                            setup["model"]["time_length_s"]], dtype=np.float32),
                                                device)
                rows.append({"arm": name, "base_arm": base_arm, "geometric_attenuation": geometric,
                             "seed": seed, "event": str(event["path"].relative_to(ROOT)),
                             "sha256": event["sha256"], "relative_l2": relative_l2(event["data"] / event["rms"], prediction)})
            print(f"{name} seed={seed}: source_loss={source_loss:.6f}, target_loss={target_loss:.6f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = []
    for name, _, _ in arm_specs:
        group = [r["relative_l2"] for r in rows if r["arm"] == name]
        summary.append({"arm": name, "outcomes": len(group),
                        "mean_development_relative_l2": float(np.mean(group)),
                        "median_development_relative_l2": float(np.median(group))})
    result = {"protocol_sha256": digest(PROTOCOL), "device": str(device),
              "checkpoints": checkpoints, "summary": summary, "rows": rows,
              "restriction": spec["restriction"]}
    lines = ["# Matched geometry-feature ablation", "",
             f"All four arms use identical events, seeds, {updates}+{updates} updates, sampling and optimizer settings.", "",
             "| Arm | Outcomes | Mean development relative L2 | Median development relative L2 |",
             "|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['arm']} | {row['outcomes']} | {row['mean_development_relative_l2']:.4f} | {row['median_development_relative_l2']:.4f} |")
    lines.extend(["", spec["restriction"]])
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
