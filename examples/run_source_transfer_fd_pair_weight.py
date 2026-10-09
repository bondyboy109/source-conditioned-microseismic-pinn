"""Paired weight sweep for observed FD differences between nearby sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.audit_source_transfer_neighbor_response import response
from examples.diagnose_source_transfer_trajectory import intermediate
from examples.run_source_transfer_fd_pair import paired_batch, setup as base_setup
from examples.run_source_transfer_moe import digest, new_model
from examples.run_traveltime_aligned_moe_development import (
    aligned_prediction, load_observations, relative_l2, sample_training_batch,
)
from examples.invert_fmm_receiver_fields import interpolate_fields

PROTOCOL = ROOT / "experiments/source_transfer_fd_pair_weight_protocol.json"


def read_setup():
    sweep = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    base, control, audit, trajectory, parent, model_spec, paths = base_setup()
    if (ROOT / sweep["base_pair_protocol"]
            != ROOT / "experiments/source_transfer_fd_pair_protocol.json"
            or sweep["seeds"] != base["seeds"] or sweep["arms"] != base["arms"]
            or not sweep["weights"] or len(set(sweep["weights"])) != len(sweep["weights"])
            or any(weight <= 0 or weight >= base["pair_loss_weight"]
                   for weight in sweep["weights"])):
        raise ValueError("Invalid or changed sweep protocol")
    return sweep, base, control, audit, trajectory, parent, model_spec, paths


def tag(weight):
    return str(weight).replace(".", "p")


def checkpoint(sweep, arm, seed, weight):
    return ROOT / sweep["checkpoint_directory"] / f"{arm}_seed{seed}_w{tag(weight)}.pth"


def train(sweep, base, audit, trajectory, parent, model_spec, paths):
    events, receivers, times = load_observations(paths)
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    arrivals = interpolate_fields(fields, np.stack([e["source"][:3] for e in events]), 80.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*model_spec["model"]["spatial_lengths_m"],
                         model_spec["model"]["time_length_s"]], dtype=np.float32)
    options = model_spec["training"]
    for weight in sweep["weights"]:
        for seed in sweep["seeds"]:
            for arm in sweep["arms"]:
                output = checkpoint(sweep, arm, seed, weight)
                if output.exists():
                    payload = torch.load(output, map_location="cpu", weights_only=False)
                    if payload["protocol_sha256"] != digest(PROTOCOL):
                        raise ValueError(f"Existing checkpoint protocol mismatch: {output}")
                    print(f"Preserving {output.name}", flush=True)
                    continue
                initial = intermediate(trajectory, arm, seed, base["initial_target_updates"])
                payload = torch.load(initial, map_location="cpu", weights_only=False)
                model = new_model(fields, receivers, model_spec, device)
                model.load_state_dict(payload["state_dict"])
                optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
                batch_rng = np.random.default_rng(seed + 50000)
                pair_rng = np.random.default_rng(seed + 70000)
                last = {}
                for step in range(1, base["additional_target_updates"] + 1):
                    model.train()
                    points, sources, receiver_ids, target = sample_training_batch(
                        events, receivers, times, arrivals, model_spec, batch_rng, domain, device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction, gates, _ = model(points, sources, receiver_ids)
                    positive = (prediction - target).square().mean()
                    balance = (gates.mean(0) - 1.0 / model.expert_count).square().mean()
                    loss = (positive + options["gate_balance_weight"] * balance
                            + options["prototype_separation_weight"]
                            * model.prototype_separation_loss())
                    pair = audit["pairs"][int(pair_rng.integers(0, len(audit["pairs"])))]
                    x, ids, src_a, src_b, delta = paired_batch(
                        events, receivers, times, pair, base["pair_samples_per_update"],
                        base["pair_sample_active_fraction"], pair_rng, domain, device)
                    delta_hat = model(x, src_b, ids)[0] - model(x, src_a, ids)[0]
                    pair_loss = (delta_hat - delta).square().mean()
                    loss = loss + weight * pair_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   options["gradient_clip_norm"])
                    optimizer.step()
                    with torch.no_grad():
                        model.prototype_centers.clamp_(
                            *model_spec["model"]["prototype_bounds_normalized"])
                    if step % 500 == 0:
                        last = {"positive_mse": float(positive.detach()),
                                "fd_pair_difference_mse": float(pair_loss.detach()),
                                "total_loss": float(loss.detach())}
                        print(f"w={weight} {arm} seed={seed} step={step}: {last}", flush=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in
                                           model.state_dict().items()},
                            "protocol_sha256": digest(PROTOCOL),
                            "base_pair_protocol_sha256": digest(ROOT / sweep["base_pair_protocol"]),
                            "initial_checkpoint_sha256": digest(initial),
                            "target_fmm_sha256": digest(ROOT / parent["target_fmm_fields"]),
                            "arm": arm, "seed": seed, "pair_loss_weight": weight,
                            "last_training_loss": last}, output)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def evaluate(sweep, base, control, audit, trajectory, parent, model_spec, paths):
    output = ROOT / sweep["result_json"]
    report = ROOT / sweep["report_file"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite weight-sweep results")
    events, receivers, times = load_observations(paths)
    dev_paths = [path for directory in trajectory["development_directories"]
                 for path in sorted((ROOT / directory).glob("*.npz"))]
    development, dev_receivers, dev_times = load_observations(dev_paths)
    if not np.array_equal(receivers, dev_receivers) or not np.array_equal(times, dev_times):
        raise ValueError("Development acquisition differs")
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*model_spec["model"]["spatial_lengths_m"],
                         model_spec["model"]["time_length_s"]], dtype=np.float32)
    sampled = [(e["data"][:, ::4] / e["rms"]).astype(np.float64).reshape(-1)
               for e in events]
    branches = [(0.0, "restarted_control")] + [(w, f"weight_{w}") for w in sweep["weights"]]
    rows = []
    for arm in sweep["arms"]:
        for seed in sweep["seeds"]:
            initial = intermediate(trajectory, arm, seed, base["initial_target_updates"])
            for weight, branch in branches:
                path = (ROOT / control["checkpoint_directory"] /
                        f"{arm}_seed{seed}_restarted_control.pth" if weight == 0
                        else checkpoint(sweep, arm, seed, weight))
                payload = torch.load(path, map_location="cpu", weights_only=False)
                expected = digest(ROOT / base["control_protocol"] if weight == 0 else PROTOCOL)
                if (payload["protocol_sha256"] != expected
                        or payload["initial_checkpoint_sha256"] != digest(initial)):
                    raise ValueError(f"Checkpoint provenance mismatch: {path}")
                model = new_model(fields, receivers, model_spec, device)
                model.load_state_dict(payload["state_dict"])
                model.eval()
                predictions = [response(model, e, receivers, times, domain, 4, device)
                               for e in events]
                pair_errors = []
                for pair in audit["pairs"]:
                    a, b = pair["a"], pair["b"]
                    actual = sampled[b] - sampled[a]
                    estimate = predictions[b] - predictions[a]
                    pair_errors.append(float(np.linalg.norm(estimate - actual)
                                             / np.linalg.norm(actual)))
                dev_errors = [relative_l2(
                    event["data"] / event["rms"],
                    aligned_prediction(model, event, receivers, times, domain, device))
                    for event in development]
                rows.append({"arm": arm, "seed": seed, "weight": weight,
                             "branch": branch,
                             "median_training_pair_delta_relative_l2": float(np.median(pair_errors)),
                             "mean_development_full_trace_relative_l2": float(np.mean(dev_errors))})
                print(f"{arm} seed={seed} w={weight}: evaluated", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    summary = []
    for arm in sweep["arms"]:
        for weight, branch in branches:
            group = [r for r in rows if r["arm"] == arm and r["weight"] == weight]
            summary.append({"arm": arm, "weight": weight, "branch": branch,
                            "median_training_pair_delta_relative_l2": float(np.median([
                                r["median_training_pair_delta_relative_l2"] for r in group])),
                            "mean_development_full_trace_relative_l2": float(np.mean([
                                r["mean_development_full_trace_relative_l2"] for r in group])),
                            "development_delta_from_control": float(np.mean([
                                r["mean_development_full_trace_relative_l2"] for r in group])
                                - np.mean([r["mean_development_full_trace_relative_l2"]
                                           for r in rows if r["arm"] == arm and r["weight"] == 0]))})
    result = {"protocol_sha256": digest(PROTOCOL), "device": str(device),
              "pair_count": len(audit["pairs"]), "development_event_count": len(development),
              "summary": summary, "per_seed": rows, "restriction": sweep["restriction"]}
    lines = ["# FD source-difference weight sweep", "",
             "All branches restart from the same 500-update checkpoints and replay the "
             "same 1000 target batches and FD-pair sequence. Weight zero is the existing "
             "restarted control.", "", "| Pretraining | Pair weight | Median training-pair "
             "delta L2 | Mean development full-trace L2 | Development change vs control |",
             "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['arm']} | {row['weight']:.3f} | "
                     f"{row['median_training_pair_delta_relative_l2']:.3f} | "
                     f"{row['mean_development_full_trace_relative_l2']:.3f} | "
                     f"{row['development_delta_from_control']:+.3f} |")
    lines.extend(["", "Lower values are better. Selection from this table would be "
                  "development tuning and requires a fresh locked localization set.", "",
                  sweep["restriction"]])
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "evaluate"))
    args = parser.parse_args()
    values = read_setup()
    if args.stage == "train":
        train(values[0], values[1], values[3], values[4], values[5], values[6], values[7])
    else:
        evaluate(*values)


if __name__ == "__main__":
    main()
