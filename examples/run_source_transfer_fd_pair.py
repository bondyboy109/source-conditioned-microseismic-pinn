"""Continue paired transfer with observed FD differences between nearby sources."""

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
from examples.diagnose_source_transfer_trajectory import config, intermediate
from examples.run_source_transfer_moe import digest, new_model
from examples.run_traveltime_aligned_moe_development import (
    aligned_prediction, load_observations, relative_l2, sample_training_batch,
)
from examples.invert_fmm_receiver_fields import interpolate_fields

PROTOCOL = ROOT / "experiments/source_transfer_fd_pair_protocol.json"


def setup():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    trajectory, parent, model_spec, _, paths = config()
    control_spec = json.loads((ROOT / spec["control_protocol"]).read_text(encoding="utf-8"))
    audit_spec = json.loads((ROOT / spec["pair_audit_protocol"]).read_text(encoding="utf-8"))
    audit = json.loads((ROOT / audit_spec["result_json"]).read_text(encoding="utf-8"))
    if (spec["seeds"] != trajectory["seeds"] or spec["arms"] != trajectory["arms"]
            or ROOT / spec["trajectory_protocol"] != ROOT / control_spec["trajectory_protocol"]
            or ROOT / spec["parent_protocol"] != ROOT / trajectory["parent_protocol"]
            or spec["initial_target_updates"] != control_spec["initial_target_updates"]
            or spec["additional_target_updates"] != control_spec["additional_target_updates"]
            or audit["protocol_sha256"] != digest(ROOT / spec["pair_audit_protocol"])):
        raise ValueError("Frozen parent, control or FD-pair provenance changed")
    if not (0 <= spec["pair_sample_active_fraction"] <= 1):
        raise ValueError("Invalid active fraction")
    return spec, control_spec, audit, trajectory, parent, model_spec, paths


def paired_batch(events, receivers, times, pair, count, active_fraction, rng,
                 domain, device):
    a, b = events[pair["a"]], events[pair["b"]]
    nt = len(times)
    n_active = int(round(count * active_fraction))
    active_a = rng.choice(a["active"], size=n_active // 2, replace=True)
    active_b = rng.choice(b["active"], size=n_active - len(active_a), replace=True)
    other = rng.integers(0, len(receivers) * nt, size=count - n_active)
    flat = rng.permutation(np.concatenate((active_a, active_b, other)))
    receiver_index = flat // nt
    time_index = flat % nt
    points = np.concatenate((receivers[receiver_index], times[time_index, None]), axis=1)
    target = b["data"][receiver_index, time_index] / b["rms"] - (
        a["data"][receiver_index, time_index] / a["rms"])
    return (torch.as_tensor(points / domain, device=device, dtype=torch.float32),
            torch.as_tensor(receiver_index, device=device, dtype=torch.long),
            torch.as_tensor(a["source"] / domain, device=device).expand(count, 4),
            torch.as_tensor(b["source"] / domain, device=device).expand(count, 4),
            torch.as_tensor(target[:, None], device=device, dtype=torch.float32))


def train(spec, audit, trajectory, parent, model_spec, paths):
    events, receivers, times = load_observations(paths)
    for pair in audit["pairs"]:
        if (np.linalg.norm(events[pair["a"]]["source"][:3]
                           - events[pair["b"]]["source"][:3])
                - pair["separation_m"]) > 1e-4:
            raise ValueError("FD pair source positions changed")
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    arrivals = interpolate_fields(fields, np.stack([e["source"][:3] for e in events]), 80.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*model_spec["model"]["spatial_lengths_m"],
                         model_spec["model"]["time_length_s"]], dtype=np.float32)
    options = model_spec["training"]
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            output = ROOT / spec["checkpoint_directory"] / f"{arm}_seed{seed}.pth"
            if output.exists():
                payload = torch.load(output, map_location="cpu", weights_only=False)
                if payload["protocol_sha256"] != digest(PROTOCOL):
                    raise ValueError(f"Existing checkpoint has different protocol: {output}")
                print(f"Preserving {output.name}", flush=True)
                continue
            initial = intermediate(trajectory, arm, seed, spec["initial_target_updates"])
            payload = torch.load(initial, map_location="cpu", weights_only=False)
            if (payload["trajectory_protocol_sha256"] != digest(
                    ROOT / spec["trajectory_protocol"])
                    or payload["target_updates"] != spec["initial_target_updates"]):
                raise ValueError(f"Initial checkpoint provenance mismatch: {initial}")
            model = new_model(fields, receivers, model_spec, device)
            model.load_state_dict(payload["state_dict"])
            optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
            batch_rng = np.random.default_rng(seed + 50000)
            pair_rng = np.random.default_rng(seed + 70000)
            last = {}
            for step in range(1, spec["additional_target_updates"] + 1):
                model.train()
                points, sources, receiver_ids, target = sample_training_batch(
                    events, receivers, times, arrivals, model_spec, batch_rng, domain, device)
                optimizer.zero_grad(set_to_none=True)
                prediction, gates, _ = model(points, sources, receiver_ids)
                positive = (prediction - target).square().mean()
                balance = (gates.mean(0) - 1.0 / model.expert_count).square().mean()
                loss = (positive + options["gate_balance_weight"] * balance
                        + options["prototype_separation_weight"] *
                        model.prototype_separation_loss())
                pair = audit["pairs"][int(pair_rng.integers(0, len(audit["pairs"])))]
                x, ids, src_a, src_b, delta = paired_batch(
                    events, receivers, times, pair, spec["pair_samples_per_update"],
                    spec["pair_sample_active_fraction"], pair_rng, domain, device)
                delta_hat = model(x, src_b, ids)[0] - model(x, src_a, ids)[0]
                pair_loss = (delta_hat - delta).square().mean()
                loss = loss + spec["pair_loss_weight"] * pair_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip_norm"])
                optimizer.step()
                with torch.no_grad():
                    model.prototype_centers.clamp_(
                        *model_spec["model"]["prototype_bounds_normalized"])
                if step % 500 == 0:
                    last = {"positive_mse": float(positive.detach()),
                            "fd_pair_difference_mse": float(pair_loss.detach()),
                            "total_loss": float(loss.detach())}
                    print(f"{arm} seed={seed} step={step}: {last}", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in
                                       model.state_dict().items()},
                        "protocol_sha256": digest(PROTOCOL),
                        "initial_checkpoint_sha256": digest(initial),
                        "pair_audit_sha256": digest(ROOT / json.loads(
                            (ROOT / spec["pair_audit_protocol"]).read_text(
                                encoding="utf-8"))["result_json"]),
                        "target_fmm_sha256": digest(ROOT / parent["target_fmm_fields"]),
                        "arm": arm, "seed": seed,
                        "last_training_loss": last}, output)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def evaluate(spec, control_spec, audit, trajectory, parent, model_spec, paths):
    output = ROOT / spec["result_json"]
    report = ROOT / spec["report_file"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite FD-pair results")
    events, receivers, times = load_observations(paths)
    development_paths = [path for directory in trajectory["development_directories"]
                         for path in sorted((ROOT / directory).glob("*.npz"))]
    if len(development_paths) != trajectory["development_event_count"]:
        raise ValueError("Development event count changed")
    development, dev_receivers, dev_times = load_observations(development_paths)
    if not np.array_equal(receivers, dev_receivers) or not np.array_equal(times, dev_times):
        raise ValueError("Development acquisition differs")
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*model_spec["model"]["spatial_lengths_m"],
                         model_spec["model"]["time_length_s"]], dtype=np.float32)
    sampled = [(e["data"][:, ::4] / e["rms"]).astype(np.float64).reshape(-1)
               for e in events]
    rows = []
    for arm in spec["arms"]:
        for seed in spec["seeds"]:
            initial = intermediate(trajectory, arm, seed, spec["initial_target_updates"])
            for branch in ("restarted_control", "fd_pair"):
                checkpoint = (ROOT / control_spec["checkpoint_directory"] /
                              f"{arm}_seed{seed}_restarted_control.pth" if branch == "restarted_control"
                              else ROOT / spec["checkpoint_directory"] / f"{arm}_seed{seed}.pth")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                protocol = ROOT / (spec["control_protocol"] if branch == "restarted_control"
                                   else "experiments/source_transfer_fd_pair_protocol.json")
                if (payload["protocol_sha256"] != digest(protocol)
                        or payload["initial_checkpoint_sha256"] != digest(initial)
                        or payload["target_fmm_sha256"] != digest(ROOT / parent["target_fmm_fields"])):
                    raise ValueError(f"Checkpoint mismatch: {checkpoint}")
                model = new_model(fields, receivers, model_spec, device)
                model.load_state_dict(payload["state_dict"])
                model.eval()
                predictions = [response(model, e, receivers, times, domain, 4, device)
                               for e in events]
                for pair in audit["pairs"]:
                    a, b = pair["a"], pair["b"]
                    actual = sampled[b] - sampled[a]
                    estimate = predictions[b] - predictions[a]
                    rows.append({"arm": arm, "seed": seed, "branch": branch,
                                 "kind": "training_pair", "a": a, "b": b,
                                 "relative_l2": float(np.linalg.norm(estimate - actual)
                                                      / np.linalg.norm(actual))})
                for event in development:
                    predicted = aligned_prediction(model, event, receivers, times,
                                                   domain, device)
                    rows.append({"arm": arm, "seed": seed, "branch": branch,
                                 "kind": "development_event", "event": str(event["path"]),
                                 "relative_l2": relative_l2(event["data"] / event["rms"],
                                                            predicted)})
                print(f"{arm} seed={seed} {branch}: FD pairs and development traces evaluated",
                      flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    summary = {}
    for arm in spec["arms"]:
        for branch in ("restarted_control", "fd_pair"):
            group = [r for r in rows if r["arm"] == arm and r["branch"] == branch]
            summary[f"{arm}_{branch}"] = {
                "median_training_pair_difference_relative_l2": float(np.median([
                    r["relative_l2"] for r in group if r["kind"] == "training_pair"])),
                "mean_development_full_trace_relative_l2": float(np.mean([
                    r["relative_l2"] for r in group if r["kind"] == "development_event"])),
            }
    result = {"protocol_sha256": digest(PROTOCOL), "control_protocol_sha256": digest(
        ROOT / spec["control_protocol"]), "device": str(device),
        "pair_count": len(audit["pairs"]), "development_event_count": len(development),
        "summary": summary, "rows": rows, "restriction": spec["restriction"]}
    lines = ["# FD neighboring-source difference supervision pilot", "",
             "The 500-update checkpoint and 1000 subsequent target batches match the "
             "existing restarted control. The new branch adds same-receiver/time FD "
             "waveform differences for observed training-source pairs.", "",
             "| Pretraining | Continuation | Median training-pair delta relative L2 | "
             "Mean development full-trace relative L2 |", "|---|---|---:|---:|"]
    for arm in spec["arms"]:
        for branch in ("restarted_control", "fd_pair"):
            s = summary[f"{arm}_{branch}"]
            lines.append(f"| {arm} | {branch} | "
                         f"{s['median_training_pair_difference_relative_l2']:.3f} | "
                         f"{s['mean_development_full_trace_relative_l2']:.3f} |")
    lines.extend(["", "Lower relative L2 is better. The 31 pair errors are in-sample; "
                  "the 16 development waveforms were previously inspected. "
                  "This experiment does not measure localization performance.", "",
                  spec["restriction"]])
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "evaluate"))
    args = parser.parse_args()
    spec, control_spec, audit, trajectory, parent, model_spec, paths = setup()
    if args.stage == "train":
        train(spec, audit, trajectory, parent, model_spec, paths)
    else:
        evaluate(spec, control_spec, audit, trajectory, parent, model_spec, paths)


if __name__ == "__main__":
    main()
