"""Trace paired MoE transfer through target fine-tuning on development records."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.run_source_transfer_moe import (
    OUTPUT as FINAL_OUTPUT, PROTOCOL as PARENT_PROTOCOL, digest, new_model,
    phase, read_setup,
)
from examples.run_traveltime_aligned_moe_development import (
    aligned_prediction, load_observations, relative_l2,
)
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, choose_diverse_starts, coarse_grid, load_event,
    pick_arrivals, refine, sampled_waveforms, waveform_scores,
)

PROTOCOL = ROOT / "experiments/source_transfer_trajectory_protocol.json"


def config():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent, setup, loc, target_paths = read_setup()
    if ROOT / spec["parent_protocol"] != PARENT_PROTOCOL:
        raise ValueError("Trajectory parent protocol changed")
    if spec["arms"] != parent["arms"][:2] or spec["seeds"] != parent["seeds"]:
        raise ValueError("Trajectory arms or seeds differ from original pilot")
    if spec["target_updates"] != [0, 500, parent["phase_updates"]]:
        raise ValueError("Unexpected target update milestones")
    return spec, parent, setup, loc, target_paths


def intermediate(spec, arm, seed, step):
    return ROOT / spec["intermediate_checkpoint_directory"] / f"{arm}_seed{seed}_target{step}.pth"


def prepare(spec, parent, setup, paths):
    source_paths = [ROOT / parent["source_observation_directory"] / f"source_{i:02d}.npz"
                    for i in parent["source_training_indices"]]
    source_events, receivers, times = load_observations(source_paths)
    target_events, target_receivers, target_times = load_observations(paths)
    if not np.array_equal(receivers, target_receivers) or not np.array_equal(times, target_times):
        raise ValueError("Acquisition mismatch")
    source_fields = np.load(ROOT / parent["source_fmm_fields"], mmap_mode="r")
    target_fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            first_path = intermediate(spec, arm, seed, 0)
            middle_path = intermediate(spec, arm, seed, 500)
            if first_path.exists() and middle_path.exists():
                print(f"Preserving both intermediate checkpoints: {arm} {seed}", flush=True)
                continue
            if first_path.exists() or middle_path.exists():
                raise FileExistsError(f"Incomplete intermediate pair: {arm} {seed}")
            final_path = FINAL_OUTPUT / f"{arm}_seed{seed}.pth"
            final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
            if (final_payload["protocol_sha256"] != digest(PARENT_PROTOCOL)
                    or final_payload["arm"] != arm or final_payload["seed"] != seed):
                raise ValueError(f"Original final checkpoint does not match: {final_path}")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            first_is_source = arm == "source_to_target"
            model = new_model(source_fields if first_is_source else target_fields,
                              receivers, setup, device)
            first_events = (source_events if first_is_source
                            else [target_events[i] for i in parent["source_training_indices"]])
            phase(model, first_events, receivers, times,
                  source_fields if first_is_source else target_fields,
                  setup, rng, parent["phase_updates"], device)
            if first_is_source:
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                model = new_model(target_fields, receivers, setup, device)
                model.load_state_dict(state)
            first_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {"arm": arm, "seed": seed,
                        "parent_protocol_sha256": digest(PARENT_PROTOCOL),
                        "trajectory_protocol_sha256": digest(PROTOCOL),
                        "target_fmm_sha256": digest(ROOT / parent["target_fmm_fields"]),
                        "final_checkpoint_sha256": digest(final_path)}
            torch.save({**metadata, "target_updates": 0,
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                       first_path)
            phase(model, target_events, receivers, times, target_fields,
                  setup, rng, 500, device)
            torch.save({**metadata, "target_updates": 500,
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                       middle_path)
            print(f"Saved trajectory: {arm} seed={seed}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def locate(model, event, receivers, domain, candidates, bounds, search, arrival, device):
    observation = load_event(event["path"], receivers)
    _, corrected = pick_arrivals(observation, arrival)
    valid = int(np.isfinite(corrected).sum())
    if valid < arrival["minimum_valid_picks"]:
        return {"valid_picks": valid, "failure": "insufficient picks"}
    coarse = sampled_waveforms(observation, domain, search["coarse_receiver_stride"],
                               search["coarse_time_stride"], device)
    fine = sampled_waveforms(observation, domain, search["refine_receiver_stride"],
                             search["refine_time_stride"], device)
    wave = waveform_scores(model, candidates, domain, coarse, search["candidate_batch"])
    travel = arrival_scores(model, candidates, corrected,
                            arrival["arrival_scale_s"], search["candidate_batch"])
    # The FMM constraint is identical for both arms, but their waveform ranking differs.
    score = wave + arrival["hybrid_weight"] * travel
    indices = choose_diverse_starts(candidates, score, search["restarts"],
                                    search["minimum_start_separation_m"])
    starts = [{"source_m_s": candidates[index].tolist(),
               "coarse_total_loss": float(score[index])} for index in indices]
    best, _ = refine(model, starts, domain, bounds, fine, corrected,
                     search, arrival, arrival["hybrid_weight"])
    estimate = np.asarray(best["estimated_source_m_s"])
    truth = event["source"]
    return {"valid_picks": valid, "estimate_m_s": estimate.tolist(),
            "location_error_m": float(np.linalg.norm((estimate-truth)[:3])),
            "origin_time_error_s": float(abs(estimate[3]-truth[3])),
            "waveform_normalized_mse": best["waveform_normalized_mse"],
            "arrival_normalized_mse": best["arrival_normalized_mse"]}


def evaluate(spec, parent, setup, loc):
    result_path = ROOT / spec["result_file"]
    report_path = ROOT / spec["report_file"]
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite trajectory results")
    paths = [path for directory in spec["development_directories"]
             for path in sorted((ROOT / directory).glob("*.npz"))]
    if len(paths) != spec["development_event_count"]:
        raise ValueError("Development set size changed")
    events, receivers, times = load_observations(paths)
    event_map = {str(p.relative_to(ROOT)).replace("\\", "/"): e
                 for p, e in zip(paths, events)}
    selected = [event_map[name] for name in spec["localization_events"]]
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    bounds = np.asarray(loc["source_bounds_m_s"], dtype=np.float32)
    search, arrival = loc["search"], loc["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            final = FINAL_OUTPUT / f"{arm}_seed{seed}.pth"
            for step in spec["target_updates"]:
                checkpoint = (final if step == parent["phase_updates"]
                              else intermediate(spec, arm, seed, step))
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if step < parent["phase_updates"]:
                    if (payload["trajectory_protocol_sha256"] != digest(PROTOCOL)
                            or payload["final_checkpoint_sha256"] != digest(final)):
                        raise ValueError(f"Invalid trajectory checkpoint: {checkpoint}")
                elif payload["protocol_sha256"] != digest(PARENT_PROTOCOL):
                    raise ValueError(f"Invalid final checkpoint: {checkpoint}")
                model = new_model(fields, receivers, setup, device)
                model.load_state_dict(payload["state_dict"])
                model.eval().requires_grad_(False)
                for event in events:
                    predicted = aligned_prediction(model, event, receivers, times,
                                                   domain, device)
                    l2 = relative_l2(event["data"] / event["rms"], predicted)
                    rows.append({"seed": seed, "arm": arm, "target_updates": step,
                                 "event": str(event["path"].relative_to(ROOT)).replace("\\", "/"),
                                 "sha256": event["sha256"], "full_waveform_relative_l2": l2})
                for event in selected:
                    key = str(event["path"].relative_to(ROOT)).replace("\\", "/")
                    row = next(r for r in rows if r["seed"] == seed and r["arm"] == arm
                               and r["target_updates"] == step and r["event"] == key)
                    row["localization"] = locate(model, event, receivers, domain, candidates,
                                                  bounds, search, arrival, device)
                    print(f"{seed} {arm} target={step} {event['path'].parent.name}/{event['path'].name}: "
                          f"L2={row['full_waveform_relative_l2']:.3f}, "
                          f"eloc={row['localization'].get('location_error_m', float('nan')):.1f} m",
                          flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    summary = {}
    for arm in spec["arms"]:
        for step in spec["target_updates"]:
            subset = [r for r in rows if r["arm"] == arm and r["target_updates"] == step]
            local = [r["localization"] for r in subset if "localization" in r]
            errors = [r["location_error_m"] for r in local if "location_error_m" in r]
            summary[f"{arm}_{step}"] = {
                "mean_full_waveform_relative_l2": float(np.mean([
                    r["full_waveform_relative_l2"] for r in subset])),
                "median_hybrid_location_error_m": float(np.median(errors)) if errors else None,
                "median_origin_time_error_s": float(np.median([
                    r["origin_time_error_s"] for r in local
                    if "origin_time_error_s" in r])) if errors else None,
                "location_successes": len(errors), "location_attempts": len(local),
            }
    output = {"purpose": spec["purpose"], "protocol_sha256": digest(PROTOCOL),
              "parent_protocol_sha256": digest(PARENT_PROTOCOL),
              "device": str(device), "development_only": True,
              "selected_event_count": len(selected), "all_development_event_count": len(events),
              "summary": summary, "rows": rows, "restriction": spec["restriction"]}
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = ["# Source transfer trajectory, development only", "",
             "The original four-event test set was excluded. The same target FMM fields, "
             "arrival picker, bias and localization settings were used at each stage.", "",
             "| Arm | Target updates | Mean waveform relative L2 (16 events x 3 seeds) | "
             "Median hybrid location error (4 events x 3 seeds) | Median origin-time error (s) |",
             "|---|---:|---:|---:|---:|"]
    for arm in spec["arms"]:
        for step in spec["target_updates"]:
            row = summary[f"{arm}_{step}"]
            lines.append(f"| {arm} | {step} | {row['mean_full_waveform_relative_l2']:.4f} | "
                         f"{row['median_hybrid_location_error_m']:.1f} m | "
                         f"{row['median_origin_time_error_s']:.4f} |")
    lines.extend(["", spec["restriction"],
                  "", "The 1500-update checkpoints are the original frozen pilot models; "
                  "the 0 and 500-update checkpoints are reconstructed with the same seeds "
                  "and sampling sequence. The target optimizer is restarted at the domain switch."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    args = parser.parse_args()
    spec, parent, setup, loc, paths = config()
    if args.stage == "prepare":
        prepare(spec, parent, setup, paths)
    else:
        evaluate(spec, parent, setup, loc)


if __name__ == "__main__":
    main()
