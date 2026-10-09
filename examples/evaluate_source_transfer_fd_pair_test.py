"""Evaluate the selected FD-pair continuation on the locked prospective test set."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.diagnose_source_transfer_trajectory import config as trajectory_config
from examples.run_source_transfer_fd_pair import setup as pair_setup
from examples.run_source_transfer_fd_pair_weight import PROTOCOL as SWEEP_PROTOCOL, checkpoint
from examples.run_source_transfer_moe import digest, new_model, read_setup as parent_setup
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, choose_diverse_starts, coarse_grid, load_event, pick_arrivals,
    refine, sampled_waveforms, waveform_scores,
)

PROTOCOL = ROOT / "experiments/source_transfer_fd_pair_test_protocol.json"


def main():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sweep = json.loads(SWEEP_PROTOCOL.read_text(encoding="utf-8"))
    base, control_spec, _, trajectory, parent, setup, paths = pair_setup()
    parent_spec, _, loc, _ = parent_setup()
    trajectory_spec, _, _, _, _ = trajectory_config()
    if (ROOT / spec["weight_sweep_protocol"] != SWEEP_PROTOCOL
            or spec["selected_weight"] not in sweep["weights"]
            or spec["selected_arm"] not in sweep["arms"]
            or spec["seeds"] != sweep["seeds"]
            or spec["test_events"] != [row["file"] for row in parent_spec["test_events"]]
            or ROOT / spec["parent_protocol"] != ROOT / "experiments/source_transfer_moe_protocol.json"
            or ROOT / spec["control_protocol"] != ROOT / "experiments/source_transfer_local_rank_protocol.json"):
        raise ValueError("Locked test protocol differs from registered parent protocols")
    output = ROOT / spec["result_json"]
    report = ROOT / spec["report_file"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite locked test result")
    test_dir = ROOT / parent_spec["test_observation_directory"]
    test_paths = [test_dir / name for name in spec["test_events"]]
    if not all(path.is_file() for path in test_paths):
        raise FileNotFoundError("Locked test observations are missing")
    receivers = np.load(ROOT / "data/observations/receiver_coords.npy").astype(np.float32)
    events = []
    for path in test_paths:
        event = load_event(path, receivers)
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        events.append((event, truth, digest(path)))
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    bounds = np.asarray(loc["source_bounds_m_s"], dtype=np.float32)
    search, arrival = loc["search"], loc["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    branches = {
        "restarted_control": lambda seed: ROOT / control_spec["checkpoint_directory"] /
        f"{spec['selected_arm']}_seed{seed}_restarted_control.pth",
        "fd_pair_weight_0.05": lambda seed: checkpoint(
            sweep, spec["selected_arm"], seed, spec["selected_weight"]),
    }
    rows = []
    for seed in spec["seeds"]:
        initial = ROOT / trajectory_spec["intermediate_checkpoint_directory"] / \
            f"{spec['selected_arm']}_seed{seed}_target{base['initial_target_updates']}.pth"
        for branch, path_fn in branches.items():
            path = path_fn(seed)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            expected = digest(ROOT / spec["control_protocol"]
                              if branch == "restarted_control" else SWEEP_PROTOCOL)
            if (payload["protocol_sha256"] != expected
                    or payload["initial_checkpoint_sha256"] != digest(initial)
                    or payload["arm"] != spec["selected_arm"]
                    or payload["seed"] != seed):
                raise ValueError(f"Checkpoint provenance mismatch: {path}")
            model = new_model(fields, receivers, setup, device)
            model.load_state_dict(payload["state_dict"])
            model.eval().requires_grad_(False)
            for event, truth, event_hash in events:
                _, picks = pick_arrivals(event, arrival)
                valid = int(np.isfinite(picks).sum())
                if valid < arrival["minimum_valid_picks"]:
                    raise ValueError(f"Insufficient picks for {event['path']}")
                coarse = sampled_waveforms(event, domain,
                                           search["coarse_receiver_stride"],
                                           search["coarse_time_stride"], device)
                fine = sampled_waveforms(event, domain,
                                         search["refine_receiver_stride"],
                                         search["refine_time_stride"], device)
                started = time.perf_counter()
                wave = waveform_scores(model, candidates, domain, coarse,
                                       search["candidate_batch"])
                travel = arrival_scores(model, candidates, picks,
                                        arrival["arrival_scale_s"],
                                        search["candidate_batch"])
                predictions = {}
                for name, alpha in (("waveform_only", 0.0),
                                    ("hybrid", arrival["hybrid_weight"])):
                    score = wave + alpha * travel
                    ids = choose_diverse_starts(candidates, score, search["restarts"],
                                                search["minimum_start_separation_m"])
                    starts = [{"source_m_s": candidates[i].tolist(),
                               "coarse_total_loss": float(score[i])} for i in ids]
                    best, _ = refine(model, starts, domain, bounds, fine, picks,
                                     search, arrival, alpha)
                    estimate = np.asarray(best["estimated_source_m_s"], dtype=np.float64)
                    predictions[name] = {
                        "estimated_source_m_s": estimate.tolist(),
                        "location_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
                        "origin_time_error_s": float(abs(estimate[3] - truth[3])),
                        "waveform_normalized_mse": float(best["waveform_normalized_mse"]),
                    }
                rows.append({"seed": seed, "branch": branch,
                             "event": event["path"].name, "sha256": event_hash,
                             "truth_m_s": truth.tolist(), "valid_picks": valid,
                             "search_seconds": time.perf_counter() - started,
                             "predictions": predictions})
                print(f"{branch} seed={seed} {event['path'].name}: "
                      f"hybrid={predictions['hybrid']['location_error_m']:.1f} m",
                      flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = {}
    for branch in branches:
        group = [r for r in rows if r["branch"] == branch]
        hybrid = [r["predictions"]["hybrid"] for r in group]
        wave = [r["predictions"]["waveform_only"] for r in group]
        summary[branch] = {
            "outcomes": len(group),
            "median_hybrid_location_error_m": float(np.median([r["location_error_m"] for r in hybrid])),
            "mean_hybrid_location_error_m": float(np.mean([r["location_error_m"] for r in hybrid])),
            "median_hybrid_origin_time_error_s": float(np.median([r["origin_time_error_s"] for r in hybrid])),
            "mean_waveform_only_location_error_m": float(np.mean([r["location_error_m"] for r in wave])),
            "hybrid_improves_over_control_count": None,
        }
    control_rows = {(r["seed"], r["event"]): r for r in rows
                    if r["branch"] == "restarted_control"}
    pair_rows = {(r["seed"], r["event"]): r for r in rows
                 if r["branch"] == "fd_pair_weight_0.05"}
    differences = []
    for key, control in control_rows.items():
        modified = pair_rows[key]
        differences.append(modified["predictions"]["hybrid"]["location_error_m"]
                           - control["predictions"]["hybrid"]["location_error_m"])
    summary["fd_pair_weight_0.05"]["hybrid_improves_over_control_count"] = sum(
        value < 0 for value in differences)
    result = {"protocol_sha256": digest(PROTOCOL),
              "parent_protocol_sha256": digest(ROOT / spec["parent_protocol"]),
              "control_protocol_sha256": digest(ROOT / spec["control_protocol"]),
              "sweep_protocol_sha256": digest(SWEEP_PROTOCOL), "device": str(device),
              "summary": summary, "paired_hybrid_location_differences_m": differences,
              "rows": rows, "restriction": spec["restriction"]}
    lines = ["# Locked FD-pair continuation test", "",
             "The four reserved source_transfer_moe_test events were evaluated once "
             "with identical search, picker, FMM fields and hybrid weight.", "",
             "| Branch | Outcomes | Median hybrid eloc (m) | Mean hybrid eloc (m) | "
             "Median hybrid et (s) |", "|---|---:|---:|---:|---:|"]
    for branch, row in summary.items():
        if branch.startswith("fd_pair") or branch == "restarted_control":
            lines.append(f"| {branch} | {row['outcomes']} | "
                         f"{row['median_hybrid_location_error_m']:.1f} | "
                         f"{row['mean_hybrid_location_error_m']:.1f} | "
                         f"{row['median_hybrid_origin_time_error_s']:.4f} |")
    lines.extend(["", f"The selected branch improves hybrid location error in "
                  f"{summary['fd_pair_weight_0.05']['hybrid_improves_over_control_count']}/12 "
                  "paired seed-event outcomes. Negative paired differences favor the "
                  "selected branch; all differences are stored in JSON.", "", spec["restriction"]])
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
