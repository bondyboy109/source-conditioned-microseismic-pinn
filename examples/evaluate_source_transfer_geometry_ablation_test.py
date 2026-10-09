"""Evaluate the geometry ablation checkpoints on the locked four-event test set."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.run_source_transfer_geometry_ablation import new_model
from examples.run_source_transfer_moe import digest, read_setup
from examples.run_traveltime_aligned_moe_development import aligned_prediction, load_observations, relative_l2
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, choose_diverse_starts, coarse_grid, load_event, pick_arrivals,
    refine, sampled_waveforms, waveform_scores,
)

PROTOCOL = ROOT / "experiments/source_transfer_geometry_ablation_test_protocol.json"


def main():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent, setup, loc, _ = read_setup()
    parent_spec = json.loads((ROOT / spec["parent_protocol"]).read_text(encoding="utf-8"))
    ablation_spec = json.loads((ROOT / spec["ablation_protocol"]).read_text(encoding="utf-8"))
    test_paths = [ROOT / parent_spec["test_observation_directory"] / row["file"]
                  for row in parent_spec["test_events"]]
    if not all(p.is_file() for p in test_paths):
        raise FileNotFoundError("Locked test records are missing")
    receivers = np.load(ROOT / "data/observations/receiver_coords.npy").astype(np.float32)
    events, receiver_check, _ = load_observations(test_paths)
    if not np.array_equal(receivers, receiver_check):
        raise ValueError("Test receiver mismatch")
    fields = np.load(ROOT / parent_spec["target_fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    bounds = np.asarray(loc["source_bounds_m_s"], dtype=np.float32)
    search, arrival = loc["search"], loc["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    arms = ["source_to_target_no_geometry", "source_to_target_geometry",
            "target_to_target_no_geometry", "target_to_target_geometry"]
    rows = []
    for seed in ablation_spec["seeds"]:
        for arm in arms:
            checkpoint = ROOT / spec["checkpoint_directory"] / f"{arm}_seed{seed}.pth"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["protocol_sha256"] != digest(ROOT / spec["ablation_protocol"]):
                raise ValueError(f"Ablation protocol changed after training: {checkpoint}")
            geometric = not arm.endswith("_no_geometry")
            model = new_model(fields, receivers, setup, device, geometric)
            model.load_state_dict(payload["state_dict"])
            model.eval().requires_grad_(False)
            for event, truth_row in zip(events, parent_spec["test_events"]):
                observed = load_event(event["path"], receivers)
                _, picks = pick_arrivals(observed, arrival)
                if int(np.isfinite(picks).sum()) < arrival["minimum_valid_picks"]:
                    raise ValueError(f"Insufficient picks for {event['path']}")
                coarse = sampled_waveforms(observed, domain, search["coarse_receiver_stride"],
                                           search["coarse_time_stride"], device)
                fine = sampled_waveforms(observed, domain, search["refine_receiver_stride"],
                                         search["refine_time_stride"], device)
                started = time.perf_counter()
                wave = waveform_scores(model, candidates, domain, coarse, search["candidate_batch"])
                fmm = arrival_scores(model, candidates, picks, arrival["arrival_scale_s"],
                                     search["candidate_batch"])
                predictions = {}
                for name, alpha in (("waveform_only", 0.0), ("hybrid", arrival["hybrid_weight"])):
                    score = wave + alpha * fmm
                    ids = choose_diverse_starts(candidates, score, search["restarts"],
                                                search["minimum_start_separation_m"])
                    starts = [{"source_m_s": candidates[i].tolist(),
                               "coarse_total_loss": float(score[i])} for i in ids]
                    best, _ = refine(model, starts, domain, bounds, fine, picks, search, arrival, alpha)
                    estimate = np.asarray(best["estimated_source_m_s"])
                    truth = np.asarray(truth_row["source_m_s"])
                    predictions[name] = {
                        "estimated_source_m_s": estimate.tolist(),
                        "location_error_m": float(np.linalg.norm((estimate - truth)[:3])),
                        "origin_time_error_s": float(abs(estimate[3] - truth[3])),
                        "waveform_normalized_mse": best["waveform_normalized_mse"],
                    }
                if device.type == "cuda":
                    torch.cuda.synchronize()
                full = aligned_prediction(model, event, receivers, observed["times"], domain, device)
                l2 = relative_l2(event["data"] / event["rms"], full)
                rows.append({"seed": seed, "arm": arm, "event": truth_row["file"],
                             "truth_m_s": truth_row["source_m_s"], "sha256": event["sha256"],
                             "full_waveform_relative_l2": l2,
                             "search_seconds": time.perf_counter() - started,
                             "predictions": predictions})
                print(f"{seed} {arm} {truth_row['file']}: hybrid={predictions['hybrid']['location_error_m']:.1f} m, L2={l2:.3f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = {}
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            "outcomes": len(selected),
            "median_hybrid_location_error_m": float(np.median([r["predictions"]["hybrid"]["location_error_m"] for r in selected])),
            "mean_hybrid_location_error_m": float(np.mean([r["predictions"]["hybrid"]["location_error_m"] for r in selected])),
            "median_hybrid_origin_time_error_s": float(np.median([r["predictions"]["hybrid"]["origin_time_error_s"] for r in selected])),
            "mean_full_waveform_relative_l2": float(np.mean([r["full_waveform_relative_l2"] for r in selected])),
        }
    result_path = ROOT / spec["result_json"]
    report_path = ROOT / spec["report_file"]
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite locked test results")
    result = {"purpose": spec["purpose"], "device": str(device), "summary": summary,
              "rows": rows, "restriction": spec["restriction"],
              "ablation_protocol_sha256": digest(ROOT / spec["ablation_protocol"])}
    lines = ["# Locked test: geometry-feature ablation", "",
             "No test event was used during training or model selection.", "",
             "| Arm | Outcomes | Median hybrid eloc (m) | Mean hybrid eloc (m) | Median hybrid et (s) | Mean full waveform relative L2 |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        row = summary[arm]
        lines.append(f"| {arm} | {row['outcomes']} | {row['median_hybrid_location_error_m']:.1f} | "
                     f"{row['mean_hybrid_location_error_m']:.1f} | {row['median_hybrid_origin_time_error_s']:.4f} | "
                     f"{row['mean_full_waveform_relative_l2']:.4f} |")
    lines.extend(["", spec["restriction"]])
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
