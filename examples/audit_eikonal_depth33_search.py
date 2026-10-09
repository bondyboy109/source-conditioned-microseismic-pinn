"""Audit coarse ranking and a fixed wider multistart search on spent FD data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.confirm_depth_gap_25 import refine
from examples.run_fmm_seeded_waveform_development import (
    choose_spatial_starts, fine_scores, hash_file, proposals,
)
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_depth33_search_development_protocol.json"


def verify(protocol: dict) -> tuple[list[dict], list[dict], dict]:
    for key, digest in (("new_checkpoint", "new_sha256"),
                        ("old_checkpoint", "old_sha256"),
                        ("fmm_results", "fmm_sha256"),
                        ("comparison", "comparison_sha256"),
                        ("prior_localization", "prior_localization_sha256"),
                        ("prior_search_protocol", "prior_search_sha256")):
        if hash_file(ROOT / protocol[key]) != protocol[digest]:
            raise ValueError(f"Registered input changed: {protocol[key]}")
    previous = json.loads((ROOT / protocol["prior_search_protocol"]).read_text())
    if (protocol["original_offsets_m"] != previous["spatial_offsets_m_each_axis"]
            or protocol["bounds_m_s"] != previous["refinement"]["bounds_m_s"]
            or protocol["refine_steps"] != previous["refinement"]["steps"]
            or protocol["refine_adam_learning_rate_on_box_fractions"]
            != previous["refinement"]["adam_learning_rate_on_box_fractions"]
            or protocol["time_stride"] != previous["fine_scoring"]["time_stride"]
            or protocol["receiver_stride"] != previous["fine_scoring"]["receiver_stride"]):
        raise ValueError("Expanded search does not retain the original scoring/refinement")
    comparison = json.loads((ROOT / protocol["comparison"]).read_text())["events"]
    fmm = json.loads((ROOT / protocol["fmm_results"]).read_text())["events"]
    old = json.loads((ROOT / protocol["prior_localization"]).read_text())
    if len(comparison) != len(fmm) != len(old["rows"]) != 8:
        raise ValueError("Expected exactly eight paired spent regional records")
    models = {}
    for label in ("old", "new"):
        model, payload = load_conditioned_checkpoint(
            ROOT / protocol[f"{label}_checkpoint"],
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval().requires_grad_(False)
        models[label] = (model, np.asarray([payload["config"][key] for key in
                      ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                      dtype=np.float32))
        trained = {item["sha256"] for item in
                   payload["training_record"]["receiver_observations"]}
        for record in comparison:
            if record["sha256"] in trained:
                raise ValueError(f"Regional record is in {label} training set")
    for record, arrival, prior in zip(comparison, fmm, old["rows"]):
        if (arrival["sha256"] != record["sha256"]
                or prior["event"] != record["event"]
                or hash_file(Path(record["observation"])) != record["sha256"]):
            raise ValueError("FD pairing or content changed")
    return comparison, fmm, models


def rank_audit(candidates: np.ndarray, losses: np.ndarray,
               starts: list[int], truth: np.ndarray) -> dict:
    spatial = candidates[:, :3]
    unique, inverse = np.unique(spatial, axis=0, return_inverse=True)
    minimum = np.full(len(unique), np.inf)
    np.minimum.at(minimum, inverse, losses)
    ordering = np.argsort(minimum, kind="stable")
    near = int(np.argmin(np.linalg.norm(unique - truth[:3], axis=1)))
    return {"spatial_sites": len(unique),
            "nearest_site_m": unique[near].tolist(),
            "nearest_site_distance_to_truth_m": float(np.linalg.norm(unique[near]-truth[:3])),
            "nearest_site_spatial_rank": int(np.where(ordering == near)[0][0] + 1),
            "nearest_site_score": float(minimum[near]),
            "best_site_score": float(minimum[ordering[0]]),
            "selected_start_distances_to_truth_m": [
                float(np.linalg.norm(spatial[index] - truth[:3])) for index in starts],
            "minimum_selected_start_distance_to_truth_m": float(min(
                np.linalg.norm(spatial[index] - truth[:3]) for index in starts))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    output = ROOT / protocol["output"]
    if output.exists():
        raise FileExistsError(output)
    comparison, fmm, models = verify(protocol)
    print("Verified 8 training-disjoint FD records and both frozen checkpoints", flush=True)
    if args.preflight:
        return
    bounds = np.asarray(protocol["bounds_m_s"], dtype=np.float32)
    lo, hi, step = protocol["origin_time_grid_s"]
    origins = np.linspace(lo, hi, round((hi-lo)/step)+1, dtype=np.float32)
    if len(origins) != 46 or len(protocol["expanded_offsets_m"]) != 7:
        raise ValueError("Unexpected registered grid")
    refine_settings = {
        **json.loads((ROOT / "experiments/cross_region_fmm_protocol.json").read_text())["pinn_inversion"],
        "refine_steps": protocol["refine_steps"],
        "refine_adam_learning_rate_on_box_fractions":
            protocol["refine_adam_learning_rate_on_box_fractions"],
    }
    results = []
    for observation, arrival in zip(comparison, fmm):
        path = Path(observation["observation"])
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        if data.shape != (25, 1001):
            raise ValueError(f"Unexpected FD dimensions: {path}")
        rms = float(np.sqrt(np.mean(data.astype(np.float64)**2)))
        center = np.asarray(arrival["estimated_source_m_s"][:3], dtype=np.float32)
        original = proposals(center, bounds, protocol["original_offsets_m"], origins)
        expanded = proposals(center, bounds, protocol["expanded_offsets_m"], origins)
        if not len(original) == 1242 or len(expanded) > 7**3 * len(origins):
            raise ValueError("Unexpected candidate count")
        record = {"event": observation["event"], "sha256": observation["sha256"],
                  "fmm_center_m": center.tolist(), "models": {}}
        for label, (model, scale) in models.items():
            device = next(model.parameters()).device
            started = time.perf_counter()
            old_scores = fine_scores(model, receivers, times, data, scale, rms, original,
                                     protocol["receiver_stride"], protocol["time_stride"],
                                     protocol["candidate_batch"])
            old_indices = choose_spatial_starts(original, old_scores, 12, 200)
            if device.type == "cuda":
                torch.cuda.synchronize()
            old_seconds = time.perf_counter() - started
            started = time.perf_counter()
            new_scores = fine_scores(model, receivers, times, data, scale, rms, expanded,
                                     protocol["receiver_stride"], protocol["time_stride"],
                                     protocol["candidate_batch"])
            indices = choose_spatial_starts(expanded, new_scores,
                      protocol["expanded_start_count"], protocol["minimum_start_separation_m"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            scoring_seconds = time.perf_counter() - started
            starts = [{"source_m_s": expanded[index].tolist(),
                       "coarse_normalized_mse": float(new_scores[index])} for index in indices]
            started = time.perf_counter()
            winner, attempts, _, _, _ = refine(model, receivers, times,
                            torch.as_tensor(data, device=device), scale, bounds,
                            refine_settings, rms, starts)
            if device.type == "cuda":
                torch.cuda.synchronize()
            refinement_seconds = time.perf_counter() - started
            # Truth is used only for post-search rank and error diagnostics.
            with np.load(path) as archive:
                truth = np.asarray(archive["source_truth"], dtype=np.float32)
            if not np.allclose(truth, observation["source_truth_m_s"], rtol=0, atol=1e-5):
                raise ValueError(f"Archived FD source truth changed: {path}")
            predicted = np.asarray(winner["estimated_source_m_s"])
            record["models"][label] = {
                "original_candidates": len(original),
                "expanded_candidates": len(expanded),
                "original_scoring_seconds": old_seconds,
                "expanded_scoring_seconds": scoring_seconds,
                "expanded_refinement_seconds": refinement_seconds,
                "original_grid_diagnostic": rank_audit(original, old_scores, old_indices, truth),
                "expanded_grid_diagnostic": rank_audit(expanded, new_scores, indices, truth),
                "expanded_starts": starts,
                "refinement_losses": [float(item["normalized_mse"]) for item in attempts],
                "estimated_source_m_s": predicted.tolist(),
                "location_error_m": float(np.linalg.norm(predicted[:3]-truth[:3])),
                "origin_time_error_s": float(abs(predicted[3]-truth[3])),
                "selected_mse": float(winner["normalized_mse"]),
                "hit_source_bounds": bool(winner["on_region_boundary"]),
            }
            print(f"{path.name} {label}: original nearest rank "
                  f"{record['models'][label]['original_grid_diagnostic']['nearest_site_spatial_rank']}, "
                  f"expanded error={record['models'][label]['location_error_m']:.1f} m, "
                  f"cost={scoring_seconds+refinement_seconds:.1f} s", flush=True)
        results.append(record)
    summary = {label: {"median_location_m": float(np.median([
                       row["models"][label]["location_error_m"] for row in results])),
                       "max_location_m": float(max(
                       row["models"][label]["location_error_m"] for row in results)),
                       "mean_search_seconds": float(np.mean([
                       row["models"][label]["expanded_scoring_seconds"]+
                       row["models"][label]["expanded_refinement_seconds"]
                       for row in results]))}
               for label in models}
    output.write_text(json.dumps({"role": protocol["role"],
                      "protocol_sha256": hash_file(PROTOCOL),
                      "checkpoint_sha256": {key: protocol[f"{key}_sha256"] for key in models},
                      "summary": summary, "events": results,
                      "limitations": protocol["interpretation"]}, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
