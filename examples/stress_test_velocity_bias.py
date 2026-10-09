"""Stress-test frozen localization under uniform inversion-velocity bias."""

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

from examples.confirm_traveltime_aligned_moe import (
    fmm_baseline,
    verify_frozen_inputs,
)
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores,
    choose_diverse_starts,
    coarse_grid,
    load_event,
    load_model,
    pick_arrivals,
    refine,
    sampled_waveforms,
    sha256,
    waveform_scores,
)
from examples.stress_test_traveltime_aligned_moe import branch_result

PROTOCOL = ROOT / "experiments/traveltime_aligned_moe_velocity_bias_protocol.json"


def scale_traveltime_fields(fields: np.ndarray,
                            velocity_factor: float) -> np.ndarray:
    """Apply the exact eikonal scaling for a uniform velocity multiplier."""
    if not np.isfinite(velocity_factor) or velocity_factor <= 0:
        raise ValueError("velocity_factor must be positive and finite")
    scaled = np.empty(fields.shape, dtype=np.float32)
    np.divide(fields, np.float32(velocity_factor), out=scaled,
              casting="unsafe")
    return scaled


def install_fields(model, fields: np.ndarray) -> None:
    values = torch.from_numpy(fields).to(
        device=model.interpolator.fields_s.device,
        dtype=model.interpolator.fields_s.dtype)
    if values.shape != model.interpolator.fields_s.shape:
        raise ValueError("Perturbed fields do not match the frozen model")
    model.interpolator.fields_s.copy_(values)


def run_condition(spec: dict, localization: dict, paths: list[Path], model,
                  domain: np.ndarray, base_fields: np.ndarray,
                  condition: dict, device: torch.device) -> list[dict]:
    factor = float(condition["velocity_factor"])
    fields = scale_traveltime_fields(base_fields, factor)
    install_fields(model, fields)
    bounds = np.asarray(localization["source_bounds_m_s"], dtype=np.float32)
    search = localization["search"]
    arrival_spec = localization["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    for path in paths:
        event = load_event(path, model.receiver_xyz_m.cpu().numpy())
        raw_picks, corrected = pick_arrivals(event, arrival_spec)
        valid_count = int(np.isfinite(corrected).sum())
        if valid_count < int(arrival_spec["minimum_valid_picks"]):
            raise ValueError(f"Clean event has only {valid_count} picks: {path}")
        coarse_sampled = sampled_waveforms(
            event, domain, int(search["coarse_receiver_stride"]),
            int(search["coarse_time_stride"]), device)
        refined_sampled = sampled_waveforms(
            event, domain, int(search["refine_receiver_stride"]),
            int(search["refine_time_stride"]), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        waveform = waveform_scores(model, candidates, domain, coarse_sampled,
                                   int(search["candidate_batch"]))
        arrival = arrival_scores(
            model, candidates, corrected,
            float(arrival_spec["arrival_scale_s"]),
            int(search["candidate_batch"]))
        branches = {
            "waveform_only": branch_result(
                model, candidates, waveform, arrival, 0.0, domain, bounds,
                refined_sampled, corrected, search, arrival_spec),
            "hybrid": branch_result(
                model, candidates, waveform, arrival,
                float(arrival_spec["hybrid_weight"]), domain, bounds,
                refined_sampled, corrected, search, arrival_spec),
        }
        pure_fmm = fmm_baseline(fields, corrected, bounds, spec)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        for branch in (*branches.values(), pure_fmm):
            estimate = np.asarray(branch["estimated_source_m_s"], dtype=np.float64)
            error = estimate - truth
            branch["component_error_m_s"] = error.tolist()
            branch["location_error_m"] = float(np.linalg.norm(error[:3]))
            branch["origin_time_error_s"] = float(abs(error[3]))
        row = {
            "condition_id": condition["id"],
            "velocity_factor": factor,
            "velocity_bias_percent": 100.0 * (factor - 1.0),
            "event": path.name,
            "observation_sha256": sha256(path),
            "valid_pick_count": valid_count,
            "source_truth_m_s_post_search": truth.tolist(),
            "search_seconds": elapsed,
            **branches,
            "fmm": pure_fmm,
        }
        rows.append(row)
        print(f"{condition['id']} {path.name}: "
              f"hybrid={branches['hybrid']['location_error_m']:.1f} m, "
              f"FMM={pure_fmm['location_error_m']:.1f} m, "
              f"et={branches['hybrid']['origin_time_error_s']:.4f} s, "
              f"time={elapsed:.1f} s", flush=True)
    return rows


def summarize(rows: list[dict], conditions: list[dict]) -> dict:
    output = {}
    for condition in conditions:
        subset = [row for row in rows if row["condition_id"] == condition["id"]]
        entry = {"trial_count": len(subset),
                 "velocity_factor": condition["velocity_factor"]}
        for branch in ("waveform_only", "hybrid", "fmm"):
            locations = np.asarray(
                [row[branch]["location_error_m"] for row in subset])
            origins = np.asarray(
                [row[branch]["origin_time_error_s"] for row in subset])
            entry[branch] = {
                "median_location_error_m": float(np.median(locations)),
                "mean_location_error_m": float(np.mean(locations)),
                "median_origin_time_error_s": float(np.median(origins)),
                "mean_origin_time_error_s": float(np.mean(origins)),
                "events_below_500_m": int(np.sum(locations < 500.0)),
            }
        entry["hybrid_improved_over_waveform_events"] = int(sum(
            row["hybrid"]["location_error_m"]
            < row["waveform_only"]["location_error_m"] for row in subset))
        output[condition["id"]] = entry
    return output


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# Frozen localization under uniform velocity bias", "",
        "FD observations use the original Overthrust velocity. Both neural travel-time features and the FMM baseline use the biased inversion velocity.", "",
        "| Inversion velocity | Trials | Waveform median (m) | Hybrid median (m) | Hybrid median time (s) | FMM median (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in result["conditions"]:
        row = result["summary"][condition["id"]]
        lines.append(
            f"| {100 * condition['velocity_factor']:.0f}% of truth | "
            f"{row['trial_count']} | "
            f"{row['waveform_only']['median_location_error_m']:.1f} | "
            f"{row['hybrid']['median_location_error_m']:.1f} | "
            f"{row['hybrid']['median_origin_time_error_s']:.4f} | "
            f"{row['fmm']['median_location_error_m']:.1f} |")
    lines.extend(["", f"Runtime: {result['runtime_seconds']:.1f} s.", "",
                  f"Limitations: {result['limitations']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != spec["implementation_sha256"]:
        raise ValueError("Velocity-bias implementation changed after registration")
    localization = verify_frozen_inputs(spec)
    clean_result = ROOT / spec["clean_confirmation_result"]
    if sha256(clean_result) != spec["clean_confirmation_result_sha256"]:
        raise ValueError("Clean confirmation result changed")
    paths = [ROOT / spec["confirmation_directory"] / row["file"]
             for row in spec["events"]]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("A confirmation observation is missing")
    output = ROOT / spec["output_result"]
    report = ROOT / spec["output_report"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite velocity-bias output")
    if args.preflight:
        print("Preflight passed: frozen assets and velocity-bias protocol match")
        return

    first = load_event(paths[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain = load_model(localization, first["receivers"], device)
    base_fields = np.load(ROOT / spec["receiver_fmm_fields"], mmap_mode="r")
    started = time.perf_counter()
    rows = []
    for condition in spec["conditions"]:
        rows.extend(run_condition(spec, localization, paths, model, domain,
                                  base_fields, condition, device))
    runtime = time.perf_counter() - started
    result = {
        "purpose": spec["purpose"],
        "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "implementation_sha256": spec["implementation_sha256"],
        "checkpoint_sha256": spec["model_checkpoint_sha256"],
        "device": str(device),
        "conditions": spec["conditions"],
        "trial_count": len(rows),
        "runtime_seconds": runtime,
        "summary": summarize(rows, spec["conditions"]),
        "records": rows,
        "limitations": spec["limitations"],
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report, result)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved {output}; runtime={runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
