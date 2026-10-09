"""Generate and evaluate an independent-source adaptive-fusion confirmation.

Run with --generate first and --evaluate second. Evaluation is resumable from
the partial JSON file and never uses source truth inside a search objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments/adaptive_fusion_independent_protocol.json"
sys.path.insert(0, str(ROOT))

from examples.cross_model_geology_audit import fmm_locate  # noqa: E402
from examples.invert_fmm_receiver_fields import score_candidates  # noqa: E402
from examples.run_low_rank_pick_perturbation_audit import (  # noqa: E402
    corrected_picks,
    crop_to_model_window,
    run_branch,
)
from examples.run_source_conditioned_low_rank_cross_geology import (  # noqa: E402
    load_model_with_fields,
)
from examples.run_traveltime_aligned_moe_localization import (  # noqa: E402
    arrival_scores,
    coarse_grid,
    load_event,
    sampled_waveforms,
    waveform_scores,
)
from examples.invert_fmm_receiver_fields import spatial_grid  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def verify_frozen_inputs(spec: dict) -> None:
    for key in ("checkpoint", "velocity_file", "receivers_file", "fmm_cache"):
        path = ROOT / spec[key]
        if not path.is_file() or sha256(path) != spec[f"{key.split('_file')[0]}_sha256"]:
            if key == "checkpoint": expected_key = "checkpoint_sha256"
            elif key == "velocity_file": expected_key = "velocity_sha256"
            elif key == "receivers_file": expected_key = "receivers_sha256"
            else: expected_key = "fmm_cache_sha256"
            if not path.is_file() or sha256(path) != spec[expected_key]:
                raise ValueError(f"Frozen input hash mismatch: {path}")


def fd_config(spec: dict) -> dict:
    source_bounds = np.asarray(spec["source_bounds_m_s"], dtype=np.float64)
    return {
        "status": "registered_adaptive_fusion_independent_fd_configuration",
        "velocity_file": str((ROOT / spec["velocity_file"]).resolve()),
        "velocity_shape": [401, 401, 94],
        "grid_spacing_m": [40.0, 40.0, 40.0],
        "domain_extent_m": [16000.0, 16000.0, 3720.0],
        "receivers_file": str((ROOT / spec["receivers_file"]).resolve()),
        "receiver_count": 25,
        "source_location_bounds_m": source_bounds[:3].T.tolist(),
        "origin_time_bounds_s": source_bounds[3].tolist(),
        "time_step_s": float(spec["fd"]["time_step_s"]),
        "record_length_s": float(spec["fd"]["duration_s"]),
        "boundary_condition": "zero Dirichlet on all six faces",
        "frequency_validity": "5 Hz; minimum wavelength exceeds ten grid cells"
    }


def make_noisy_archive(clean_path: Path, output_path: Path,
                       snr_db: int, seed: int) -> dict:
    if output_path.exists():
        with np.load(output_path) as saved:
            if (int(saved["noise_seed"]) != seed
                    or int(saved["noise_snr_db"]) != snr_db):
                raise FileExistsError(f"Refusing to replace mismatched noise data: {output_path}")
        return {"file": str(output_path.relative_to(ROOT)),
                "sha256": sha256(output_path), "snr_db": snr_db, "seed": seed}

    with np.load(clean_path) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        truth = np.asarray(archive["source_truth"], dtype=np.float32)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError(f"Invalid clean FD trace: {clean_path}")
    generator = np.random.default_rng(seed)
    signal_rms = np.sqrt(np.mean(data.astype(np.float64) ** 2, axis=1))
    noise_std = signal_rms / (10.0 ** (snr_db / 20.0))
    noise = generator.standard_normal(data.shape).astype(np.float32)
    noisy = data + noise * noise_std.astype(np.float32)[:, None]
    measured_snr = 20.0 * np.log10(
        np.maximum(signal_rms, 1e-20)
        / np.maximum(np.sqrt(np.mean((noisy - data).astype(np.float64) ** 2, axis=1)), 1e-20)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, data=noisy.astype(np.float32), times=times,
        receiver_coords=receivers, source_truth=truth,
        noise_snr_db=np.int32(snr_db), noise_seed=np.int64(seed),
    )
    return {"file": str(output_path.relative_to(ROOT)),
            "sha256": sha256(output_path), "snr_db": snr_db, "seed": seed,
            "median_realized_trace_snr_db": float(np.median(measured_snr))}


def generate_data(spec: dict) -> None:
    verify_frozen_inputs(spec)
    folder = ROOT / spec["fd"]["observation_directory"]
    folder.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / spec["fd"]["config_file"]
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config != fd_config(spec):
            raise FileExistsError(f"Existing FD configuration differs: {config_path}")
    else:
        config_path.write_text(json.dumps(fd_config(spec), indent=2) + "\n",
                               encoding="utf-8")

    clean_records = []
    generator = ROOT / spec["fd"]["generator"]
    for event in spec["events"]:
        output = folder / event["file"]
        truth = np.asarray(event["source_m_s"], dtype=np.float32)
        if output.exists():
            with np.load(output) as archive:
                if not np.allclose(archive["source_truth"], truth, atol=1e-5):
                    raise FileExistsError(f"Existing record has different truth: {output}")
        else:
            subprocess.run(
                [sys.executable, str(generator), "--config", str(config_path),
                 "--output", str(output), "--source",
                 *(str(float(value)) for value in truth)],
                cwd=ROOT, check=True,
            )
        with np.load(output) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.allclose(archive["source_truth"], truth, atol=1e-5)):
                raise ValueError(f"Generated observation failed validation: {output}")
        clean_records.append({"event": event["name"],
                              "file": str(output.relative_to(ROOT)),
                              "sha256": sha256(output),
                              "source_m_s": truth.tolist()})
        print(f"clean record ready: {event['name']}", flush=True)

    noise_records = []
    for event, clean in zip(spec["events"], clean_records):
        clean_path = ROOT / clean["file"]
        event_index = spec["events"].index(event)
        for snr_db in spec["noise"]["snr_db"]:
            for realization, base_seed in enumerate(spec["noise"]["realization_seeds"], start=1):
                seed = int(base_seed + event_index * 1009)
                filename = f"{Path(event['file']).stem}_snr{int(snr_db):02d}_r{realization}.npz"
                noise_records.append(make_noisy_archive(
                    clean_path, folder / filename, int(snr_db), seed))
    manifest = {
        "protocol": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "fd_config_sha256": sha256(config_path),
        "clean_records": clean_records,
        "noise_records": noise_records,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = ROOT / spec["fd"]["generation_manifest"]
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in previous.get("clean_records", []) + previous.get("noise_records", []):
            path = ROOT / row["file"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError(f"Existing generation manifest no longer matches {path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
    print(f"Generated/verified {len(clean_records)} clean and {len(noise_records)} noisy records.",
          flush=True)


def fmm_estimate(fields: np.ndarray, picks: np.ndarray, bounds: np.ndarray,
                 spacing_m: float, search: dict,
                 arrival: dict) -> dict:
    valid = np.isfinite(picks)
    if int(valid.sum()) < int(arrival["minimum_valid_picks_for_fmm_identifiability"]):
        return {"status": "unavailable_insufficient_picks",
                "valid_picks": int(valid.sum()), "estimated_source_m_s": None}
    candidates = spatial_grid(bounds[:3], tuple(search["coarse_grid_xyz"]))
    losses, origins = score_candidates(
        fields, candidates, picks, spacing_m,
        tuple(arrival["origin_time_bounds_s"]))
    index = int(np.argmin(losses))
    best = (float(losses[index]), candidates[index].copy(), float(origins[index]))
    steps = (bounds[:3, 1] - bounds[:3, 0]) / (np.asarray(search["coarse_grid_xyz"]) - 1)
    count = len(candidates)
    for _ in range(int(search["refinement_levels"])):
        low = np.maximum(best[1] - steps, bounds[:3, 0])
        high = np.minimum(best[1] + steps, bounds[:3, 1])
        local = spatial_grid(np.stack((low, high), axis=1),
                             (int(search["refinement_grid_xyz"]),) * 3)
        local_losses, local_origins = score_candidates(
            fields, local, picks, spacing_m,
            tuple(arrival["origin_time_bounds_s"]))
        local_index = int(np.argmin(local_losses))
        if local_losses[local_index] < best[0]:
            best = (float(local_losses[local_index]), local[local_index].copy(),
                    float(local_origins[local_index]))
        steps *= 2.0 / (int(search["refinement_grid_xyz"]) - 1)
        count += len(local)
    return {"status": "estimate_available", "valid_picks": int(valid.sum()),
            "estimated_source_m_s": np.r_[best[1], best[2]].tolist(),
            "arrival_rmse_s": float(np.sqrt(best[0])),
            "candidates_evaluated": int(count), "source_truth_used_in_search": False}


def adaptive_weight(valid_picks: int, arrival: dict) -> tuple[float | None, str]:
    threshold = int(arrival["minimum_valid_picks_for_hybrid"])
    if valid_picks < threshold:
        return None, "fmm_fallback"
    ramp = int(arrival["adaptive_ramp_picks"])
    fraction = float(np.clip((valid_picks - threshold) / ramp, 0.0, 1.0))
    alpha = (float(arrival["adaptive_min_hybrid_weight"])
             + (float(arrival["adaptive_max_hybrid_weight"])
                - float(arrival["adaptive_min_hybrid_weight"])) * fraction)
    return alpha, "adaptive_hybrid"


def branch_metrics(branch: dict, truth: np.ndarray,
                   threshold_m: float) -> dict:
    estimate = branch.get("estimated_source_m_s")
    if estimate is None:
        return {**branch, "location_error_m": None,
                "origin_time_error_s": None, "within_success_threshold": False}
    delta = np.asarray(estimate, dtype=np.float64) - truth
    eloc = float(np.linalg.norm(delta[:3]))
    return {**branch, "component_error_m_s": delta.tolist(),
            "location_error_m": eloc,
            "origin_time_error_s": float(abs(delta[3])),
            "within_success_threshold": bool(eloc <= threshold_m)}


def evaluate_condition(model, fields: np.ndarray, event_path: Path,
                       truth_path: Path, spec: dict,
                       candidates: np.ndarray, domain: np.ndarray,
                       bounds: np.ndarray, device: torch.device) -> dict:
    search, arrival = spec["search"], spec["arrival_constraint"]
    event = crop_to_model_window(load_event(event_path), float(domain[3]))
    picks = corrected_picks(event, arrival)
    valid_count = int(np.isfinite(picks).sum())
    coarse = sampled_waveforms(event, domain,
                               int(search["coarse_receiver_stride"]),
                               int(search["coarse_time_stride"]), device)
    fine = sampled_waveforms(event, domain,
                             int(search["refine_receiver_stride"]),
                             int(search["refine_time_stride"]), device)
    started = time.perf_counter()
    waveform = waveform_scores(model, candidates, domain, coarse,
                               int(search["candidate_batch"]))
    waveform_seconds = time.perf_counter() - started
    branches: dict[str, dict] = {}
    branch_seconds: dict[str, float] = {}

    started = time.perf_counter()
    best = run_branch(model, candidates, waveform, None, 0.0,
                      domain, bounds, fine, picks, search, arrival)
    branches["waveform_only"] = {"status": "estimate_available",
                                 "estimated_source_m_s": best["estimated_source_m_s"]}
    branch_seconds["waveform_only"] = time.perf_counter() - started

    enough_hybrid_picks = valid_count >= int(arrival["minimum_valid_picks_for_hybrid"])
    travel = None
    travel_seconds = 0.0
    if enough_hybrid_picks:
        started = time.perf_counter()
        travel = arrival_scores(model, candidates, picks,
                                float(arrival["arrival_scale_s"]),
                                int(search["candidate_batch"]))
        travel_seconds = time.perf_counter() - started
        started = time.perf_counter()
        best = run_branch(model, candidates, waveform, travel,
                          float(arrival["fixed_hybrid_weight"]), domain,
                          bounds, fine, picks, search, arrival)
        branches["fixed_hybrid"] = {"status": "estimate_available",
                                    "estimated_source_m_s": best["estimated_source_m_s"],
                                    "alpha": float(arrival["fixed_hybrid_weight"])}
        branch_seconds["fixed_hybrid"] = time.perf_counter() - started
        adaptive_alpha_value, decision = adaptive_weight(valid_count, arrival)
        if np.isclose(adaptive_alpha_value, float(arrival["fixed_hybrid_weight"])):
            branches["adaptive"] = {**branches["fixed_hybrid"],
                                    "decision": decision,
                                    "reused_fixed_hybrid": True}
            branch_seconds["adaptive"] = 0.0
        else:
            started = time.perf_counter()
            best = run_branch(model, candidates, waveform, travel,
                              float(adaptive_alpha_value), domain, bounds,
                              fine, picks, search, arrival)
            branches["adaptive"] = {"status": "estimate_available",
                                    "estimated_source_m_s": best["estimated_source_m_s"],
                                    "alpha": float(adaptive_alpha_value),
                                    "decision": decision,
                                    "reused_fixed_hybrid": False}
            branch_seconds["adaptive"] = time.perf_counter() - started
    else:
        branches["fixed_hybrid"] = {
            "status": "unavailable_insufficient_picks",
            "valid_picks": valid_count,
            "minimum_valid_picks": int(arrival["minimum_valid_picks_for_hybrid"]),
            "estimated_source_m_s": None,
        }
        branch_seconds["fixed_hybrid"] = 0.0
        branches["adaptive"], branch_seconds["adaptive"] = {}, 0.0

    started = time.perf_counter()
    branches["pure_fmm"] = fmm_estimate(
        fields, picks, bounds, float(spec["fmm_spacing_m"]), search, arrival)
    branch_seconds["pure_fmm"] = time.perf_counter() - started
    if not enough_hybrid_picks:
        if branches["pure_fmm"]["estimated_source_m_s"] is not None:
            branches["adaptive"] = {**branches["pure_fmm"],
                                     "decision": "fmm_fallback",
                                     "alpha": None,
                                     "reused_fixed_hybrid": False}
            branch_seconds["adaptive"] = branch_seconds["pure_fmm"]
        else:
            branches["adaptive"] = {**branches["pure_fmm"],
                                     "decision": "fmm_fallback",
                                     "alpha": None,
                                     "reused_fixed_hybrid": False}
            branch_seconds["adaptive"] = branch_seconds["pure_fmm"]

    # Source truth is intentionally loaded only after every estimate is fixed.
    with np.load(truth_path) as archive:
        truth = np.asarray(archive["source_truth"], dtype=np.float64)
    threshold = float(spec["evaluation"]["location_success_threshold_m"])
    branches = {name: branch_metrics(branch, truth, threshold)
                for name, branch in branches.items()}
    with np.load(event_path) as archive:
        snr_db = int(archive["noise_snr_db"]) if "noise_snr_db" in archive.files else None
        seed = int(archive["noise_seed"]) if "noise_seed" in archive.files else None
    return {
        "condition": event_path.stem,
        "event_file": str(event_path.relative_to(ROOT)),
        "event_sha256": sha256(event_path),
        "snr_db": snr_db,
        "noise_seed": seed,
        "valid_picks": valid_count,
        "adaptive_decision": branches["adaptive"].get("decision", "adaptive_hybrid"),
        "adaptive_alpha": branches["adaptive"].get("alpha"),
        "source_truth_m_s_post_search": truth.tolist(),
        "waveform_candidate_scoring_seconds": waveform_seconds,
        "arrival_candidate_scoring_seconds": travel_seconds,
        "branch_seconds": branch_seconds,
        "branches": branches,
        "source_truth_used_in_search": False,
    }


def summarize(rows: list[dict], spec: dict) -> dict:
    groups: dict[str, list[dict]] = {"clean": []}
    for snr in spec["noise"]["snr_db"]:
        groups[str(snr)] = []
    for row in rows:
        key = "clean" if row["snr_db"] is None else str(row["snr_db"])
        groups.setdefault(key, []).append(row)
    result = {}
    names = ("waveform_only", "fixed_hybrid", "adaptive", "pure_fmm")
    for group, selected in groups.items():
        methods = {}
        for name in names:
            estimates = [r["branches"][name] for r in selected
                         if r["branches"][name]["location_error_m"] is not None]
            errors = np.asarray([x["location_error_m"] for x in estimates], dtype=np.float64)
            times = np.asarray([x["origin_time_error_s"] for x in estimates], dtype=np.float64)
            methods[name] = {
                "n_conditions": len(selected),
                "n_estimates": len(estimates),
                "n_unavailable": len(selected) - len(estimates),
                "n_within_500m": int(np.sum(errors <= float(spec["evaluation"]["location_success_threshold_m"]))),
                "success_rate_all_conditions": float(np.sum(errors <= float(spec["evaluation"]["location_success_threshold_m"])) / len(selected)) if selected else None,
                "median_eloc_m": float(np.median(errors)) if len(errors) else None,
                "mean_eloc_m": float(np.mean(errors)) if len(errors) else None,
                "median_et_s": float(np.median(times)) if len(times) else None,
            }
        paired = {}
        for comparator in ("fixed_hybrid", "pure_fmm", "waveform_only"):
            pairs = [(r["branches"]["adaptive"]["location_error_m"],
                      r["branches"][comparator]["location_error_m"])
                     for r in selected
                     if r["branches"]["adaptive"]["location_error_m"] is not None
                     and r["branches"][comparator]["location_error_m"] is not None]
            delta = np.asarray([a - b for a, b in pairs], dtype=np.float64)
            paired[f"adaptive_minus_{comparator}_eloc_m"] = {
                "n_paired": len(pairs),
                "adaptive_wins": int(np.sum(delta < 0)),
                "ties": int(np.sum(delta == 0)),
                "adaptive_losses": int(np.sum(delta > 0)),
                "median_delta_m": float(np.median(delta)) if len(delta) else None,
            }
        result[group] = {"conditions": len(selected), "methods": methods,
                         "paired_comparisons": paired,
                         "adaptive_fallback_count": sum(
                             r["adaptive_decision"] == "fmm_fallback" for r in selected),
                         "valid_picks": [r["valid_picks"] for r in selected]}
    return result


def write_report(path: Path, result: dict, spec: dict) -> None:
    lines = ["# Independent adaptive-fusion confirmation", "",
             spec["limitations"], "",
             "Success means a finite estimate with location error <= 500 m. Unavailable estimates count as failures in the all-condition success rate.", "",
             "| SNR | Method | Available | <=500 m | Success rate | Median eloc (m) | Median et (s) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group, block in result["summary"].items():
        for method, stats in block["methods"].items():
            lines.append(
                f"| {group} | {method} | {stats['n_estimates']}/{stats['n_conditions']} | "
                f"{stats['n_within_500m']}/{stats['n_conditions']} | "
                f"{stats['success_rate_all_conditions']:.3f} | "
                f"{stats['median_eloc_m'] if stats['median_eloc_m'] is not None else float('nan'):.1f} | "
                f"{stats['median_et_s'] if stats['median_et_s'] is not None else float('nan'):.4f} |"
            )
    lines.extend(["", "Adaptive-minus-comparator paired location-error changes are in metres; negative values favor adaptive fusion.", "",
                  f"Overall conditions evaluated: **{len(result['events'])}**. ",
                  f"Runtime total: **{result['runtime_seconds']:.1f} s**.", "",
                  "The experiment is synthetic, uses one Overthrust model and one surface array, and cannot establish field-data or cross-geology performance."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_all(spec: dict) -> None:
    verify_frozen_inputs(spec)
    manifest_path = ROOT / spec["fd"]["generation_manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError("Run --generate before --evaluate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise ValueError("Generation manifest belongs to a different protocol")
    for row in manifest["clean_records"] + manifest["noise_records"]:
        path = ROOT / row["file"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Observation hash mismatch: {path}")

    output = ROOT / spec["evaluation"]["output_result"]
    report = ROOT / spec["evaluation"]["output_report"]
    partial = ROOT / spec["evaluation"]["partial_results_file"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite independent confirmation results")
    protocol_hash = sha256(PROTOCOL_PATH)
    completed: dict[str, dict] = {}
    if partial.exists():
        saved = json.loads(partial.read_text(encoding="utf-8"))
        if saved.get("protocol_sha256") != protocol_hash:
            raise ValueError("Partial results belong to a different protocol")
        completed = {row["condition"]: row for row in saved["events"]}

    cache = np.load(ROOT / spec["fmm_cache"], mmap_mode="r")
    if cache.shape != (25, 201, 201, 47) or not np.isfinite(cache).all():
        raise ValueError("FMM cache shape/content mismatch")
    receiver_path = ROOT / spec["receivers_file"]
    receivers = np.asarray(np.load(receiver_path), dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain, _ = load_model_with_fields(
        ROOT / spec["checkpoint"], cache, receivers,
        float(spec["fmm_spacing_m"]), device)
    bounds = np.asarray(spec["source_bounds_m_s"], dtype=np.float32)
    candidates = coarse_grid(bounds, spec["search"]["coarse_grid_xyzt"])
    folder = ROOT / spec["fd"]["observation_directory"]
    conditions = []
    for event in spec["events"]:
        truth_path = folder / event["file"]
        conditions.append((event, truth_path, truth_path))
        for snr_db in spec["noise"]["snr_db"]:
            for realization in range(1, len(spec["noise"]["realization_seeds"]) + 1):
                filename = f"{Path(event['file']).stem}_snr{int(snr_db):02d}_r{realization}.npz"
                conditions.append((event, folder / filename, truth_path))

    started_all = time.perf_counter()
    for event, observation_path, truth_path in conditions:
        key = observation_path.stem
        if key in completed:
            continue
        row = evaluate_condition(model, cache, observation_path, truth_path,
                                 spec, candidates, domain, bounds, device)
        row["event"] = event["name"]
        completed[key] = row
        partial_payload = {
            "purpose": spec["purpose"],
            "protocol": str(PROTOCOL_PATH.resolve()),
            "protocol_sha256": protocol_hash,
            "generation_manifest_sha256": sha256(manifest_path),
            "checkpoint_sha256": sha256(ROOT / spec["checkpoint"]),
            "velocity_sha256": sha256(ROOT / spec["velocity_file"]),
            "device": str(device),
            "events": list(completed.values()),
            "summary": summarize(list(completed.values()), spec),
        }
        partial.write_text(json.dumps(partial_payload, indent=2) + "\n",
                           encoding="utf-8")
        adaptive = row["branches"]["adaptive"]
        error = adaptive["location_error_m"]
        print(f"{key}: picks={row['valid_picks']} decision={row['adaptive_decision']} "
              f"adaptive_eloc={'NA' if error is None else f'{error:.1f}m'}",
              flush=True)

    rows = list(completed.values())
    result = {
        "purpose": spec["purpose"],
        "protocol": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": protocol_hash,
        "generation_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": sha256(ROOT / spec["checkpoint"]),
        "velocity_sha256": sha256(ROOT / spec["velocity_file"]),
        "receiver_sha256": sha256(ROOT / spec["receivers_file"]),
        "fmm_cache_sha256": sha256(ROOT / spec["fmm_cache"]),
        "device": str(device),
        "runtime_seconds": time.perf_counter() - started_all,
        "events": rows,
        "summary": summarize(rows, spec),
        "limitations": spec["limitations"],
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(report, result, spec)
    partial.unlink(missing_ok=True)
    print(f"Saved {output} and {report}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--generate", action="store_true",
                         help="Generate clean FD observations and fixed noise realizations")
    actions.add_argument("--evaluate", action="store_true",
                         help="Run the frozen four-branch localization comparison")
    args = parser.parse_args()
    spec = read_protocol()
    if not spec["protocol_written_before_data_generation"]:
        raise ValueError("Protocol must be registered before generating observations")
    if args.generate:
        generate_data(spec)
    else:
        evaluate_all(spec)


if __name__ == "__main__":
    main()
