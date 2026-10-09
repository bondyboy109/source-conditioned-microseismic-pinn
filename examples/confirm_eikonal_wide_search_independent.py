"""Preregistered, one-time independent FMM and wide-search PINN evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.confirm_depth_gap_25 import refine
from examples.invert_fmm_receiver_fields import InsufficientPicks, invert_one, receiver_fields
from examples.run_fmm_seeded_waveform_development import (
    choose_spatial_starts, fine_scores, hash_file, proposals,
)
from source_conditioned_pinn import conditioned_prediction, load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_wide_search_independent_protocol.json"


def preflight(protocol: dict, metadata: dict) -> tuple[dict, dict, list[Path]]:
    required = {
        protocol["checkpoint"]: protocol["checkpoint_sha256"],
        protocol["velocity"]: protocol["velocity_sha256"],
        protocol["development_search"]: protocol["development_search_sha256"],
        protocol["fmm_settings"]: protocol["fmm_settings_sha256"],
        "examples/generate_fd_observation.py": protocol["fd_generator_sha256"],
        **protocol["search_implementation_sha256"],
    }
    for name, expected in required.items():
        if hash_file(ROOT / name) != expected:
            raise ValueError(f"Input changed since registration: {name}")
    search = json.loads((ROOT / protocol["development_search"]).read_text(encoding="utf-8"))
    fmm = json.loads((ROOT / protocol["fmm_settings"]).read_text(encoding="utf-8"))
    if (search["old_sha256"] != protocol["checkpoint_sha256"]
            or search["bounds_m_s"] != fmm["shared_source_bounds_m_s"]
            or search["receiver_stride"] != 1 or search["time_stride"] != 4
            or len(metadata["training_record"]["receiver_observations"]) != 29
            or metadata["frequency_hz"] != 5):
        raise ValueError("Frozen model and search settings differ from registration")
    training = metadata["training_record"]["receiver_observations"]
    prior = [("training manifest", np.asarray(row["source_truth"][:3])) for row in training]
    data_dir = ROOT / protocol["data_directory"]
    for path in (ROOT / "data/observations").rglob("*.npz"):
        if data_dir in path.parents:
            raise FileExistsError(f"Held-out data already exist; refusing a second run: {path}")
        with np.load(path) as archive:
            if "source_truth" in archive.files:
                prior.append((str(path), np.asarray(archive["source_truth"][:3])))
    events = protocol["events"]
    source = np.asarray([event["source_m_s"] for event in events], dtype=np.float64)
    bounds = np.asarray(search["bounds_m_s"], dtype=np.float64)
    if (len(events) != 8 or len({event["file"] for event in events}) != 8
            or any(Path(event["file"]).name != event["file"] for event in events)
            or not np.isfinite(source).all() or source.shape != (8, 4)
            or np.any(source < bounds[:, 0]) or np.any(source > bounds[:, 1])):
        raise ValueError("Invalid or nonunique registered event files / source bounds")
    for i, point in enumerate(source):
        for path, earlier in prior:
            if np.linalg.norm(point[:3] - earlier) < protocol["minimum_distance_to_prior_sources_m"]:
                raise ValueError(f"test_{i+1:02d} overlaps an earlier event: {path}")
        if any(np.linalg.norm(point[:3] - other[:3]) < 100 for other in source[:i]):
            raise ValueError("Registered test sources overlap each other")
    outputs = [ROOT / protocol[key] for key in ("estimates_output", "results_output")]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"One-time result already exists: {outputs}")
    print(f"Preflight: {len(events)} new positions; checked {len(prior)} previous/training positions; "
          f"all registered hashes match", flush=True)
    return search, fmm, outputs


def generate(protocol: dict, training_hashes: set[str]) -> list[Path]:
    directory = ROOT / protocol["data_directory"]
    directory.mkdir(parents=True, exist_ok=False)
    paths = []
    for event in protocol["events"]:
        path = directory / event["file"]
        subprocess.run([sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                        "--duration", "2", "--output", str(path), "--source",
                        *(str(value) for value in event["source_m_s"])],
                       cwd=ROOT, check=True)
        report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.isfinite(archive["data"]).all()
                    or not np.allclose(archive["times"][-1], 2.0, atol=1e-5)
                    or archive["receiver_coords"].shape != (25, 3)):
                raise ValueError(f"Invalid FD record: {path}")
        if (report["frequency_hz"] != 5 or report["duration_s"] != 2
                or report["boundary"] != "zero Dirichlet on all six faces"
                or hash_file(path) in training_hashes):
            raise ValueError(f"FD settings or training overlap invalid: {path}")
        paths.append(path)
    return paths


def forward_l2(model, receivers, times, observed, source, scale) -> float:
    device = next(model.parameters()).device
    with torch.inference_mode():
        predicted = conditioned_prediction(
            model, torch.as_tensor(receivers / scale[:3], device=device),
            torch.as_tensor(times / scale[3], device=device),
            torch.as_tensor(source.astype(np.float32) / scale, device=device)).cpu().numpy()
    return float(np.linalg.norm(predicted.astype(np.float64) - observed) /
                 np.linalg.norm(observed))


def error(estimate: list[float] | None, truth: np.ndarray) -> dict:
    if estimate is None:
        return {"component_error_m_s": None, "location_error_m": None,
                "origin_time_error_s": None}
    delta = np.asarray(estimate, dtype=np.float64) - truth
    return {"component_error_m_s": delta.tolist(),
            "location_error_m": float(np.linalg.norm(delta[:3])),
            "origin_time_error_s": float(abs(delta[3]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Validate before generating any FD data")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(ROOT / protocol["checkpoint"], device)
    model.eval().requires_grad_(False)
    search, fmm, outputs = preflight(protocol, metadata)
    if args.preflight:
        return
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    training_hashes = {row["sha256"] for row in metadata["training_record"]["receiver_observations"]}
    paths = generate(protocol, training_hashes)
    with np.load(paths[0]) as archive:
        receiver_coords = np.asarray(archive["receiver_coords"], dtype=np.float64)
    fmm_config = fmm["fmm_baseline"]
    stride = fmm_config["velocity_grid_stride"]
    velocity = np.load(ROOT / protocol["velocity"], mmap_mode="r")
    if velocity.shape != (401, 401, 94):
        raise ValueError("Velocity geometry changed")
    started = time.perf_counter()
    fields = receiver_fields(np.ascontiguousarray(velocity[::stride, ::stride, ::stride],
                                                  dtype=np.float32), receiver_coords, 40. * stride)
    fmm_setup_seconds = time.perf_counter() - started
    bounds = np.asarray(search["bounds_m_s"], dtype=np.float32)
    origin_times = np.linspace(*search["origin_time_grid_s"][:2],
                               round((search["origin_time_grid_s"][1]
                                      - search["origin_time_grid_s"][0])
                                     / search["origin_time_grid_s"][2]) + 1,
                               dtype=np.float32)
    refine_config = {**fmm["pinn_inversion"], "refine_steps": search["refine_steps"],
                     "refine_adam_learning_rate_on_box_fractions":
                         search["refine_adam_learning_rate_on_box_fractions"]}
    rows = []
    for event, path in zip(protocol["events"], paths):
        with np.load(path) as archive:
            observed = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        if not np.array_equal(receivers.astype(np.float64), receiver_coords):
            raise ValueError("Receiver geometry differs between events")
        digest = hash_file(path)
        rms = float(np.sqrt(np.mean(observed.astype(np.float64) ** 2)))
        if rms <= 0:
            raise ValueError(f"Empty observation: {path}")
        started = time.perf_counter()
        try:
            arrival = invert_one(fields, receivers, observed, times, 40. * stride,
                                 tuple(fmm_config["coarse_grid_xyz"]),
                                 fmm_config["refinement_grid_xyz"],
                                 fmm_config["refinement_levels"],
                                 tuple(fmm["shared_source_bounds_m_s"][3]),
                                 fmm_config["pick_threshold_fraction_per_trace"],
                                 fmm_config["minimum_valid_picks"],
                                 fmm_config["minimum_station_peak_fraction_of_record"],
                                 fmm_config["last_record_s_excluded_for_pick"])
            arrival["status"] = "ok"
        except InsufficientPicks as exc:
            arrival = {"status": "insufficient_valid_picks", "valid_fd_picks": exc.count,
                       "estimated_source_m_s": None, "reason": str(exc)}
        arrival["search_seconds"] = time.perf_counter() - started
        neural = {"status": "fmm_initializer_unavailable", "estimated_source_m_s": None}
        if arrival["status"] == "ok":
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            candidates = proposals(np.asarray(arrival["estimated_source_m_s"][:3]),
                                   bounds, search["expanded_offsets_m"], origin_times)
            scores = fine_scores(model, receivers, times, observed, scale, rms, candidates,
                                 search["receiver_stride"], search["time_stride"],
                                 search["candidate_batch"])
            starts_indices = choose_spatial_starts(candidates, scores,
                                                    search["expanded_start_count"],
                                                    search["minimum_start_separation_m"])
            starts = [{"source_m_s": candidates[i].tolist(),
                       "coarse_normalized_mse": float(scores[i])} for i in starts_indices]
            if device.type == "cuda":
                torch.cuda.synchronize()
            scoring_seconds = time.perf_counter() - started
            started = time.perf_counter()
            winner, _, _, _, _ = refine(model, receivers, times,
                                        torch.as_tensor(observed, device=device),
                                        scale, bounds, refine_config, rms, starts)
            if device.type == "cuda":
                torch.cuda.synchronize()
            neural = {"status": "ok", "estimated_source_m_s": winner["estimated_source_m_s"],
                      "normalized_mse": winner["normalized_mse"],
                      "on_boundary": winner["on_region_boundary"],
                      "candidate_count": len(candidates), "start_count": len(starts),
                      "scoring_seconds": scoring_seconds,
                      "refinement_seconds": time.perf_counter() - started}
        rows.append({"event": event["file"], "observation": str(path.resolve()),
                     "observation_sha256": digest, "fmm": arrival, "pinn": neural})
        print(f"Search {event['file']}: FMM {arrival['status']}, PINN {neural['status']}", flush=True)

    # Persist all estimates first; truth remains unopened by both inversion methods.
    estimates = {"protocol_sha256": hash_file(PROTOCOL),
                 "checkpoint_sha256": protocol["checkpoint_sha256"],
                 "velocity_sha256": protocol["velocity_sha256"],
                 "fmm_setup_seconds": fmm_setup_seconds,
                 "fmm_receiver_fields_mib": fields.nbytes / 1048576, "events": rows}
    with outputs[0].open("x", encoding="utf-8") as stream:
        json.dump(estimates, stream, indent=2)
    for row, event, path in zip(rows, protocol["events"], paths):
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
            observed = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        if not np.allclose(truth, event["source_m_s"], atol=1e-5, rtol=0):
            raise ValueError(f"FD truth differs from registration: {path}")
        row["truth_post_search_m_s"] = truth.tolist()
        row["fd_generation_seconds"] = json.loads(path.with_suffix(".json").read_text())[
            "runtime_seconds"]
        row["fmm"].update(error(row["fmm"]["estimated_source_m_s"], truth))
        row["pinn"].update(error(row["pinn"]["estimated_source_m_s"], truth))
        row["pinn"]["truth_source_raw_waveform_relative_l2"] = forward_l2(
            model, receivers, times, observed, truth, scale)
    gates = protocol["predeclared_checks"]
    valid = [row for row in rows if row["pinn"]["status"] == "ok"]
    checks = {
        "fmm_picks": all(row["fmm"]["status"] == "ok" and row["fmm"]["valid_fd_picks"]
                         >= gates["minimum_valid_fmm_picks_per_event"] for row in rows),
        "pinn_location": sum(row["pinn"]["location_error_m"] < 500 for row in valid)
                         >= gates["minimum_events_with_location_error_below_500_m"],
        "pinn_origin_time": sum(row["pinn"]["origin_time_error_s"] < .05 for row in valid)
                            >= gates["minimum_events_with_origin_time_error_below_0_05_s"],
        "pinn_median_location": len(valid) == len(rows) and float(np.median([
            row["pinn"]["location_error_m"] for row in valid]))
            < gates["maximum_pinn_median_location_error_m"],
        "pinn_beats_fmm": sum(row["pinn"]["location_error_m"]
                              < row["fmm"]["location_error_m"] for row in valid)
                          >= gates["minimum_events_pinn_better_than_fmm"],
    }
    report = {**estimates, "checks": checks, "all_gates_passed": all(checks.values()),
              "median_location_error_m": {
                  label: float(np.median([row[label]["location_error_m"] for row in rows
                                          if row[label]["location_error_m"] is not None]))
                  if any(row[label]["location_error_m"] is not None for row in rows) else None
                  for label in ("fmm", "pinn")},
              "limitations": protocol["limitations"]}
    with outputs[1].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    for row in rows:
        print(f"{row['event']}: FMM={row['fmm']['location_error_m']} m, "
              f"PINN={row['pinn']['location_error_m']} m; "
              f"time error={row['pinn']['origin_time_error_s']} s", flush=True)
    print(f"Saved {outputs[1]}; checks={checks}", flush=True)


if __name__ == "__main__":
    main()
