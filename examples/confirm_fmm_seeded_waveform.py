"""One-time registered FD comparison of frozen FMM, PINN and hybrid searches."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.confirm_depth_gap_25 import inversion_without_truth, refine
from examples.run_fmm_seeded_waveform_development import (
    choose_spatial_starts, fine_scores, hash_file, proposals,
)
from source_conditioned_pinn import conditioned_prediction, load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/fmm_seeded_independent_protocol.json"
DATA = ROOT / "data/observations/fmm_seeded_independent"
RESULTS = ROOT / "experiments"


def prepare_events(protocol: dict, training: list[dict]) -> list[Path]:
    entries = protocol["events"]
    sources = [np.asarray(item["source_m_s"], dtype=np.float32) for item in entries]
    if (len(entries) != 8 or len({item["file"] for item in entries}) != 8
            or any(np.allclose(a, b, rtol=0, atol=1e-4)
                   for index, a in enumerate(sources) for b in sources[index + 1:])):
        raise ValueError("Registered event list is not eight distinct sources")
    for source in sources:
        if any(np.allclose(source, item["source_truth"], rtol=0, atol=1e-4)
               for item in training):
            raise ValueError("Registered source overlaps receiver training")
    # Verify no registered source appeared in any previous observation before generation.
    for old in (ROOT / "data/observations").rglob("*.npz"):
        if DATA in old.parents:
            continue
        with np.load(old) as archive:
            if "source_truth" not in archive.files:
                continue
            earlier = archive["source_truth"]
        if any(np.allclose(earlier, source, rtol=0, atol=1e-4) for source in sources):
            raise ValueError(f"Registered source appeared previously: {old}")
    DATA.mkdir(parents=True, exist_ok=True)
    paths = []
    hashes = {item["sha256"] for item in training}
    for item in entries:
        path = DATA / item["file"]
        if not path.exists():
            print(f"FD generating {path.name}", flush=True)
            subprocess.run([sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                            "--duration", "2", "--source",
                            *(str(value) for value in item["source_m_s"]),
                            "--output", str(path)], cwd=ROOT, check=True)
        with np.load(path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.isfinite(archive["data"]).all()
                    or not np.allclose(archive["source_truth"], item["source_m_s"],
                                       rtol=0, atol=1e-5)):
                raise ValueError(f"Unexpected FD observation: {path}")
        fd_meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (fd_meta["duration_s"] != 2 or fd_meta["frequency_hz"] != 5
                or fd_meta["boundary"] != "zero Dirichlet on all six faces"
                or hash_file(path) in hashes):
            raise ValueError(f"FD generation differs or overlaps training: {path}")
        paths.append(path)
    return paths


def main() -> None:
    registered = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    hybrid = json.loads((ROOT / registered["hybrid_settings"]).read_text(encoding="utf-8"))
    original = json.loads((ROOT / "experiments/cross_region_fmm_protocol.json").read_text())
    checkpoint = ROOT / hybrid["model_checkpoint"]
    velocity = ROOT / hybrid["fmm_velocity"]
    if (hash_file(checkpoint) != registered["hybrid_checkpoint_sha256"]
            or registered["hybrid_checkpoint_sha256"] != hybrid["model_sha256"]
            or hash_file(velocity) != registered["velocity_sha256"]
            or registered["velocity_sha256"] != hybrid["fmm_velocity_sha256"]
            or hybrid["refinement"]["bounds_m_s"] != original["shared_source_bounds_m_s"]):
        raise ValueError("Frozen model, velocity or search bounds changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False)
    training = metadata["training_record"]["receiver_observations"]
    if len(training) != 29 or metadata["frequency_hz"] != 5:
        raise ValueError("Expected the registered 29-source checkpoint")
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    paths = prepare_events(registered, training)
    baseline = original["fmm_baseline"]
    fmm_target = RESULTS / "fmm_seeded_independent_fmm.json"
    subprocess.run([sys.executable, str(ROOT / "examples/invert_fmm_receiver_fields.py"),
                    "--velocity", str(velocity), "--observation", *(str(path) for path in paths),
                    "--output", str(fmm_target), "--stride", str(baseline["velocity_grid_stride"]),
                    "--coarse-grid", *(str(value) for value in baseline["coarse_grid_xyz"]),
                    "--refinement-grid", str(baseline["refinement_grid_xyz"]),
                    "--refinement-levels", str(baseline["refinement_levels"]),
                    "--pick-threshold", str(baseline["pick_threshold_fraction_per_trace"]),
                    "--min-peak-fraction", str(baseline["minimum_station_peak_fraction_of_record"]),
                    "--pick-end-margin-s", str(baseline["last_record_s_excluded_for_pick"]),
                    "--min-valid-picks", str(baseline["minimum_valid_picks"]),
                    "--origin-bounds", *(str(value) for value in original["shared_source_bounds_m_s"][3])],
                   cwd=ROOT, check=True)
    fmm = json.loads(fmm_target.read_text(encoding="utf-8"))
    if len(fmm["events"]) != 8:
        raise ValueError("FMM baseline omitted registered events")
    bounds = np.asarray(hybrid["refinement"]["bounds_m_s"], dtype=np.float32)
    scan = hybrid["origin_time_grid_s"]
    origins = np.linspace(scan["minimum"], scan["maximum"],
                          round((scan["maximum"] - scan["minimum"]) / scan["step"]) + 1,
                          dtype=np.float32)
    score = hybrid["fine_scoring"]
    starts_spec = hybrid["initialization"]
    config = {**original["pinn_inversion"],
              "refine_steps": hybrid["refinement"]["steps"],
              "refine_adam_learning_rate_on_box_fractions":
              hybrid["refinement"]["adam_learning_rate_on_box_fractions"]}
    events = []
    for entry, path, fmm_row in zip(registered["events"], paths, fmm["events"]):
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        digest = hash_file(path)
        if fmm_row["sha256"] != digest or Path(fmm_row["observation"]) != path:
            raise ValueError(f"FMM and PINN records differ: {path}")
        independent, _ = inversion_without_truth(model, data, times, receivers,
                                                  {"bounds_m_s": bounds.tolist()},
                                                  original["pinn_inversion"], scale)
        proposed = None
        if fmm_row["status"] == "ok":
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            candidates = proposals(np.asarray(fmm_row["estimated_source_m_s"][:3]), bounds,
                                   hybrid["spatial_offsets_m_each_axis"], origins)
            rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
            losses = fine_scores(model, receivers, times, data, scale, rms, candidates,
                                 score["receiver_stride"], score["time_stride"],
                                 score["candidate_batch"])
            indices = choose_spatial_starts(candidates, losses,
                                            starts_spec["max_restarts"],
                                            starts_spec["minimum_spatial_distance_m"])
            candidate_starts = [{"source_m_s": candidates[index].tolist(),
                                 "coarse_normalized_mse": float(losses[index])}
                                for index in indices]
            if device.type == "cuda":
                torch.cuda.synchronize()
            scoring_s = time.perf_counter() - started
            started = time.perf_counter()
            winner, refined, _, _, _ = refine(
                model, receivers, times, torch.as_tensor(data, device=device),
                scale, bounds, config, rms, candidate_starts)
            if device.type == "cuda":
                torch.cuda.synchronize()
            refinement_s = time.perf_counter() - started
            proposed = {"estimated_source_m_s": winner["estimated_source_m_s"],
                        "normalized_mse": winner["normalized_mse"],
                        "candidate_count": len(candidates),
                        "starts": [{"source_m_s": item["source_m_s"],
                                    "fine_normalized_mse": item["coarse_normalized_mse"]}
                                   for item in candidate_starts],
                        "winner_on_boundary": winner["on_region_boundary"],
                        "scoring_seconds": scoring_s,
                        "refinement_seconds": refinement_s}
        # Ground truth is opened only after both source searches finish.
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float32)
        if not np.allclose(truth, entry["source_m_s"], atol=1e-5, rtol=0):
            raise ValueError(f"FD truth differs from registration: {path}")
        delta = np.asarray(independent["estimated_source_m_s"]) - truth
        independent_result = {"estimated_source_m_s": independent["estimated_source_m_s"],
                              "location_error_m": float(np.linalg.norm(delta[:3])),
                              "origin_time_error_s": float(abs(delta[3])),
                              "search_seconds": independent["runtime_seconds"]}
        if proposed:
            difference = np.asarray(proposed["estimated_source_m_s"]) - truth
            proposed["location_error_m"] = float(np.linalg.norm(difference[:3]))
            proposed["origin_time_error_s"] = float(abs(difference[3]))
        full_rms = float(np.sqrt(np.sum(data.astype(np.float64) ** 2)))
        with torch.inference_mode():
            prediction = conditioned_prediction(
                model, torch.as_tensor(receivers / scale[:3], device=device),
                torch.as_tensor(times / scale[3], device=device),
                torch.as_tensor(truth / scale, device=device)).cpu().numpy()
        forward_l2 = float(np.linalg.norm(prediction.astype(np.float64) - data) / full_rms)
        events.append({"event": entry["file"], "sha256": digest,
                       "truth_post_search": truth.tolist(), "forward_raw_relative_l2": forward_l2,
                       "fd_generation_seconds": json.loads(path.with_suffix(".json").read_text())[
                           "runtime_seconds"],
                       "fmm": {key: fmm_row[key] for key in (
                           "status", "valid_fd_picks", "estimated_source_m_s",
                           "location_error_m", "origin_time_error_s", "search_seconds")},
                       "pinn_only": independent_result, "hybrid": proposed})
        print(f"{entry['file']}: PINN={independent_result['location_error_m']:.1f} m, "
              f"FMM={fmm_row['location_error_m']}, "
              f"hybrid={None if proposed is None else round(proposed['location_error_m'], 1)} m",
              flush=True)
    gates = registered["predeclared_gates"]
    complete = [row["hybrid"] for row in events if row["hybrid"] is not None]
    valid = len(complete) == 8
    fmm_median = float(np.median([row["fmm"]["location_error_m"] for row in events
                                  if row["fmm"]["status"] == "ok"]))
    hybrid_median = float(np.median([row["location_error_m"] for row in complete])) if valid else None
    checks = {
        "fmm_valid_picks": all(row["fmm"]["valid_fd_picks"] >=
                               gates["all_eight_fmm_valid_picks_at_least"] for row in events),
        "hybrid_location_count": sum(row["location_error_m"] < 500 for row in complete)
                                 >= gates["hybrid_location_below_500_m_at_least"],
        "hybrid_origin_time_count": sum(row["origin_time_error_s"] < .05 for row in complete)
                                    >= gates["hybrid_origin_time_below_0_05_s_at_least"],
        "hybrid_median_location": valid and hybrid_median <
                                  gates["hybrid_median_location_below_m"],
        "hybrid_vs_fmm_median": valid and hybrid_median < fmm_median,
        "hybrid_forward_count": sum(row["forward_raw_relative_l2"] < .75 for row in events)
                                >= gates["hybrid_raw_forward_l2_below_0_75_at_least"]}
    report = {"purpose": "one-time same-model synthetic FMM-seeded waveform confirmation",
              "protocol": str(PROTOCOL.resolve()), "checkpoint_sha256": hash_file(checkpoint),
              "velocity_sha256": hash_file(velocity),
              "fmm_setup_seconds": fmm["setup_seconds"],
              "fmm_receiver_fields_mib": fmm["receiver_field_cache_mib"],
              "fmm_median_location_m": fmm_median,
              "hybrid_median_location_m": hybrid_median,
              "checks": checks, "all_gates_passed": all(checks.values()),
              "events": events}
    target = RESULTS / "fmm_seeded_independent_results.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {target}; checks={checks}; all_pass={report['all_gates_passed']}", flush=True)


if __name__ == "__main__":
    main()
