"""Run the registered, one-time independent FD/FMM/PINN comparison."""

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

PROTOCOL = ROOT / "experiments/eikonal_independent_protocol.json"
DATA = ROOT / "data/observations/eikonal_independent"
RESULT = ROOT / "experiments/eikonal_independent_results.json"


def check_inputs(protocol: dict) -> tuple[dict, dict]:
    global_settings = json.loads((ROOT / protocol["prior_search_settings"]).read_text())
    hybrid_settings = json.loads((ROOT / protocol["prior_fmm_seeded_settings"]).read_text())
    for path_key, hash_key in (("eikonal_checkpoint", "eikonal_sha256"),
                               ("straight_checkpoint", "straight_sha256"),
                               ("velocity_file", "velocity_sha256")):
        if hash_file(ROOT / protocol[path_key]) != protocol[hash_key]:
            raise ValueError(f"Registered input changed: {protocol[path_key]}")
    if (hybrid_settings["refinement"]["bounds_m_s"] != global_settings["shared_source_bounds_m_s"]
            or hybrid_settings["model_sha256"] != protocol["straight_sha256"]
            or hybrid_settings["fmm_velocity_sha256"] != protocol["velocity_sha256"]):
        raise ValueError("Historical search settings differ from the frozen protocol")
    events = protocol["events"]
    if len(events) != 8 or len({row["file"] for row in events}) != 8:
        raise ValueError("Expected eight distinct event files")
    sources = np.asarray([row["source_m_s"] for row in events], dtype=np.float64)
    bounds = np.asarray(global_settings["shared_source_bounds_m_s"], dtype=np.float64)
    if (np.any(sources < bounds[:, 0]) or np.any(sources > bounds[:, 1])
            or len({tuple(row) for row in sources[:, :3]}) != len(events)):
        raise ValueError("Sources must have distinct positions inside the fixed search bounds")

    # Audit all existing observations before generating or reading new FD waveforms.
    previous = []
    for path in sorted((ROOT / "data/observations").rglob("*.npz")):
        if DATA in path.parents:
            continue
        with np.load(path) as archive:
            if "source_truth" in archive.files:
                previous.append((path, np.asarray(archive["source_truth"][:3], dtype=np.float64)))
    for source in sources:
        for path, earlier in previous:
            if np.allclose(source[:3], earlier, atol=0.01, rtol=0):
                raise ValueError(f"Registered source position already observed in {path}")
    print(f"Pre-generation spatial overlap audit passed: {len(previous)} previous FD records, "
          f"{len(sources)} distinct new positions", flush=True)
    return global_settings, hybrid_settings


def prepare_events(protocol: dict, training_hashes: set[str]) -> list[Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in protocol["events"]:
        path = DATA / row["file"]
        if not path.exists():
            subprocess.run([sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                            "--duration", "2", "--output", str(path), "--source",
                            *(str(value) for value in row["source_m_s"])], cwd=ROOT, check=True)
        with np.load(path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.isfinite(archive["data"]).all()
                    or not np.allclose(archive["source_truth"], row["source_m_s"],
                                       atol=1e-5, rtol=0)):
                raise ValueError(f"FD record does not match registration: {path}")
        fd = json.loads(path.with_suffix(".json").read_text())
        if (fd["duration_s"] != 2 or fd["frequency_hz"] != 5
                or fd["boundary"] != "zero Dirichlet on all six faces"
                or hash_file(path) in training_hashes):
            raise ValueError(f"FD provenance or training overlap invalid: {path}")
        paths.append(path)
    return paths


def forward_l2(model, data: np.ndarray, times: np.ndarray, receivers: np.ndarray,
               truth: np.ndarray, scale: np.ndarray) -> float:
    device = next(model.parameters()).device
    with torch.inference_mode():
        prediction = conditioned_prediction(
            model, torch.as_tensor(receivers / scale[:3], device=device),
            torch.as_tensor(times / scale[3], device=device),
            torch.as_tensor(truth.astype(np.float32) / scale, device=device)).cpu().numpy()
    return float(np.linalg.norm(prediction.astype(np.float64) - data) /
                 np.linalg.norm(data.astype(np.float64)))


def errors(estimate: list[float], truth: np.ndarray) -> dict:
    difference = np.asarray(estimate, dtype=np.float64) - truth
    return {"component_error_m_s": difference.tolist(),
            "location_error_m": float(np.linalg.norm(difference[:3])),
            "origin_time_error_s": float(abs(difference[3]))}


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(f"One-time result already exists: {RESULT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    global_settings, hybrid_settings = check_inputs(protocol)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for label in ("eikonal", "straight"):
        started = time.perf_counter()
        model, metadata = load_conditioned_checkpoint(
            ROOT / protocol[f"{label}_checkpoint"], device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model.eval().requires_grad_(False)
        if (metadata["frequency_hz"] != 5 or len(metadata["training_record"]["receiver_observations"]) != 29):
            raise ValueError(f"Unexpected {label} checkpoint training manifest")
        scale = np.asarray([metadata["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        loaded[label] = (model, scale, time.perf_counter() - started)
    training_hashes = [
        {row["sha256"] for row in load_conditioned_checkpoint(
            ROOT / protocol[f"{label}_checkpoint"], torch.device("cpu"))[1][
                "training_record"]["receiver_observations"]}
        for label in ("eikonal", "straight")]
    if training_hashes[0] != training_hashes[1]:
        raise ValueError("Neural operators were not trained on the same 29 events")
    paths = prepare_events(protocol, training_hashes[0])

    fmm_settings = global_settings["fmm_baseline"]
    fmm_target = ROOT / "experiments/eikonal_independent_fmm.json"
    if not fmm_target.exists():
        subprocess.run([sys.executable, str(ROOT / "examples/invert_fmm_receiver_fields.py"),
                    "--velocity", str(ROOT / protocol["velocity_file"]),
                    "--observation", *(str(path) for path in paths),
                    "--output", str(fmm_target),
                    "--stride", str(fmm_settings["velocity_grid_stride"]),
                    "--coarse-grid", *(str(n) for n in fmm_settings["coarse_grid_xyz"]),
                    "--refinement-grid", str(fmm_settings["refinement_grid_xyz"]),
                    "--refinement-levels", str(fmm_settings["refinement_levels"]),
                    "--pick-threshold", str(fmm_settings["pick_threshold_fraction_per_trace"]),
                    "--min-peak-fraction", str(fmm_settings["minimum_station_peak_fraction_of_record"]),
                    "--pick-end-margin-s", str(fmm_settings["last_record_s_excluded_for_pick"]),
                    "--min-valid-picks", str(fmm_settings["minimum_valid_picks"]),
                    "--origin-bounds", *(str(v) for v in global_settings["shared_source_bounds_m_s"][3])],
                       cwd=ROOT, check=True)
    fmm = json.loads(fmm_target.read_text(encoding="utf-8"))
    if (len(fmm["events"]) != len(paths)
            or Path(fmm["velocity"]) != (ROOT / protocol["velocity_file"]).resolve()
            or fmm["grid_stride"] != fmm_settings["velocity_grid_stride"]
            or fmm["coarse_grid_xyz"] != fmm_settings["coarse_grid_xyz"]
            or fmm["refinement_grid_xyz"] != fmm_settings["refinement_grid_xyz"]
            or fmm["refinement_levels"] != fmm_settings["refinement_levels"]
            or fmm["pick_threshold_fraction"] != fmm_settings["pick_threshold_fraction_per_trace"]
            or fmm["minimum_station_peak_fraction"] != fmm_settings["minimum_station_peak_fraction_of_record"]
            or fmm["pick_end_margin_s"] != fmm_settings["last_record_s_excluded_for_pick"]
            or fmm["origin_bounds_s"] != global_settings["shared_source_bounds_m_s"][3]
            or any(row["sha256"] != hash_file(path) or Path(row["observation"]) != path
                   for row, path in zip(fmm["events"], paths))):
        raise ValueError("Existing FMM report differs from the registered inputs or events")

    bounds = np.asarray(hybrid_settings["refinement"]["bounds_m_s"], dtype=np.float32)
    scan = hybrid_settings["origin_time_grid_s"]
    origins = np.linspace(scan["minimum"], scan["maximum"],
                          round((scan["maximum"] - scan["minimum"]) / scan["step"]) + 1,
                          dtype=np.float32)
    score = hybrid_settings["fine_scoring"]
    starts_spec = hybrid_settings["initialization"]
    refine_config = {**global_settings["pinn_inversion"],
                     "refine_steps": hybrid_settings["refinement"]["steps"],
                     "refine_adam_learning_rate_on_box_fractions":
                     hybrid_settings["refinement"]["adam_learning_rate_on_box_fractions"]}
    results = []
    for entry, path, arrival in zip(protocol["events"], paths, fmm["events"]):
        digest = hash_file(path)
        if arrival["sha256"] != digest or Path(arrival["observation"]) != path:
            raise ValueError(f"FMM and neural operators evaluated different records: {path}")
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
        if rms <= 0:
            raise ValueError(f"Empty waveform: {path}")
        straight_model, straight_scale, _ = loaded["straight"]
        straight, _ = inversion_without_truth(
            straight_model, data, times, receivers, {"bounds_m_s": bounds.tolist()},
            global_settings["pinn_inversion"], straight_scale)
        hybrid = None
        if arrival["status"] == "ok":
            model, scale, _ = loaded["eikonal"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            candidates = proposals(np.asarray(arrival["estimated_source_m_s"][:3]),
                                   bounds, hybrid_settings["spatial_offsets_m_each_axis"], origins)
            scores = fine_scores(model, receivers, times, data, scale, rms, candidates,
                                 score["receiver_stride"], score["time_stride"],
                                 score["candidate_batch"])
            indices = choose_spatial_starts(candidates, scores, starts_spec["max_restarts"],
                                             starts_spec["minimum_spatial_distance_m"])
            starts = [{"source_m_s": candidates[i].tolist(),
                       "coarse_normalized_mse": float(scores[i])} for i in indices]
            if device.type == "cuda":
                torch.cuda.synchronize()
            scoring_seconds = time.perf_counter() - started
            started = time.perf_counter()
            winner, _, _, _, _ = refine(
                model, receivers, times, torch.as_tensor(data, device=device),
                scale, bounds, refine_config, rms, starts)
            if device.type == "cuda":
                torch.cuda.synchronize()
            hybrid = {"estimated_source_m_s": winner["estimated_source_m_s"],
                      "normalized_mse": winner["normalized_mse"],
                      "candidate_count": len(candidates), "start_count": len(starts),
                      "scoring_seconds": scoring_seconds,
                      "refinement_seconds": time.perf_counter() - started,
                      "winner_on_boundary": winner["on_region_boundary"]}

        # All three searches finish before consulting the FD generator's source truth.
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if not np.allclose(truth, entry["source_m_s"], atol=1e-5, rtol=0):
            raise ValueError(f"Generated source differs from registered event: {path}")
        straight_row = {"estimated_source_m_s": straight["estimated_source_m_s"],
                        "normalized_mse": straight["final_normalized_mse"],
                        "search_seconds": straight["runtime_seconds"],
                        **errors(straight["estimated_source_m_s"], truth),
                        "raw_waveform_relative_l2": forward_l2(
                            straight_model, data, times, receivers, truth, straight_scale)}
        if hybrid is not None:
            hybrid.update(errors(hybrid["estimated_source_m_s"], truth))
        eikonal_model, eikonal_scale, _ = loaded["eikonal"]
        results.append({"event": entry["file"], "sha256": digest,
                        "truth_post_search_m_s": truth.tolist(),
                        "fd_generation_seconds": json.loads(path.with_suffix(".json").read_text())[
                            "runtime_seconds"],
                        "eikonal_raw_waveform_relative_l2": forward_l2(
                            eikonal_model, data, times, receivers, truth, eikonal_scale),
                        "fmm": {key: arrival.get(key) for key in
                                ("status", "valid_fd_picks", "estimated_source_m_s",
                                 "location_error_m", "origin_time_error_s", "search_seconds")},
                        "straight_global": straight_row, "eikonal_hybrid": hybrid})
        print(f"{entry['file']}: FMM={arrival['location_error_m']}, "
              f"straight={straight_row['location_error_m']:.1f} m, "
              f"eikonal={None if hybrid is None else round(hybrid['location_error_m'], 1)} m",
              flush=True)

    gates = protocol["predeclared_gates"]
    hybrids = [row["eikonal_hybrid"] for row in results if row["eikonal_hybrid"] is not None]
    checks = {
        "fmm_picks": all(row["fmm"]["status"] == "ok" and row["fmm"]["valid_fd_picks"]
                         >= gates["all_eight_fmm_valid_picks_at_least"] for row in results),
        "eikonal_location": sum(row["location_error_m"] < 500 for row in hybrids)
                            >= gates["eikonal_location_below_500_m_at_least"],
        "eikonal_origin_time": sum(row["origin_time_error_s"] < .05 for row in hybrids)
                               >= gates["eikonal_origin_time_below_0_05_s_at_least"],
        "eikonal_median_location": len(hybrids) == 8 and
                                   float(np.median([r["location_error_m"] for r in hybrids]))
                                   < gates["eikonal_median_location_below_m"],
        "eikonal_beats_fmm": sum(row["eikonal_hybrid"] is not None and
                                 row["eikonal_hybrid"]["location_error_m"] < row["fmm"]["location_error_m"]
                                 for row in results if row["fmm"]["location_error_m"] is not None)
                              >= gates["eikonal_location_better_than_fmm_at_least"],
        "eikonal_forward_l2": sum(row["eikonal_raw_waveform_relative_l2"] < .85
                                  for row in results) >= gates["eikonal_forward_l2_below_0_85_at_least"],
    }
    report = {"purpose": protocol["purpose"], "protocol": str(PROTOCOL.resolve()),
              "eikonal_sha256": protocol["eikonal_sha256"],
              "straight_sha256": protocol["straight_sha256"],
              "velocity_sha256": protocol["velocity_sha256"],
              "model_load_seconds": {key: loaded[key][2] for key in loaded},
              "fmm_setup_seconds": fmm["setup_seconds"],
              "fmm_receiver_fields_mib": fmm["receiver_field_cache_mib"],
              "median_location_error_m": {label: float(np.median([
                  row[label]["location_error_m"] for row in results if row[label] is not None
                  and row[label]["location_error_m"] is not None]))
                  for label in ("fmm", "straight_global", "eikonal_hybrid")},
              "checks": checks, "all_gates_passed": all(checks.values()),
              "events": results, "limitations": protocol["limitations"]}
    RESULT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {RESULT}; gates={checks}; all_pass={report['all_gates_passed']}", flush=True)


if __name__ == "__main__":
    main()
