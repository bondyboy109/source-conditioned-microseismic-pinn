"""Run registered frozen-model cross-region PINN/FMM comparison."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from confirm_depth_gap_25 import inversion_without_truth
from invert_source_conditioned import waveform_mse
from source_conditioned_pinn import load_conditioned_checkpoint


PROTOCOL = ROOT / "experiments/cross_region_fmm_protocol.json"
DATA = ROOT / "data/observations/cross_region_fmm"
RESULTS = ROOT / "experiments"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_events(protocol: dict, training: list[dict]) -> list[Path]:
    entries = protocol["events"]
    candidates = [np.asarray(row["source_m_s"], dtype=np.float32) for row in entries]
    if (len(entries) != 8 or len({row["file"] for row in entries}) != 8
            or any(np.allclose(a, b, rtol=0, atol=1e-3)
                   for i, a in enumerate(candidates) for b in candidates[i+1:])):
        raise ValueError("The registered source list must have eight unique records")
    bounds = np.asarray(protocol["shared_source_bounds_m_s"])
    for source in candidates:
        if np.any(source < bounds[:, 0]) or np.any(source > bounds[:, 1]):
            raise ValueError("Registered event lies outside preassigned source bounds")
        if any(np.allclose(source, row["source_truth"], atol=1e-3, rtol=0)
               for row in training):
            raise ValueError("Registered event overlaps a PINN training source")
    # Check earlier records before generating new ones; prior test events are consumed.
    for previous in (ROOT / "data/observations").rglob("*.npz"):
        if DATA in previous.parents:
            continue
        with np.load(previous) as archive:
            if "source_truth" not in archive.files:
                continue
            source = archive["source_truth"]
        if any(np.allclose(source, new, atol=1e-3, rtol=0) for new in candidates):
            raise ValueError(f"Registered source already appeared in earlier data: {previous}")
    DATA.mkdir(parents=True, exist_ok=True)
    paths = []
    training_hashes = {row["sha256"] for row in training}
    for row in entries:
        path = DATA / row["file"]
        if not path.exists():
            print(f"FD generating {path.name}", flush=True)
            subprocess.run([sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                            "--output", str(path), "--duration", "2", "--source",
                            *(str(value) for value in row["source_m_s"])], cwd=ROOT, check=True)
        if not path.with_suffix(".json").is_file():
            raise ValueError(f"Missing FD provenance: {path}")
        report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (report["frequency_hz"] != 5 or report["duration_s"] != 2
                or report["boundary"] != "zero Dirichlet on all six faces"):
            raise ValueError(f"FD protocol differs from registered configuration: {path}")
        with np.load(path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.isfinite(archive["data"]).all()
                    or not np.allclose(archive["source_truth"], row["source_m_s"], rtol=0, atol=1e-5)):
                raise ValueError(f"Unexpected FD source or waveform: {path}")
        if hash_file(path) in training_hashes:
            raise ValueError(f"FD test file appears in PINN training set: {path}")
        paths.append(path)
    return paths


def gate_results(entries: list[dict], gates: dict) -> dict:
    locations = [row["pinn"]["location_error_m"] for row in entries]
    times = [row["pinn"]["origin_time_error_s"] for row in entries]
    fmm = [row["fmm"]["location_error_m"] for row in entries
           if row["fmm"]["status"] == "ok"]
    return {
        "pinn_all_forward": all(row["pinn"]["raw_waveform_relative_l2"] <
                                gates["all_eight_pinn_raw_waveform_relative_l2_below"]
                                for row in entries),
        "pinn_six_location": sum(value < gates["at_least_six_of_eight_pinn_location_error_m_below"]
                                  for value in locations) >= 6,
        "pinn_six_origin_time": sum(value < gates["at_least_six_of_eight_pinn_origin_time_error_s_below"]
                                     for value in times) >= 6,
        "pinn_median_location": float(np.median(locations)) < gates[
            "pinn_median_location_error_m_below"],
        "pinn_vs_fmm_median_location": (len(fmm) == len(entries) and
                                          float(np.median(locations)) < float(np.median(fmm))),
        "fmm_all_valid_picks": all(row["fmm"]["valid_fd_picks"] >= 6
                                    for row in entries),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checkpoint = ROOT / protocol["fixed_pinn_checkpoint"]
    velocity = ROOT / protocol["fixed_velocity_file"]
    if (hash_file(checkpoint).lower() != protocol["fixed_pinn_sha256"]
            or hash_file(velocity).lower() != protocol["fixed_velocity_sha256"]):
        raise ValueError("Frozen network or velocity-grid checksum changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    model, metadata = load_conditioned_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False)
    pinn_setup_s = time.perf_counter() - started
    training = metadata["training_record"]["receiver_observations"]
    if len(training) != 25 or metadata["frequency_hz"] != 5:
        raise ValueError("Unexpected frozen PINN checkpoint")
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    paths = prepare_events(protocol, training)
    baseline = protocol["fmm_baseline"]
    fmm_path = RESULTS / "cross_region_fmm_baseline.json"
    subprocess.run([sys.executable, str(ROOT / "examples/invert_fmm_receiver_fields.py"),
                    "--velocity", str(velocity), "--observation", *(str(path) for path in paths),
                    "--output", str(fmm_path), "--stride", str(baseline["velocity_grid_stride"]),
                    "--coarse-grid", *(str(value) for value in baseline["coarse_grid_xyz"]),
                    "--refinement-grid", str(baseline["refinement_grid_xyz"]),
                    "--refinement-levels", str(baseline["refinement_levels"]),
                    "--pick-threshold", str(baseline["pick_threshold_fraction_per_trace"]),
                    "--min-peak-fraction", str(baseline["minimum_station_peak_fraction_of_record"]),
                    "--pick-end-margin-s", str(baseline["last_record_s_excluded_for_pick"]),
                    "--min-valid-picks", str(baseline["minimum_valid_picks"]),
                    "--origin-bounds", *(str(value) for value in protocol["shared_source_bounds_m_s"][3])],
                   cwd=ROOT, check=True)
    fmm_output = json.loads(fmm_path.read_text(encoding="utf-8"))
    if len(fmm_output["events"]) != len(paths):
        raise ValueError("FMM did not return all registered events")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inversions = []
    for entry, path in zip(protocol["events"], paths):
        # Deliberately omit source_truth when calling the search interface.
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            receiver_coords = np.asarray(archive["receiver_coords"], dtype=np.float32)
            sample_times = np.asarray(archive["times"], dtype=np.float32)
        bounds = {"bounds_m_s": protocol["shared_source_bounds_m_s"]}
        inverse, (rms, xyz, time_t, selected) = inversion_without_truth(
            model, data, sample_times, receiver_coords, bounds,
            protocol["pinn_inversion"], scale)
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float32)
        if not np.allclose(truth, entry["source_m_s"], atol=1e-5, rtol=0):
            raise ValueError("FD source does not match preregistered coordinates")
        estimate = np.asarray(inverse["estimated_source_m_s"])
        error = estimate - truth
        with torch.inference_mode():
            truth_t = torch.as_tensor(truth / scale, device=device)
            truth_loss = float(waveform_mse(model, xyz, time_t, selected, truth_t, rms))
        inverse.update({"location_error_m": float(np.linalg.norm(error[:3])),
                        "origin_time_error_s": float(abs(error[3])),
                        "component_error_m_s": error.tolist(),
                        "truth_source_normalized_mse": truth_loss,
                        "source_truth_m_s_post_search": truth.tolist()})
        inversions.append(inverse)
        output = RESULTS / f"cross_region_pinn_{path.stem}.json"
        output.write_text(json.dumps(inverse, indent=2), encoding="utf-8")
        print(f"PINN {path.name}: location={inverse['location_error_m']:.1f}m, "
              f"time={inverse['origin_time_error_s']:.4f}s, "
              f"search={inverse['runtime_seconds']:.2f}s", flush=True)
    gpu_peak = (int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda"
                else None)
    forward_path = RESULTS / "cross_region_forward.json"
    subprocess.run([sys.executable, str(ROOT / "examples/audit_forward_error_decomposition.py"),
                    str(checkpoint), *(str(path) for path in paths),
                    "--evaluation-label", "synthetic-confirmation", "--output", str(forward_path)],
                   cwd=ROOT, check=True)
    forward = json.loads(forward_path.read_text(encoding="utf-8"))
    if len(forward["events"]) != len(paths):
        raise ValueError("Forward evaluator omitted test events")
    records = []
    for entry, path, inverse, fmm, predicted in zip(
            protocol["events"], paths, inversions, fmm_output["events"], forward["events"]):
        digest = hash_file(path)
        if (predicted["sha256"] != digest or fmm["sha256"] != digest
                or predicted["file_in_receiver_training_set"]
                or predicted["source_in_receiver_training_set"]):
            raise ValueError("FMM/PINN observations or training overlap do not match")
        records.append({"event": entry["file"], "source_truth_m_s": entry["source_m_s"],
                        "observation": str(path), "sha256": digest,
                        "fd_runtime_s": json.loads(path.with_suffix(".json").read_text(
                            encoding="utf-8"))["runtime_seconds"],
                        "pinn": {"estimated_source_m_s": inverse["estimated_source_m_s"],
                                 "location_error_m": inverse["location_error_m"],
                                 "origin_time_error_s": inverse["origin_time_error_s"],
                                 "runtime_seconds": inverse["runtime_seconds"],
                                 "raw_waveform_relative_l2": predicted[
                                     "original_amplitude_relative_l2"],
                                 "true_source_normalized_mse": inverse[
                                     "truth_source_normalized_mse"],
                                 "estimated_source_normalized_mse": inverse[
                                     "final_normalized_mse"]},
                        "fmm": {"status": fmm["status"],
                                "estimated_source_m_s": fmm["estimated_source_m_s"],
                                "location_error_m": fmm["location_error_m"],
                                "origin_time_error_s": fmm["origin_time_error_s"],
                                "valid_fd_picks": fmm["valid_fd_picks"],
                                "arrival_rmse_s": fmm.get("arrival_rmse_s"),
                                "search_seconds": fmm["search_seconds"]}})
    checks = gate_results(records, protocol["predeclared_gates"])
    report = {"purpose": "one-time synthetic cross-region localization comparison",
              "protocol": str(PROTOCOL.resolve()),
              "pinn_checkpoint_sha256": hash_file(checkpoint),
              "velocity_sha256": hash_file(velocity),
              "pinn_setup_seconds": pinn_setup_s,
              "pinn_evaluation_peak_cuda_allocation_bytes": gpu_peak,
              "fmm_setup_seconds": fmm_output["setup_seconds"],
              "fmm_receiver_fields_mib": fmm_output["receiver_field_cache_mib"],
              "gates": checks, "all_gates_passed": all(checks.values()),
              "events": records}
    target = RESULTS / "cross_region_fmm_comparison.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {target}; gates={checks}; all_pass={report['all_gates_passed']}", flush=True)


if __name__ == "__main__":
    main()
