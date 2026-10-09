"""Run the registered untouched-event five-member ensemble confirmation."""

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

from examples.confirm_depth_gap_25 import inversion_without_truth
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/source_conditioned_ensemble_confirmation_protocol.json"
MANIFEST = ROOT / "experiments/source_conditioned_ensemble_members.json"
DATA = ROOT / "data/observations/source_conditioned_ensemble_confirmation"
FMM_OUTPUT = ROOT / "experiments/source_conditioned_ensemble_confirmation_fmm.json"
OUTPUT = ROOT / "experiments/source_conditioned_ensemble_confirmation_results.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_unused_sources(entries: list[dict]) -> None:
    proposed = [np.asarray(row["source_m_s"], dtype=np.float64) for row in entries]
    for path in (ROOT / "data/observations").rglob("*.npz"):
        if DATA in path.parents:
            continue
        try:
            with np.load(path) as archive:
                if "source_truth" not in archive.files:
                    continue
                source = np.asarray(archive["source_truth"], dtype=np.float64)
        except (OSError, ValueError, KeyError):
            continue
        if any(np.allclose(source, candidate, rtol=0, atol=1e-3)
               for candidate in proposed):
            raise ValueError(f"Confirmation source was already present: {path}")


def prepare_events(protocol: dict) -> list[Path]:
    if DATA.exists() and any(DATA.glob("*.npz")):
        paths = [DATA / row["file"] for row in protocol["events"]]
        if not all(path.is_file() for path in paths):
            raise ValueError("Confirmation directory is partially populated")
    else:
        verify_unused_sources(protocol["events"])
        DATA.mkdir(parents=True, exist_ok=True)
        paths = []
        for row in protocol["events"]:
            path = DATA / row["file"]
            subprocess.run([
                sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                "--output", str(path), "--duration", "2", "--source",
                *(str(value) for value in row["source_m_s"]),
            ], cwd=ROOT, check=True)
            paths.append(path)
    for row, path in zip(protocol["events"], paths):
        with np.load(path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.isfinite(archive["data"]).all()
                    or not np.allclose(archive["source_truth"], row["source_m_s"],
                                       rtol=0, atol=1e-5)):
                raise ValueError(f"Invalid confirmation observation: {path}")
        provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (provenance["frequency_hz"] != 5.0 or provenance["duration_s"] != 2.0
                or provenance["boundary"] != "zero Dirichlet on all six faces"):
            raise ValueError(f"FD configuration changed: {path}")
    return paths


def raw_prediction(model: torch.nn.Module, data: np.ndarray, times: np.ndarray,
                   receivers: np.ndarray, truth: np.ndarray,
                   scale: np.ndarray, device: torch.device) -> tuple[float, np.ndarray]:
    shape = data.shape
    coordinates = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*shape, 3)),
        np.broadcast_to(times[None, :, None], (*shape, 1)),
    ), axis=-1).reshape(-1, 4) / scale
    source = torch.as_tensor(truth / scale, device=device, dtype=torch.float32)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coordinates), 8192):
            points = torch.as_tensor(coordinates[start:start + 8192], device=device,
                                     dtype=torch.float32)
            chunks.append(model(points, source.expand(len(points), 4))
                          .squeeze(-1).cpu().numpy())
    prediction = np.concatenate(chunks).reshape(shape).astype(np.float64)
    relative_l2 = float(np.linalg.norm(prediction - data) / np.linalg.norm(data))
    return relative_l2, prediction


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (not protocol["protocol_written_before_fd_generation"]
            or protocol["seeds"] != manifest["seeds"]
            or protocol["ensemble_size"] != 5):
        raise ValueError("Protocol and frozen ensemble manifest disagree")
    protocol_members = {row["seed"]: row["sha256"] for row in protocol["members"]}
    for member in manifest["members"]:
        checkpoint = Path(member["checkpoint"])
        if (protocol_members.get(member["seed"]) != member["sha256"]
                or sha256(checkpoint) != member["sha256"]):
            raise ValueError(f"Frozen member changed: seed {member['seed']}")
    velocity = ROOT / protocol["velocity_file"]
    if sha256(velocity) != protocol["velocity_sha256"]:
        raise ValueError("Frozen velocity file changed")

    paths = prepare_events(protocol)
    baseline = protocol["fmm_baseline"]
    subprocess.run([
        sys.executable, str(ROOT / "examples/invert_fmm_receiver_fields.py"),
        "--velocity", str(velocity), "--observation", *(str(path) for path in paths),
        "--output", str(FMM_OUTPUT), "--stride", str(baseline["velocity_grid_stride"]),
        "--coarse-grid", *(str(value) for value in baseline["coarse_grid_xyz"]),
        "--refinement-grid", str(baseline["refinement_grid_xyz"]),
        "--refinement-levels", str(baseline["refinement_levels"]),
        "--pick-threshold", str(baseline["pick_threshold_fraction_per_trace"]),
        "--min-peak-fraction", str(baseline["minimum_station_peak_fraction_of_record"]),
        "--pick-end-margin-s", str(baseline["last_record_s_excluded_for_pick"]),
        "--min-valid-picks", str(baseline["minimum_valid_picks"]),
        "--origin-bounds", *(str(value) for value in baseline["origin_time_bounds_s"]),
    ], cwd=ROOT, check=True)
    fmm = json.loads(FMM_OUTPUT.read_text(encoding="utf-8"))

    events = []
    for row, path, fmm_row in zip(protocol["events"], paths, fmm["events"]):
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if fmm_row["sha256"] != sha256(path):
            raise ValueError("FMM observation hash mismatch")
        events.append({"event": path.name, "observation": str(path.resolve()),
                       "sha256": sha256(path), "truth": truth, "data": data,
                       "times": times, "receivers": receivers, "fmm": fmm_row,
                       "members": [], "predictions": []})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inversion_seconds = 0.0
    for member in manifest["members"]:
        model, metadata = load_conditioned_checkpoint(Path(member["checkpoint"]), device)
        if metadata["training_record"]["seed"] != member["seed"]:
            raise ValueError("Checkpoint seed mismatch")
        model.eval().requires_grad_(False)
        scale = np.asarray([metadata["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        for event in events:
            started = time.perf_counter()
            inverse, _ = inversion_without_truth(
                model, event["data"], event["times"], event["receivers"],
                {"bounds_m_s": protocol["shared_source_bounds_m_s"]},
                protocol["pinn_inversion"], scale)
            elapsed = time.perf_counter() - started
            raw_l2, prediction = raw_prediction(
                model, event["data"], event["times"], event["receivers"],
                event["truth"], scale, device)
            estimate = np.asarray(inverse["estimated_source_m_s"], dtype=np.float64)
            error = estimate - event["truth"]
            event["members"].append({
                "seed": member["seed"],
                "estimated_source_m_s": estimate.tolist(),
                "component_error_m_s": error.tolist(),
                "location_error_m": float(np.linalg.norm(error[:3])),
                "origin_time_error_s": float(abs(error[3])),
                "raw_true_source_waveform_relative_l2": raw_l2,
                "inversion_seconds": elapsed,
            })
            event["predictions"].append(prediction)
            inversion_seconds += elapsed
            print(f"seed {member['seed']} {event['event']}: "
                  f"location={np.linalg.norm(error[:3]):.1f} m, "
                  f"time={abs(error[3]):.4f} s, L2={raw_l2:.3f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    component_coverage = np.zeros(4, dtype=int)
    widths = []
    ensemble_location = []
    ensemble_time = []
    serializable_events = []
    for event in events:
        estimates = np.asarray([row["estimated_source_m_s"] for row in event["members"]])
        mean = estimates.mean(axis=0)
        std = estimates.std(axis=0, ddof=1)
        lower, upper = mean - 1.96 * std, mean + 1.96 * std
        covered = (event["truth"] >= lower) & (event["truth"] <= upper)
        width = upper - lower
        error = mean - event["truth"]
        location_error = float(np.linalg.norm(error[:3]))
        time_error = float(abs(error[3]))
        ensemble_prediction = np.mean(np.asarray(event["predictions"]), axis=0)
        ensemble_waveform_l2 = float(
            np.linalg.norm(ensemble_prediction - event["data"])
            / np.linalg.norm(event["data"]))
        component_coverage += covered.astype(int)
        widths.append(width)
        ensemble_location.append(location_error)
        ensemble_time.append(time_error)
        serializable_events.append({
            "event": event["event"], "observation": event["observation"],
            "sha256": event["sha256"], "source_truth_m_s": event["truth"].tolist(),
            "members": event["members"], "ensemble_mean_m_s": mean.tolist(),
            "sample_standard_deviation_m_s": std.tolist(),
            "nominal_95_percent_interval_lower_m_s": lower.tolist(),
            "nominal_95_percent_interval_upper_m_s": upper.tolist(),
            "nominal_interval_width_m_s": width.tolist(),
            "truth_covered_by_nominal_interval": covered.tolist(),
            "ensemble_mean_location_error_m": location_error,
            "ensemble_mean_origin_time_error_s": time_error,
            "ensemble_mean_waveform_relative_l2": ensemble_waveform_l2,
            "fmm": event["fmm"],
        })
        del event["predictions"]

    widths = np.asarray(widths)
    fmm_ok = [row["fmm"] for row in serializable_events if row["fmm"]["status"] == "ok"]
    fmm_location = [row["location_error_m"] for row in fmm_ok]
    gates = protocol["predeclared_gates"]
    gate_results = {
        "six_ensemble_locations_below_500_m": bool(
            sum(value < 500 for value in ensemble_location)
            >= gates["ensemble_mean_location_error_below_500_m_at_least"]),
        "six_ensemble_origin_times_below_0_05_s": bool(
            sum(value < .05 for value in ensemble_time)
            >= gates["ensemble_mean_origin_time_error_below_0_05_s_at_least"]),
        "ensemble_median_location_below_500_m": bool(
            np.median(ensemble_location)
            < gates["ensemble_mean_median_location_error_below_m"]),
        "ensemble_median_location_below_fmm_median": bool(
            len(fmm_location) == len(events)
            and np.median(ensemble_location) < np.median(fmm_location)),
        "each_component_coverage_at_least_six_of_eight": bool(np.all(
            component_coverage >= gates["component_coverage_at_least_events_each"])),
        "mean_x_width_below_2000_m": bool(widths[:, 0].mean()
            < gates["mean_interval_width_x_m_below"]),
        "mean_y_width_below_2000_m": bool(widths[:, 1].mean()
            < gates["mean_interval_width_y_m_below"]),
        "mean_z_width_below_1000_m": bool(widths[:, 2].mean()
            < gates["mean_interval_width_z_m_below"]),
        "mean_origin_time_width_below_0_30_s": bool(widths[:, 3].mean()
            < gates["mean_interval_width_origin_time_s_below"]),
        "all_fmm_events_have_six_valid_picks": bool(
            len(fmm_ok) == len(events) and all(
                row["valid_fd_picks"] >= gates["valid_fmm_picks_at_least_each_event"]
                for row in fmm_ok)),
    }
    labels = ("x", "y", "z", "origin_time")
    fd_seconds = sum(json.loads(Path(row["observation"]).with_suffix(".json").read_text(
        encoding="utf-8"))["runtime_seconds"] for row in serializable_events)
    result = {
        "purpose": protocol["purpose"],
        "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "ensemble_manifest_sha256": sha256(MANIFEST),
        "independent_confirmation": True,
        "ensemble_size": 5,
        "event_count": len(events),
        "component_summary": {
            label: {"covered_events": int(component_coverage[index]),
                    "empirical_coverage_fraction": float(component_coverage[index] / len(events)),
                    "mean_nominal_interval_width": float(widths[:, index].mean()),
                    "median_nominal_interval_width": float(np.median(widths[:, index])),
                    "unit": "s" if index == 3 else "m"}
            for index, label in enumerate(labels)
        },
        "ensemble_mean_summary": {
            "mean_location_error_m": float(np.mean(ensemble_location)),
            "median_location_error_m": float(np.median(ensemble_location)),
            "locations_below_500_m": int(sum(value < 500 for value in ensemble_location)),
            "mean_origin_time_error_s": float(np.mean(ensemble_time)),
            "median_origin_time_error_s": float(np.median(ensemble_time)),
            "origin_times_below_0_05_s": int(sum(value < .05 for value in ensemble_time)),
        },
        "fmm_summary": {
            "successful_events": len(fmm_ok),
            "mean_location_error_m": float(np.mean(fmm_location)) if fmm_location else None,
            "median_location_error_m": float(np.median(fmm_location)) if fmm_location else None,
            "setup_seconds": fmm["setup_seconds"],
            "total_search_seconds": float(sum(row["search_seconds"] for row in fmm["events"])),
        },
        "predeclared_gate_results": gate_results,
        "all_predeclared_gates_passed": bool(all(gate_results.values())),
        "cost": {"fd_data_generation_seconds": fd_seconds,
                 "forty_member_event_inversions_seconds": inversion_seconds,
                 "mean_inversion_seconds_per_member_event": inversion_seconds / 40,
                 "fmm_setup_seconds": fmm["setup_seconds"],
                 "fmm_eight_search_seconds": float(sum(
                     row["search_seconds"] for row in fmm["events"]))},
        "events": serializable_events,
        "limitations": protocol["limitations"],
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("component_summary", "ensemble_mean_summary", "fmm_summary",
                       "predeclared_gate_results", "all_predeclared_gates_passed", "cost")},
                     indent=2), flush=True)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
