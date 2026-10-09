"""Evaluate five independently seeded source-conditioned members."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.confirm_depth_gap_25 import inversion_without_truth
from source_conditioned_pinn import load_conditioned_checkpoint


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "experiments/source_conditioned_ensemble_members.json")
    parser.add_argument("--search-protocol", type=Path,
                        default=ROOT / "experiments/cross_region_fmm_protocol.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/source_conditioned_ensemble_results.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    search = json.loads(args.search_protocol.read_text(encoding="utf-8"))
    members = manifest["members"]
    if len(members) != 5 or len({row["seed"] for row in members}) != 5:
        raise ValueError("Expected five unique ensemble members")
    for member in members:
        path = Path(member["checkpoint"])
        if not path.is_file() or sha256(path) != member["sha256"]:
            raise ValueError(f"Missing or changed ensemble checkpoint: {path}")

    events = []
    data_dir = ROOT / "data/observations/cross_region_fmm"
    for registered in search["events"]:
        path = data_dir / registered["file"]
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if not np.allclose(truth, registered["source_m_s"], rtol=0, atol=1e-5):
            raise ValueError(f"Registered truth changed: {path}")
        events.append({"event": path.name, "path": path, "sha256": sha256(path),
                       "data": data, "times": times, "receivers": receivers,
                       "truth": truth, "members": []})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_inversion_seconds = 0.0
    member_summaries = []
    for member in members:
        checkpoint = Path(member["checkpoint"])
        model, metadata = load_conditioned_checkpoint(checkpoint, device)
        if (metadata["training_record"]["seed"] != member["seed"]
                or len(metadata["training_record"]["receiver_observations"]) != 25):
            raise ValueError(f"Unexpected training provenance: {checkpoint}")
        model.eval().requires_grad_(False)
        scale = np.asarray([metadata["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        member_location_errors = []
        member_time_errors = []
        member_runtime = 0.0
        for event in events:
            started = time.perf_counter()
            inverse, _ = inversion_without_truth(
                model, event["data"], event["times"], event["receivers"],
                {"bounds_m_s": search["shared_source_bounds_m_s"]},
                search["pinn_inversion"], scale)
            runtime = time.perf_counter() - started
            estimate = np.asarray(inverse["estimated_source_m_s"], dtype=np.float64)
            error = estimate - event["truth"]
            location_error = float(np.linalg.norm(error[:3]))
            time_error = float(abs(error[3]))
            event["members"].append({
                "seed": member["seed"],
                "estimated_source_m_s": estimate.tolist(),
                "component_error_m_s": error.tolist(),
                "location_error_m": location_error,
                "origin_time_error_s": time_error,
                "inversion_seconds": runtime,
            })
            member_location_errors.append(location_error)
            member_time_errors.append(time_error)
            member_runtime += runtime
            print(f"seed {member['seed']} {event['event']}: "
                  f"location={location_error:.1f} m, time={time_error:.4f} s",
                  flush=True)
        total_inversion_seconds += member_runtime
        member_summaries.append({
            "seed": member["seed"],
            "mean_location_error_m": float(np.mean(member_location_errors)),
            "median_location_error_m": float(np.median(member_location_errors)),
            "mean_origin_time_error_s": float(np.mean(member_time_errors)),
            "median_origin_time_error_s": float(np.median(member_time_errors)),
            "eight_event_inversion_seconds": member_runtime,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    serializable_events = []
    coverage = np.zeros(4, dtype=int)
    widths = []
    mean_location_errors = []
    mean_time_errors = []
    for event in events:
        estimates = np.asarray([row["estimated_source_m_s"] for row in event["members"]])
        mean = estimates.mean(axis=0)
        std = estimates.std(axis=0, ddof=1)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        covered = (event["truth"] >= lower) & (event["truth"] <= upper)
        width = upper - lower
        coverage += covered.astype(int)
        widths.append(width)
        mean_error = mean - event["truth"]
        location_error = float(np.linalg.norm(mean_error[:3]))
        time_error = float(abs(mean_error[3]))
        mean_location_errors.append(location_error)
        mean_time_errors.append(time_error)
        serializable_events.append({
            "event": event["event"],
            "sha256": event["sha256"],
            "source_truth_m_s": event["truth"].tolist(),
            "members": event["members"],
            "ensemble_mean_m_s": mean.tolist(),
            "sample_standard_deviation_m_s": std.tolist(),
            "nominal_95_percent_interval_lower_m_s": lower.tolist(),
            "nominal_95_percent_interval_upper_m_s": upper.tolist(),
            "nominal_interval_width_m_s": width.tolist(),
            "truth_covered_by_nominal_interval": covered.tolist(),
            "ensemble_mean_location_error_m": location_error,
            "ensemble_mean_origin_time_error_s": time_error,
        })
    widths_array = np.asarray(widths)

    baseline_receiver = json.loads(
        (ROOT / "outputs/depth_gap_25_receiver_5000/training_summary.json").read_text())
    baseline_hybrid = json.loads(
        (ROOT / "outputs/depth_gap_25_hybrid_1000/training_summary.json").read_text())
    baseline_training = (baseline_receiver["5Hz"]["seconds"]
                         + baseline_hybrid["5Hz"]["seconds"])
    new_training = sum(float(row.get("total_training_seconds", 0.0)) for row in members)
    total_training = baseline_training + new_training
    labels = ("x", "y", "z", "origin_time")
    output = {
        "purpose": "post-hoc five-seed source-conditioned ensemble stability audit",
        "ensemble_size": 5,
        "seeds": [row["seed"] for row in members],
        "interval_definition": "member mean plus or minus 1.96 sample standard deviations",
        "calibrated_statistical_coverage_claimed": False,
        "evaluation_status": "already-inspected development events; not independent confirmation",
        "event_count": len(events),
        "component_summary": {
            label: {
                "covered_events": int(coverage[index]),
                "empirical_coverage_fraction": float(coverage[index] / len(events)),
                "mean_nominal_interval_width": float(widths_array[:, index].mean()),
                "median_nominal_interval_width": float(np.median(widths_array[:, index])),
                "unit": "s" if index == 3 else "m",
            } for index, label in enumerate(labels)
        },
        "ensemble_mean_summary": {
            "mean_location_error_m": float(np.mean(mean_location_errors)),
            "median_location_error_m": float(np.median(mean_location_errors)),
            "mean_origin_time_error_s": float(np.mean(mean_time_errors)),
            "median_origin_time_error_s": float(np.median(mean_time_errors)),
        },
        "member_summaries": member_summaries,
        "cost": {
            "five_member_training_seconds": total_training,
            "mean_training_seconds_per_member": total_training / 5,
            "forty_member_event_inversions_seconds": total_inversion_seconds,
            "mean_inversion_seconds_per_member_event": total_inversion_seconds / 40,
        },
        "events": serializable_events,
        "limitations": (
            "Eight post-hoc same-model synthetic events and M=5 only. The interval "
            "describes seed dispersion and excludes data noise, velocity uncertainty, "
            "receiver geometry, source physics and model-form uncertainty."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"component_summary": output["component_summary"],
                      "ensemble_mean_summary": output["ensemble_mean_summary"],
                      "cost": output["cost"]}, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
