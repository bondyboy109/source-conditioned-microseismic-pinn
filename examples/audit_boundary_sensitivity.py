"""Compare the registered zero-boundary and lateral/bottom-sponge FD records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.compare_ray_shifted_template import (
    quality, ray_travel_times, shifted_template)


def observation(path: Path, expected_source: list[float]):
    with np.load(path) as archive:
        data = np.asarray(archive["data"], dtype=np.float64)
        times = np.asarray(archive["times"], dtype=np.float64)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
        source = np.asarray(archive["source_truth"], dtype=np.float64)
    if data.shape != (25, 4501) or times.shape != (4501,) or receivers.shape != (25, 3):
        raise ValueError(f"Unexpected geometry in {path}")
    if not np.isfinite(data).all() or not np.allclose(
            source, expected_source, atol=1e-5, rtol=0):
        raise ValueError(f"Invalid source or waveform in {path}")
    return data, times, receivers, source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "dev_fd_boundary_sensitivity.json")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "experiments" / "boundary_sensitivity_development.json")
                          .read_text(encoding="utf-8"))
    results = []
    arrays = []
    for entry in protocol["events"]:
        old_path = ROOT / "data" / "observations" / entry["original"]
        new_path = ROOT / "data" / "observations" / entry["sponge"]
        old = observation(old_path, entry["source_m_s"])
        new = observation(new_path, entry["source_m_s"])
        if not (np.array_equal(old[1], new[1])
                and np.array_equal(old[2], new[2])
                and np.array_equal(old[3], new[3])):
            raise ValueError("Boundary comparison must share time, stations and source")
        old_report = json.loads(old_path.with_suffix(".json").read_text(encoding="utf-8"))
        new_report = json.loads(new_path.with_suffix(".json").read_text(encoding="utf-8"))
        if (old_report["boundary"] != protocol["unchanged_boundary"]
                or new_report["boundary"] != protocol["modified_boundary"]
                or new_report["sponge_width_m"] != protocol["sponge_width_m"]
                or new_report["sponge_max_damping_s"] != protocol["sponge_max_damping_s"]):
            raise ValueError("FD boundary reports differ from registered settings")
        original, sponge, times = old[0], new[0], old[1]
        total_energy = np.sum(original ** 2)
        segment_results = {}
        for start, end in protocol["time_windows_s"]:
            mask = (times >= start - 1e-6) & (times <= end + 1e-6)
            baseline = original[:, mask]
            changed = sponge[:, mask]
            segment_results[f"{start:g}_to_{end:g}_s"] = {
                "original_energy_fraction": float(np.sum(baseline ** 2) / total_energy),
                "sponge_vs_original": quality(changed, baseline),
                "sponge_to_original_energy": float(
                    np.sum(changed ** 2) / np.sum(baseline ** 2)),
            }
        for start, end in ((0.0, 1.0), (1.0, 2.0)):
            mask = (times >= start - 1e-6) & (times <= end + 1e-6)
            baseline = original[:, mask]
            changed = sponge[:, mask]
            segment_results[f"{start:g}_to_{end:g}_s"] = {
                "original_energy_fraction": float(np.sum(baseline ** 2) / total_energy),
                "sponge_vs_original": quality(changed, baseline),
                "sponge_to_original_energy": float(
                    np.sum(changed ** 2) / np.sum(baseline ** 2)),
            }
        record = {"event": entry["original"],
                  "full_sponge_vs_original": quality(sponge, original),
                  "time_windows": segment_results}
        results.append(record)
        arrays.append((old, new))
        print(f"{record['event']}: boundary-change L2 "
              + ", ".join(f"{name}={value['sponge_vs_original']['relative_l2']:.3f}"
                          for name, value in segment_results.items()), flush=True)

    original_source, sponge_source = arrays[0]
    original_near, sponge_near = arrays[1]
    model = np.load(ROOT / "data" / "velocity" / "vp_model.npy", mmap_mode="r")
    receivers = original_source[2]
    origin = original_source[3]
    target = original_near[3]
    constant_speed = 3500.0
    travel_old_constant = np.linalg.norm(receivers - origin[:3], axis=1) / constant_speed
    travel_new_constant = np.linalg.norm(receivers - target[:3], axis=1) / constant_speed
    travel_old_ray = ray_travel_times(model, origin, receivers)
    travel_new_ray = ray_travel_times(model, target, receivers)
    comparisons = {}
    for label, (source_record, target_record) in {
            "zero_boundary": (original_source, original_near),
            "lateral_bottom_sponge": (sponge_source, sponge_near),
    }.items():
        comparisons[label] = {}
        for kind, old_time, new_time in (
                ("fixed_speed", travel_old_constant, travel_new_constant),
                ("integrated_slowness", travel_old_ray, travel_new_ray)):
            predicted = shifted_template(source_record[0], source_record[1],
                                         origin, target, receivers, old_time, new_time)
            comparisons[label][kind] = quality(predicted, target_record[0])
            for end_s in (2.0, 4.0):
                mask = source_record[1] <= end_s + 1e-6
                comparisons[label][kind][f"first_{end_s:g}_s"] = quality(
                    predicted[:, mask], target_record[0][:, mask])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "purpose": "development-only lateral/bottom boundary sensitivity, not PML",
        "configuration": protocol,
        "events": results,
        "train01_template_to_250m": comparisons,
    }, indent=2), encoding="utf-8")
    for label, versions in comparisons.items():
        print(f"250 m template {label}: fixed={versions['fixed_speed']['relative_l2']:.3f}, "
              f"ray={versions['integrated_slowness']['relative_l2']:.3f}", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
