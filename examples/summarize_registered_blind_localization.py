"""Verify and summarize the post-hoc registered-blind localization diagnostic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import hash_file

PROTOCOL = ROOT / "experiments/registered_blind_localization_diagnostic_protocol.json"


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["summary_file"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoint = ROOT / spec["checkpoint"]
    if hash_file(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("Registered checkpoint changed")
    source_protocol = json.loads((ROOT / spec["source_protocol"]).read_text())
    registered_truth = {
        item["file"]: np.asarray(item["source"], dtype=np.float64)
        for item in source_protocol["data"]["blind_sources"]
    }
    rows = []
    for item in spec["observations"]:
        observation = ROOT / item["path"]
        if hash_file(observation) != item["sha256"]:
            raise ValueError(f"Observation changed: {observation}")
        raw_path = ROOT / spec["raw_result_directory"] / f"{item['name']}.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        truth = np.asarray(raw["source_truth"], dtype=np.float64)
        if not np.allclose(truth, registered_truth[observation.name], atol=1e-5, rtol=0):
            raise ValueError(f"Registered truth mismatch: {observation}")
        if raw["grid_shape"] != spec["inversion"]["grid"]:
            raise ValueError(f"Grid setting mismatch: {raw_path}")
        row = {
            "event": item["name"],
            "observation_sha256": item["sha256"],
            "truth_source_m_s": raw["source_truth"],
            "estimated_source_m_s": raw["estimated_source"],
            "component_error_m_s": raw["component_error"],
            "location_error_m": raw["location_error_m"],
            "origin_time_error_s": raw["origin_time_error_s"],
            "coarse_normalized_mse": raw["coarse_loss"],
            "final_normalized_mse": raw["final_loss"],
            "truth_normalized_mse": raw["truth_source_normalized_mse"],
            "truth_to_optimized_misfit_ratio": (
                raw["truth_source_normalized_mse"] / raw["final_loss"]
            ),
            "runtime_seconds": raw["runtime_seconds"],
            "all_source_gradient_paths_nonzero": raw["all_source_gradient_paths_nonzero"],
        }
        rows.append(row)
    location = np.asarray([row["location_error_m"] for row in rows])
    origin = np.asarray([row["origin_time_error_s"] for row in rows])
    result = {
        "role": spec["role"],
        "protocol_sha256": hash_file(PROTOCOL),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "configuration": spec["inversion"],
        "summary": {
            "event_count": len(rows),
            "mean_location_error_m": float(np.mean(location)),
            "median_location_error_m": float(np.median(location)),
            "minimum_location_error_m": float(np.min(location)),
            "maximum_location_error_m": float(np.max(location)),
            "mean_origin_time_error_s": float(np.mean(origin)),
            "median_origin_time_error_s": float(np.median(origin)),
            "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows])),
            "truth_misfit_exceeds_optimized_count": int(sum(
                row["truth_normalized_mse"] > row["final_normalized_mse"]
                for row in rows
            )),
        },
        "events": rows,
        "limitations": spec["limitations"],
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
