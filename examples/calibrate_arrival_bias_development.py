"""Audit picker timing bias with leave-one-event-out FMM localization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.invert_fmm_receiver_fields import interpolate_fields, receiver_fields
from examples.validate_arrival_constrained import (
    digest, event_files, load_protocol, pick_receivers, source_search,
)

PROTOCOL = ROOT / "experiments/arrival_bias_development_protocol.json"


def estimate_from_picks(fields: np.ndarray, picks: np.ndarray, truth: np.ndarray,
                        spec: dict) -> dict:
    search = source_search(fields, picks, spec)
    if search is None:
        return {"status": "insufficient_valid_picks", "valid_picks": int(np.isfinite(picks).sum())}
    candidates, arrival_rmse, _ = search
    estimate = candidates[int(np.argmin(arrival_rmse))]
    return {"status": "ok", "valid_picks": int(np.isfinite(picks).sum()),
            "estimated_source_m_s": estimate.tolist(),
            "location_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
            "origin_time_error_s": float(abs(estimate[3] - truth[3])),
            "origin_time_signed_error_s": float(estimate[3] - truth[3])}


def summarize(rows, key):
    valid = [row[key] for row in rows if row[key]["status"] == "ok"]
    return {"event_count": len(rows), "failure_count": len(rows) - len(valid),
            "mean_location_error_m": float(np.mean([v["location_error_m"] for v in valid])) if valid else None,
            "mean_origin_time_error_s": float(np.mean([v["origin_time_error_s"] for v in valid])) if valid else None,
            "mean_signed_origin_error_s": float(np.mean([v["origin_time_signed_error_s"] for v in valid])) if valid else None}


def audit(spec, paths, fields, picker):
    records = []
    for path in paths:
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float64)
            source = np.asarray(archive["source_truth"], dtype=np.float64)
        picks = pick_receivers(data, times, picker, spec)
        travel_at_truth = interpolate_fields(fields, source[None, :3], spec["fmm_spacing_m"])[0]
        offsets = picks - source[3] - travel_at_truth
        valid = np.isfinite(offsets)
        if valid.sum() < spec["min_valid_picks"]:
            raise ValueError(f"Insufficient calibration picks: {path}")
        records.append({"path": path, "source": source, "picks": picks,
                        "offsets": offsets[valid],
                        "median_offset_s": float(np.median(offsets[valid]))})
    medians = np.asarray([record["median_offset_s"] for record in records])
    rows = []
    for index, record in enumerate(records):
        loo_bias = float(np.median(np.delete(medians, index)))
        rows.append({"observation": str(record["path"]),
                     "observation_sha256": digest(record["path"]),
                     "true_source_m_s": record["source"].tolist(),
                     "median_pick_bias_at_true_source_s": record["median_offset_s"],
                     "receiver_bias_mad_s": float(np.median(np.abs(
                         record["offsets"] - record["median_offset_s"]))),
                     "leave_one_event_out_bias_s": loo_bias,
                     "raw": estimate_from_picks(fields, record["picks"], record["source"], spec),
                     "calibrated": estimate_from_picks(
                         fields, record["picks"] - loo_bias, record["source"], spec)})
        print(record["path"].name, "offset", round(record["median_offset_s"], 4),
              "LOO", round(loo_bias, 4), flush=True)
    return medians, rows


def main():
    calibration = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if digest(ROOT / calibration["source_protocol"]) != calibration["source_protocol_sha256"]:
        raise ValueError("Frozen source protocol changed")
    if (digest(ROOT / calibration["selected_development"])
            != calibration["selected_development_sha256"]):
        raise ValueError("Frozen picker decision changed")
    output = ROOT / calibration["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    spec = load_protocol()
    picker = json.loads((ROOT / calibration["selected_development"]).read_text(
        encoding="utf-8"))["selected"]["picker"]
    paths = event_files(spec, "development")
    if len(paths) != 16 or any(not path.is_file() for path in paths):
        raise ValueError("Expected all 16 development records")
    velocity = np.asarray(np.load(ROOT / spec["velocity"], mmap_mode="r")
                          [::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"]],
                          dtype=np.float32)
    receivers = np.load(ROOT / spec["receivers"])
    fields = receiver_fields(velocity, receivers, spec["fmm_spacing_m"])
    medians, rows = audit(spec, paths, fields, picker)
    raw, corrected = summarize(rows, "raw"), summarize(rows, "calibrated")
    center = float(np.median(medians))
    dispersion = float(np.median(np.abs(medians - center)))
    rule = calibration["calibration_adoption_rule"]
    time_gain = 1.0 - corrected["mean_origin_time_error_s"] / raw["mean_origin_time_error_s"]
    spatial_increase = corrected["mean_location_error_m"] / raw["mean_location_error_m"] - 1.0
    usable = (dispersion <= rule["max_event_offset_mad_s"]
              and time_gain >= rule["min_mean_origin_error_reduction_fraction"]
              and spatial_increase <= rule["max_mean_location_error_increase_fraction"]
              and corrected["failure_count"] - raw["failure_count"] <= rule["max_new_failures"])
    result = {"role": calibration["role"], "protocol_sha256": digest(PROTOCOL),
              "source_protocol_sha256": calibration["source_protocol_sha256"],
              "selected_picker": picker, "event_count": len(rows),
              "all_development_event_median_bias_s": center,
              "event_median_bias_mad_s": dispersion,
              "event_median_bias_range_s": [float(medians.min()), float(medians.max())],
              "raw": raw, "leave_one_event_out_calibrated": corrected,
              "relative_origin_time_error_reduction": time_gain,
              "relative_location_error_increase": spatial_increase,
              "adoption_rule": rule, "calibration_adoptable": bool(usable),
              "events": rows,
              "limits": "Only historical development records are used. No previously inspected holdout contributes to the correction or to the leave-one-event-out decision."}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "all_development_event_median_bias_s", "event_median_bias_mad_s",
        "raw", "leave_one_event_out_calibrated", "calibration_adoptable")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
