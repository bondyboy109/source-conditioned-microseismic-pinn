"""Post-hoc pick-error stress audit for the frozen low-rank operator.

The same perturbed picks are supplied to the neural hybrid and pure FMM
branches. No model, weight, picker or search parameter is changed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.cross_model_geology_audit import fmm_locate
from examples.invert_fmm_receiver_fields import receiver_fields
from examples.run_holdout_noise_picker_comparison import robust_arrival
from examples.run_source_conditioned_low_rank_cross_geology import load_model_with_fields
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, choose_diverse_starts, coarse_grid, load_event, refine,
    sampled_waveforms, waveform_scores,
)

PROTOCOL = ROOT / "experiments/low_rank_pick_perturbation_protocol.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def corrected_picks(event: dict, arrival: dict) -> np.ndarray:
    raw = np.asarray([robust_arrival(trace, event["times"])
                      for trace in event["data"]], dtype=np.float32)
    peaks = np.max(np.abs(event["data"]), axis=1)
    valid = (np.isfinite(raw)
             & (peaks >= float(arrival["minimum_receiver_peak_fraction_of_record"])
                * peaks.max())
             & (raw <= event["times"][-1]
                - float(arrival["last_record_s_excluded_for_pick"])))
    picks = raw - float(arrival["pick_bias_s"])
    picks[~valid] = np.nan
    return picks


def run_branch(model, candidates, waveform, travel, alpha, domain, bounds,
               refined, picks, search, arrival):
    score = waveform if alpha == 0.0 else waveform + alpha * travel
    ids = choose_diverse_starts(candidates, score, int(search["restarts"]),
                                float(search["minimum_start_separation_m"]))
    starts = [{"source_m_s": candidates[i].tolist(),
               "coarse_total_loss": float(score[i])} for i in ids]
    best, _ = refine(model, starts, domain, bounds, refined, picks, search,
                     arrival, alpha)
    return best


def crop_to_model_window(event: dict, time_length_s: float) -> dict:
    """Restrict observations to the time interval represented by the frozen model."""
    times = np.asarray(event["times"], dtype=np.float32)
    keep = times <= float(time_length_s) + 1.0e-7
    if int(keep.sum()) < 2:
        raise ValueError(f"Observation has fewer than two samples in [0, {time_length_s}] s")
    cropped = dict(event)
    cropped["data"] = np.asarray(event["data"][:, keep], dtype=np.float32)
    cropped["times"] = times[keep]
    cropped["original_time_samples"] = int(len(times))
    cropped["original_record_length_s"] = float(times[-1] - times[0])
    cropped["evaluation_time_window_s"] = [float(times[0]), float(cropped["times"][-1])]
    cropped["rms"] = float(np.sqrt(np.mean(cropped["data"].astype(np.float64) ** 2)))
    if not np.isfinite(cropped["rms"]) or cropped["rms"] <= 0:
        raise ValueError("Cropped observation has invalid RMS")
    return cropped


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["output_result"]
    report = ROOT / spec["output_report"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite pick-perturbation output")
    checkpoint = ROOT / spec["checkpoint"]
    velocity_path = ROOT / spec["velocity_file"]
    if sha256(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash changed")
    if sha256(velocity_path) != spec["velocity_sha256"]:
        raise ValueError("Velocity hash changed")
    receiver_path = ROOT / spec["receivers_file"]
    receivers = np.asarray(np.load(receiver_path), dtype=np.float32)
    if sha256(receiver_path) != spec["receivers_sha256"]:
        raise ValueError("Receiver geometry hash changed")
    velocity = np.asarray(np.load(velocity_path, mmap_mode="r")[::2, ::2, ::2], dtype=np.float32)
    cache = ROOT / "outputs/low_rank_pick_perturbation/fmm_fields_stride2.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    expected_shape = (len(receivers), 201, 201, 47)
    cache_reused = False
    started = time.perf_counter()
    if cache.exists():
        fields = np.load(cache, mmap_mode="r")
        if fields.shape != expected_shape or not np.isfinite(fields).all():
            raise ValueError("FMM cache shape or values are invalid")
        cache_reused = True
    else:
        fields = receiver_fields(velocity, receivers, float(spec["fmm_spacing_m"]))
        if fields.shape != expected_shape:
            raise ValueError(f"Unexpected FMM field shape: {fields.shape}")
        np.save(cache, fields)
    fmm_seconds = time.perf_counter() - started
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain, _ = load_model_with_fields(checkpoint, fields, receivers,
                                               float(spec["fmm_spacing_m"]), device)
    bounds = np.asarray(spec["source_bounds_m_s"], dtype=np.float32)
    search = spec["search"]
    arrival = spec["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    for event_spec in spec["events"]:
        path = ROOT / spec["test_directory"] / event_spec["file"]
        raw_event = load_event(path, receivers)
        event = crop_to_model_window(raw_event, float(domain[3]))
        picks = corrected_picks(event, arrival)
        valid = np.isfinite(picks)
        enough_picks = int(valid.sum()) >= int(arrival["minimum_valid_picks"])
        coarse = sampled_waveforms(event, domain,
                                   int(search["coarse_receiver_stride"]),
                                   int(search["coarse_time_stride"]), device)
        fine = sampled_waveforms(event, domain,
                                 int(search["refine_receiver_stride"]),
                                 int(search["refine_time_stride"]), device)
        waveform = waveform_scores(model, candidates, domain, coarse,
                                   int(search["candidate_batch"]))
        truth = np.asarray(event_spec["truth_m_s"], dtype=np.float64)
        for offset_samples in spec["offset_samples"]:
            for seed in spec["seeds"]:
                rng = np.random.default_rng(int(seed) + int(offset_samples) * 100003)
                perturb = rng.normal(0.0, float(offset_samples) * float(spec["dt_s"]), size=len(picks))
                perturbed = picks.copy()
                perturbed[valid] += perturb[valid].astype(np.float32)
                branches = {}
                for name, alpha in (("waveform_only", 0.0),):
                    best = run_branch(model, candidates, waveform, None, alpha,
                                      domain, bounds, fine, perturbed, search, arrival)
                    estimate = np.asarray(best["estimated_source_m_s"], dtype=np.float64)
                    error = estimate - truth
                    branches[name] = {"estimated_source_m_s": estimate.tolist(),
                                      "location_error_m": float(np.linalg.norm(error[:3])),
                                      "origin_time_error_s": float(abs(error[3])),
                                      "waveform_normalized_mse": float(best["waveform_normalized_mse"]),
                                      "arrival_normalized_mse": float(best["arrival_normalized_mse"])}
                if enough_picks:
                    travel = arrival_scores(model, candidates, perturbed,
                                            float(arrival["arrival_scale_s"]),
                                            int(search["candidate_batch"]))
                    best = run_branch(model, candidates, waveform, travel,
                                      float(arrival["hybrid_weight"]), domain,
                                      bounds, fine, perturbed, search, arrival)
                    estimate = np.asarray(best["estimated_source_m_s"], dtype=np.float64)
                    error = estimate - truth
                    branches["hybrid"] = {
                        "estimated_source_m_s": estimate.tolist(),
                        "location_error_m": float(np.linalg.norm(error[:3])),
                        "origin_time_error_s": float(abs(error[3])),
                        "waveform_normalized_mse": float(best["waveform_normalized_mse"]),
                        "arrival_normalized_mse": float(best["arrival_normalized_mse"]),
                    }
                    fmm_event = {"truth": truth}
                    branches["fmm"] = fmm_locate(
                        fields, fmm_event, perturbed, bounds[:3],
                        float(spec["fmm_spacing_m"]), search, arrival)
                else:
                    reason = (f"only {int(valid.sum())} valid picks in the "
                              f"{event['evaluation_time_window_s'][1]:.3f}-s window; "
                              f"minimum is {int(arrival['minimum_valid_picks'])}")
                    for name in ("hybrid", "fmm"):
                        branches[name] = {"status": "failure", "failure_reason": reason,
                                          "location_error_m": None,
                                          "origin_time_error_s": None}
                rows.append({"event": event_spec["name"], "file": path.name,
                             "offset_samples_sd": int(offset_samples), "seed": int(seed),
                             "valid_picks": int(valid.sum()), "branches": branches,
                             "truth_m_s": truth.tolist(),
                             "evaluation_time_window_s": event["evaluation_time_window_s"],
                             "n_time_samples_used": int(len(event["times"])),
                             "original_time_samples": int(event["original_time_samples"]),
                             "original_record_length_s": event["original_record_length_s"]})
        print(f"{event_spec['name']}: completed", flush=True)
    summaries = {}
    for offset in spec["offset_samples"]:
        selected = [r for r in rows if r["offset_samples_sd"] == offset]
        branch_summary = {}
        for name in ("waveform_only", "hybrid", "fmm"):
            values = [float(r["branches"][name]["location_error_m"])
                      for r in selected
                      if r["branches"][name].get("location_error_m") is not None]
            times = [float(r["branches"][name]["origin_time_error_s"])
                     for r in selected
                     if r["branches"][name].get("origin_time_error_s") is not None]
            branch_summary[name] = {
                "mean_eloc_m": float(np.mean(values)) if values else None,
                "median_eloc_m": float(np.median(values)) if values else None,
                "mean_et_s": float(np.mean(times)) if times else None,
                "failure_count": len(selected) - len(values),
                "n_success": len(values),
            }
        summaries[str(offset)] = {
            "n": len(selected),
            "valid_picks_by_event": {r["event"]: r["valid_picks"]
                                      for r in selected[::len(spec["seeds"])]},
            "branches": branch_summary,
        }
    result = {"purpose": spec["purpose"], "protocol": str(PROTOCOL.resolve()),
              "protocol_sha256": sha256(PROTOCOL), "checkpoint_sha256": sha256(checkpoint),
              "velocity_sha256": sha256(velocity_path), "device": str(device),
              "fmm_setup_seconds": fmm_seconds, "fmm_cache": str(cache.resolve()),
              "fmm_cache_reused": cache_reused, "evaluation_time_length_s": float(domain[3]),
              "summary": summaries, "events": rows,
              "limitations": spec["limitations"]}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Pick perturbation audit for the frozen low-rank method", "",
             spec["limitations"], "",
             "| Pick-error SD (samples) | Trials | Waveform median eloc (m) | Hybrid median eloc (m) | FMM median eloc (m) |",
             "|---:|---:|---:|---:|---:|"]
    for offset in spec["offset_samples"]:
        row = summaries[str(offset)]
        def fmt(value):
            return "NA" if value is None else f"{value:.1f}"
        lines.append(f"| {offset} | {row['n']} | {fmt(row['branches']['waveform_only']['median_eloc_m'])} | {fmt(row['branches']['hybrid']['median_eloc_m'])} | {fmt(row['branches']['fmm']['median_eloc_m'])} |")
    setup_label = "cached FMM field access" if cache_reused else "FMM field construction"
    lines.extend(["", f"{setup_label}: {fmm_seconds:.2f} s.", "", "The same perturbed picks were supplied to all branches. This is a post-hoc stress audit and not a prospective blind test."])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
