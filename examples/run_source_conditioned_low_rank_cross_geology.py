"""Post-hoc cross-geology audit for the frozen low-rank source operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from examples.run_source_conditioned_low_rank_localization import load_model
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, choose_diverse_starts, coarse_grid, load_event, refine,
    sampled_waveforms, waveform_scores,
)
from source_conditioned_low_rank import SourceConditionedLowRankOperator
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator

PROTOCOL = ROOT / "experiments/source_conditioned_low_rank_cross_geology_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_with_fields(checkpoint: Path, fields: np.ndarray,
                           receivers: np.ndarray, spacing: float,
                           device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    spec = payload["protocol"]["model"]
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)), spacing).to(device)
    model = SourceConditionedLowRankOperator(
        interpolator, torch.from_numpy(receivers),
        tuple(spec["spatial_lengths_m"]),
        time_length_s=float(spec["time_length_s"]),
        frequency_hz=float(spec["frequency_hz"]),
        rank=int(spec["rank"]), hidden_dim=int(spec["hidden_dim"]),
        hidden_layers=int(spec["hidden_layers"]),
        causal_width_s=float(spec["causal_width_s"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval().requires_grad_(False)
    domain = np.asarray([*spec["spatial_lengths_m"], spec["time_length_s"]],
                        dtype=np.float32)
    return model, domain, payload


def picks_for(event: dict, arrival: dict) -> tuple[np.ndarray, int]:
    raw = np.asarray([robust_arrival(trace, event["times"])
                      for trace in event["data"]], dtype=np.float32)
    peaks = np.max(np.abs(event["data"]), axis=1)
    valid = (np.isfinite(raw)
             & (peaks >= float(arrival["minimum_receiver_peak_fraction_of_record"])
                * peaks.max())
             & (raw <= event["times"][-1]
                - float(arrival["last_record_s_excluded_for_pick"])))
    corrected = raw - float(arrival["pick_bias_s"])
    corrected[~valid] = np.nan
    return corrected, int(np.isfinite(corrected).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-fmm", action="store_true",
                        help="Measure a fresh FMM solve without overwriting an existing cache")
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / (args.output_result or spec["output_result"])
    report = ROOT / (args.output_report or spec["output_report"])
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite cross-geology results")
    checkpoint = ROOT / spec["checkpoint"]
    velocity_path = ROOT / spec["velocity_file"]
    if sha256(velocity_path) != spec["velocity_sha256"]:
        raise ValueError("Independent velocity model hash mismatch")
    receiver_path = ROOT / "data/observations/receiver_coords.npy"
    receivers = np.load(receiver_path).astype(np.float32)
    cache = ROOT / spec["fmm_cache"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache_load_seconds = None
    cache_comparison = None
    cache_reused = cache.exists() and not args.rebuild_fmm
    if cache_reused:
        started = time.perf_counter()
        fields = np.load(cache, mmap_mode="r")
        if fields.shape != (len(receivers), 201, 201, 47):
            raise ValueError("Existing FMM cache shape mismatch")
        cache_load_seconds = time.perf_counter() - started
        fmm_seconds = None
    else:
        velocity = np.asarray(np.load(velocity_path, mmap_mode="r")[::2, ::2, ::2],
                              dtype=np.float32)
        started = time.perf_counter()
        fields = receiver_fields(velocity, receivers, float(spec["fmm_spacing_m"]))
        fmm_seconds = time.perf_counter() - started
        if cache.exists():
            archived_fields = np.load(cache, mmap_mode="r")
            if archived_fields.shape != fields.shape:
                raise ValueError("Existing FMM cache shape mismatch")
            cache_comparison = max(float(np.max(np.abs(fresh - archived)))
                                   for fresh, archived in zip(fields, archived_fields))
        else:
            np.save(cache, fields)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain, checkpoint_payload = load_model_with_fields(
        checkpoint, fields, receivers, float(spec["fmm_spacing_m"]), device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    bounds = np.asarray(spec["shared_source_bounds_m_s"], dtype=np.float32)
    search, arrival = spec["search"], spec["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    started_all = time.perf_counter()
    for event_spec in spec["events"]:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        event_started = time.perf_counter()
        path = ROOT / spec["observation_directory"] / event_spec["file"]
        if sha256(path) != event_spec["sha256"]:
            raise ValueError(f"Observation hash mismatch: {path}")
        event = load_event(path, receivers)
        corrected, valid_count = picks_for(event, arrival)
        if valid_count < int(arrival["minimum_valid_picks"]):
            raise ValueError(f"Insufficient picks for {path.name}: {valid_count}")
        coarse = sampled_waveforms(event, domain,
                                   int(search["coarse_receiver_stride"]),
                                   int(search["coarse_time_stride"]), device)
        fine = sampled_waveforms(event, domain,
                                 int(search["refine_receiver_stride"]),
                                 int(search["refine_time_stride"]), device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        preparation_seconds = time.perf_counter() - event_started
        started = time.perf_counter()
        wave = waveform_scores(model, candidates, domain, coarse,
                               int(search["candidate_batch"]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wave_coarse_seconds = time.perf_counter() - started
        started = time.perf_counter()
        travel = arrival_scores(model, candidates, corrected,
                                float(arrival["arrival_scale_s"]),
                                int(search["candidate_batch"]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        travel_coarse_seconds = time.perf_counter() - started
        branches = {}
        for name, alpha in (("waveform_only", 0.0),
                            ("waveform_plus_fmm", float(arrival["hybrid_weight"]))):
            started = time.perf_counter()
            score = wave if alpha == 0.0 else wave + alpha * travel
            indices = choose_diverse_starts(
                candidates, score, int(search["restarts"]),
                float(search["minimum_start_separation_m"]))
            starts = [{"source_m_s": candidates[i].tolist(),
                       "coarse_total_loss": float(score[i])} for i in indices]
            best, _ = refine(model, starts, domain, bounds, fine, corrected,
                             search, arrival, alpha)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            refinement_seconds = time.perf_counter() - started
            estimate = np.asarray(best["estimated_source_m_s"], dtype=np.float64)
            truth = np.asarray(event_spec["truth_m_s"], dtype=np.float64)
            error = estimate - truth
            branches[name] = {"estimated_source_m_s": estimate.tolist(),
                              "location_error_m": float(np.linalg.norm(error[:3])),
                              "origin_time_error_s": float(abs(error[3])),
                              "search_seconds": wave_coarse_seconds + refinement_seconds
                              + (travel_coarse_seconds if alpha else 0.0),
                              "refinement_seconds": refinement_seconds,
                              "waveform_normalized_mse": float(best["waveform_normalized_mse"]),
                              "arrival_normalized_mse": float(best["arrival_normalized_mse"])}
        fmm_event = {"truth": np.asarray(event_spec["truth_m_s"], dtype=np.float64)}
        started = time.perf_counter()
        branches["fmm"] = fmm_locate(
            fields, fmm_event, corrected, bounds[:3],
            float(spec["fmm_spacing_m"]), search, arrival)
        branches["fmm"]["search_seconds"] = time.perf_counter() - started
        rows.append({"event": event_spec["name"], "file": path.name,
                     "observation_sha256": event["sha256"],
                     "valid_arrival_picks": valid_count,
                     "shared_preparation_seconds": preparation_seconds,
                     "waveform_coarse_seconds": wave_coarse_seconds,
                     "arrival_coarse_seconds": travel_coarse_seconds,
                     "truth_m_s": event_spec["truth_m_s"], "branches": branches})
        print(event_spec["name"], {key: round(value["location_error_m"], 1)
                                    for key, value in branches.items()}, flush=True)
    summaries = {}
    for name in ("waveform_only", "waveform_plus_fmm", "fmm"):
        loc = np.asarray([row["branches"][name]["location_error_m"] for row in rows])
        origin = np.asarray([row["branches"][name]["origin_time_error_s"] for row in rows])
        durations = np.asarray([row["branches"][name]["search_seconds"] for row in rows])
        summaries[name] = {"n": len(loc), "mean_location_error_m": float(loc.mean()),
                            "median_location_error_m": float(np.median(loc)),
                            "std_location_error_m": float(loc.std(ddof=1)),
                            "mean_search_seconds": float(durations.mean()),
                            "median_search_seconds": float(np.median(durations)),
                            "mean_origin_time_error_s": float(origin.mean()),
                            "events_below_500_m": int(np.sum(loc < 500.0))}
    result = {"purpose": spec["purpose"],
              "prospective_blind_test": spec["prospective_blind_test"],
              "reporting_note": spec["reporting_note"],
              "protocol": str(PROTOCOL.resolve()),
              "protocol_sha256": sha256(PROTOCOL),
              "checkpoint_sha256": sha256(checkpoint),
              "velocity_sha256": spec["velocity_sha256"],
              "device": str(device), "fmm_setup_seconds": fmm_seconds,
              "fmm_cache_reused": cache_reused,
              "fmm_cache_open_seconds": cache_load_seconds,
              "rebuilt_vs_cached_max_absolute_difference_s": cache_comparison,
              "timing_note": "Search timers exclude offline FMM construction, model loading and shared event preparation. Neural search timers include coarse waveform scoring; hybrid additionally includes arrival scoring. Shared coarse scoring is attributed to both hypothetical standalone neural searches. No branch-specific GPU memory was measured.",
              "environment": {"python": platform.python_version(),
                              "numpy": np.__version__, "torch": torch.__version__,
                              "cuda": torch.version.cuda, "cpu": platform.processor(),
                              "platform": platform.platform(),
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "trainable_architecture_parameters": sum(p.numel() for p in model.parameters()),
              "cuda_peak_allocated_mib_during_evaluation": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
              "fmm_fields_shape": list(fields.shape),
              "fmm_fields_mib": fields.nbytes / 2**20,
              "evaluation_seconds": time.perf_counter() - started_all,
              "summary": summaries, "events": rows,
              "limitations": "Four synthetic events, one receiver geometry, one frozen Overthrust-trained operator, and an arrival-pick bias calibrated on Overthrust. Existing event records had prior use in other model audits.",
              "loaded_checkpoint_training_record": checkpoint_payload.get("training_seconds")}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Low-rank operator cross-geology audit", "",
             spec["reporting_note"], "",
             "| Method | n | Median eloc (m) | Mean eloc (m) | SD eloc (m) | Mean et (s) | <500 m |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("waveform_only", "Waveform only"),
                        ("waveform_plus_fmm", "Waveform + FMM arrival"),
                        ("fmm", "FMM arrival baseline")):
        row = summaries[name]
        lines.append(f"| {label} | {row['n']} | {row['median_location_error_m']:.1f} | {row['mean_location_error_m']:.1f} | {row['std_location_error_m']:.1f} | {row['mean_origin_time_error_s']:.4f} | {row['events_below_500_m']}/{row['n']} |")
    lines.extend(["", "## Event errors", "",
                  "| Event | Valid picks | Waveform eloc (m) | Hybrid eloc (m) | FMM eloc (m) | Hybrid et (s) |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        b = row["branches"]
        lines.append(f"| {row['event']} | {row['valid_arrival_picks']} | {b['waveform_only']['location_error_m']:.1f} | {b['waveform_plus_fmm']['location_error_m']:.1f} | {b['fmm']['location_error_m']:.1f} | {b['waveform_plus_fmm']['origin_time_error_s']:.4f} |")
    setup_note = (f"Fresh FMM field construction: {fmm_seconds:.2f} s (CPU solver)."
                  if fmm_seconds is not None else
                  "FMM fields reused from cache; construction time was not measured in this run.")
    lines.extend(["", setup_note, "",
                  "## Search cost", "", result["timing_note"], "",
                  "| Method | Mean search (s/event) | Median search (s/event) |",
                  "|---|---:|---:|"])
    for name in ("waveform_only", "waveform_plus_fmm", "fmm"):
        row = summaries[name]
        lines.append(f"| {name} | {row['mean_search_seconds']:.4f} | {row['median_search_seconds']:.4f} |")
    lines.extend(["", f"Environment: {result['environment']}", "",
                  f"Architecture parameters: {result['trainable_architecture_parameters']}; peak CUDA allocation during joint evaluation: {result['cuda_peak_allocated_mib_during_evaluation']} MiB.", "",
                  f"Rebuilt/cache maximum absolute difference: {cache_comparison} s.", "",
                  "Interpretation: this four-event post-hoc audit tests transfer of the new operator to a distinct synthetic geology. It is not a prospective blind test and cannot support a broad generalization claim.", "",
                  result["limitations"]])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
