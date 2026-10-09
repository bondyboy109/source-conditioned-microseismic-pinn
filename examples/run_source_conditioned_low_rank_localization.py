"""One-time localization test for the frozen low-rank source operator."""

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

from examples.run_traveltime_aligned_moe_localization import (
    coarse_grid, load_event, objective_batch, pick_arrivals, refine,
    sampled_waveforms, choose_diverse_starts, waveform_scores, arrival_scores,
)
from examples.run_holdout_noise_picker_comparison import robust_arrival  # noqa: F401
from source_conditioned_low_rank import SourceConditionedLowRankOperator
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator

PROTOCOL = ROOT / "experiments/source_conditioned_low_rank_localization_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(spec: dict, receivers: np.ndarray, device: torch.device):
    checkpoint_path = ROOT / spec["model_checkpoint"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_spec = payload["protocol"]["model"]
    fmm_spec = payload["protocol"]["fmm"]
    field_path = ROOT / fmm_spec["cache_file"]
    fields = np.load(field_path, mmap_mode="r")
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)),
        float(fmm_spec["grid_spacing_m"])).to(device)
    model = SourceConditionedLowRankOperator(
        interpolator, torch.from_numpy(receivers),
        tuple(model_spec["spatial_lengths_m"]),
        time_length_s=float(model_spec["time_length_s"]),
        frequency_hz=float(model_spec["frequency_hz"]),
        rank=int(model_spec["rank"]), hidden_dim=int(model_spec["hidden_dim"]),
        hidden_layers=int(model_spec["hidden_layers"]),
        causal_width_s=float(model_spec["causal_width_s"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval().requires_grad_(False)
    domain = np.asarray([*model_spec["spatial_lengths_m"],
                         model_spec["time_length_s"]], dtype=np.float32)
    return model, domain


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["output_result"]
    report = ROOT / spec["output_report"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite locked localization results")
    checkpoint_path = ROOT / spec["model_checkpoint"]
    forward_path = ROOT / spec["forward_development_result"]
    if not json.loads(forward_path.read_text(encoding="utf-8"))["all_development_gates_passed"]:
        raise ValueError("Forward development gates did not pass")
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint_payload["protocol_sha256"] != sha256(
            ROOT / "experiments/source_conditioned_low_rank_development_protocol.json"):
        raise ValueError("Development checkpoint protocol hash mismatch")
    paths = sorted((ROOT / spec["confirmation_directory"]).glob("*.npz"))
    if len(paths) != int(spec["confirmation_event_count"]):
        raise ValueError("Locked event set changed")
    first = load_event(paths[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain = load_model(spec, first["receivers"], device)
    bounds = np.asarray(spec["shared_source_bounds_m_s"], dtype=np.float32)
    search, arrival = spec["search"], spec["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    total_seconds = 0.0
    for path in paths:
        event = load_event(path, first["receivers"])
        # Keep the established robust picker and bias correction fixed.
        data, times = event["data"], event["times"]
        raw_picks = np.asarray([robust_arrival(trace, times) for trace in data],
                               dtype=np.float32)
        peaks = np.max(np.abs(data), axis=1)
        valid = (np.isfinite(raw_picks)
                 & (peaks >= arrival["minimum_receiver_peak_fraction_of_record"]
                    * peaks.max())
                 & (raw_picks <= times[-1]
                    - arrival["last_record_s_excluded_for_pick"]))
        corrected = raw_picks - float(arrival["pick_bias_s"])
        corrected[~valid] = np.nan
        valid_count = int(np.isfinite(corrected).sum())
        arrival_available = valid_count >= int(arrival["minimum_valid_picks"])
        coarse = sampled_waveforms(
            event, domain, int(search["coarse_receiver_stride"]),
            int(search["coarse_time_stride"]), device)
        fine = sampled_waveforms(
            event, domain, int(search["refine_receiver_stride"]),
            int(search["refine_time_stride"]), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        wave = waveform_scores(model, candidates, domain, coarse,
                               int(search["candidate_batch"]))
        travel = (arrival_scores(model, candidates, corrected,
                                 float(arrival["arrival_scale_s"]),
                                 int(search["candidate_batch"]))
                  if arrival_available else np.full(len(candidates), np.inf))
        branches = {}
        branch_specs = [("waveform_only", 0.0)]
        if arrival_available:
            branch_specs.append(("waveform_plus_fmm",
                                 float(arrival["hybrid_weight"])))
        for name, alpha in branch_specs:
            # Keep the waveform-only branch independent of unavailable arrival
            # values.  Otherwise ``0 * inf`` can create NaNs in the coarse
            # ranking even though the arrival term is disabled.
            score = wave if alpha == 0.0 else wave + alpha * travel
            selected = choose_diverse_starts(
                candidates, score, int(search["restarts"]),
                float(search["minimum_start_separation_m"]))
            starts = [{"source_m_s": candidates[i].tolist(),
                       "coarse_total_loss": float(score[i])} for i in selected]
            winner, refinements = refine(
                model, starts, domain, bounds, fine, corrected, search, arrival, alpha)
            branches[name] = {"estimated_source_m_s": winner["estimated_source_m_s"],
                              "total_loss": winner["total_loss"],
                              "waveform_normalized_mse": winner["waveform_normalized_mse"],
                              "arrival_normalized_mse": winner["arrival_normalized_mse"],
                              "selected_spatial_move_m": winner["spatial_move_m"],
                              "starts": starts, "refinements": refinements}
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        # Read the truth only after both searches have completed.
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        for result in branches.values():
            estimate = np.asarray(result["estimated_source_m_s"], dtype=np.float64)
            error = estimate - truth
            result["component_error_m_s"] = error.tolist()
            result["location_error_m"] = float(np.linalg.norm(error[:3]))
            result["origin_time_error_s"] = float(abs(error[3]))
        row = {"event": path.name, "observation_sha256": event["sha256"],
               "truth_m_s_post_search": truth.tolist(), "valid_arrival_picks": valid_count,
               "search_seconds": elapsed, **branches}
        rows.append(row)
        hybrid_text = (f"hybrid={branches['waveform_plus_fmm']['location_error_m']:.1f} m, "
                       f"et={branches['waveform_plus_fmm']['origin_time_error_s']:.4f} s"
                       if arrival_available else "hybrid=NA (insufficient picks)")
        print(f"{path.name}: waveform={branches['waveform_only']['location_error_m']:.1f} m, "
              f"{hybrid_text}, "
              f"runtime={elapsed:.1f} s", flush=True)
    summaries = {}
    for name in ("waveform_only", "waveform_plus_fmm"):
        available = [row for row in rows if name in row]
        locations = np.asarray([row[name]["location_error_m"] for row in available])
        origins = np.asarray([row[name]["origin_time_error_s"] for row in available])
        summaries[name] = {"mean_location_error_m": float(locations.mean()),
                           "median_location_error_m": float(np.median(locations)),
                           "events_below_500_m": int(np.sum(locations < 500)),
                           "mean_origin_time_error_s": float(origins.mean()),
                           "median_origin_time_error_s": float(np.median(origins)),
                           "event_count": len(available)}
    hybrid_rows = [row for row in rows if "waveform_plus_fmm" in row]
    improved = sum(row["waveform_plus_fmm"]["location_error_m"]
                   < row["waveform_only"]["location_error_m"] for row in hybrid_rows)
    result = {"purpose": spec["purpose"], "protocol": str(PROTOCOL.resolve()),
              "protocol_sha256": sha256(PROTOCOL),
              "checkpoint_sha256": sha256(checkpoint_path),
              "forward_development_result_sha256": sha256(forward_path),
              "device": str(device), "candidate_count_per_event": len(candidates),
              "total_search_seconds": total_seconds, "summary": summaries,
              "hybrid_improved_events": improved, "events": rows,
              "limitations": spec["limitations"]}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Source-conditioned low-rank localization result", "",
             "This is a one-time locked-event evaluation for this frozen operator. The synthetic records have been examined in studies of other models; this is not a field-data validation.", "",
             "| Objective | n / attempted | Median eloc (m) | Mean eloc (m) | Median et (s) | Mean et (s) | <500 m among available |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("waveform_only", "Waveform only"),
                        ("waveform_plus_fmm", "Waveform + FMM arrival")):
        row = summaries[name]
        lines.append(f"| {label} | {row['event_count']}/{len(rows)} | {row['median_location_error_m']:.1f} | {row['mean_location_error_m']:.1f} | {row['median_origin_time_error_s']:.4f} | {row['mean_origin_time_error_s']:.4f} | {row['events_below_500_m']}/{row['event_count']} |")
    lines.extend(["", f"FMM arrival constraint improved spatial error in {improved}/{len(hybrid_rows)} paired available events. Error summaries exclude events without usable arrival constraints.", "",
                  "## Event results", "", "| Event | eloc waveform (m) | eloc hybrid (m) | et hybrid (s) | Runtime (s) |",
                  "|---|---:|---:|---:|---:|"])
    for row in rows:
        if "waveform_plus_fmm" in row:
            hybrid_error = f"{row['waveform_plus_fmm']['location_error_m']:.1f}"
            hybrid_time = f"{row['waveform_plus_fmm']['origin_time_error_s']:.4f}"
        else:
            hybrid_error, hybrid_time = "NA", "NA"
        lines.append(f"| {row['event']} | {row['waveform_only']['location_error_m']:.1f} | {hybrid_error} | {hybrid_time} | {row['search_seconds']:.1f} |")
    lines.extend(["", spec["limitations"]])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summaries, "hybrid_improved_events": improved,
                      "total_search_seconds": total_seconds}, indent=2), flush=True)


if __name__ == "__main__":
    main()
