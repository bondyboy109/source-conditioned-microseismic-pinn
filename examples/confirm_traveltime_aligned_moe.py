"""Run the one-time prospective confirmation of the FMM-aligned MoE."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.invert_fmm_receiver_fields import (
    interpolate_fields,
    score_candidates,
    spatial_grid,
)
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores,
    choose_diverse_starts,
    coarse_grid,
    load_event,
    load_model,
    pick_arrivals,
    refine,
    sampled_waveforms,
    sha256,
    summarize,
    waveform_scores,
)

PROTOCOL = ROOT / "experiments/traveltime_aligned_moe_confirmation_protocol.json"


def verify_frozen_inputs(spec: dict) -> dict:
    checks = (
        ("model_checkpoint", "model_checkpoint_sha256"),
        ("receiver_fmm_fields", "receiver_fmm_fields_sha256"),
        ("localization_protocol", "localization_protocol_sha256"),
        ("localization_development_result", "localization_development_result_sha256"),
        ("frozen_localization_implementation",
         "frozen_localization_implementation_sha256"),
    )
    for path_key, hash_key in checks:
        if sha256(ROOT / spec[path_key]) != spec[hash_key]:
            raise ValueError(f"Frozen input changed: {spec[path_key]}")
    development = json.loads(
        (ROOT / spec["localization_development_result"]).read_text(encoding="utf-8"))
    if not development["all_development_gates_passed"]:
        raise ValueError("Localization development did not pass all frozen gates")
    localization = json.loads(
        (ROOT / spec["localization_protocol"]).read_text(encoding="utf-8"))
    if localization["model_checkpoint_sha256"] != spec["model_checkpoint_sha256"]:
        raise ValueError("Confirmation and development model hashes disagree")
    return localization


def verify_unused_sources(spec: dict, target: Path) -> None:
    proposed = [np.asarray(row["source_m_s"], dtype=np.float64)
                for row in spec["events"]]
    if len({tuple(source) for source in proposed}) != len(proposed):
        raise ValueError("Confirmation sources are not unique")
    for path in (ROOT / "data/observations").rglob("*.npz"):
        if target in path.resolve().parents:
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
            raise ValueError(f"Confirmation source already exists: {path}")


def prepare_events(spec: dict) -> list[Path]:
    target = (ROOT / spec["confirmation_directory"]).resolve()
    existing = list(target.glob("*.npz")) if target.exists() else []
    if existing:
        paths = [target / row["file"] for row in spec["events"]]
        if len(existing) != len(paths) or not all(path.is_file() for path in paths):
            raise ValueError("Confirmation directory is partially populated")
    else:
        verify_unused_sources(spec, target)
        target.mkdir(parents=True, exist_ok=True)
        paths = []
        for row in spec["events"]:
            path = target / row["file"]
            subprocess.run([
                sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                "--output", str(path), "--duration", "2", "--source",
                *(str(value) for value in row["source_m_s"]),
            ], cwd=ROOT, check=True)
            paths.append(path)
    expected = spec["fd_configuration"]
    for row, path in zip(spec["events"], paths):
        with np.load(path) as archive:
            if (archive["data"].shape != (expected["receiver_count"], 1001)
                    or not np.isfinite(archive["data"]).all()
                    or not np.allclose(archive["source_truth"], row["source_m_s"],
                                       rtol=0, atol=1e-5)
                    or not np.isclose(archive["times"][1] - archive["times"][0],
                                      expected["time_step_s"])):
                raise ValueError(f"Invalid confirmation record: {path}")
        provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (provenance["frequency_hz"] != expected["frequency_hz"]
                or provenance["duration_s"] != expected["duration_s"]
                or provenance["boundary"] != expected["boundary"]):
            raise ValueError(f"FD provenance changed: {path}")
    return paths


def fmm_baseline(fields: np.ndarray, corrected_picks: np.ndarray,
                 bounds: np.ndarray, spec: dict) -> dict:
    baseline = spec["fmm_baseline"]
    spacing = float(baseline["grid_spacing_m"])
    candidates = spatial_grid(bounds[:3], tuple(baseline["coarse_grid_xyz"]))
    losses, origins = score_candidates(
        fields, candidates, corrected_picks, spacing,
        tuple(baseline["origin_time_bounds_s"]))
    best_index = int(np.argmin(losses))
    best = (float(losses[best_index]), candidates[best_index].copy(),
            float(origins[best_index]))
    steps = ((bounds[:3, 1] - bounds[:3, 0])
             / (np.asarray(baseline["coarse_grid_xyz"]) - 1))
    assessed = len(candidates)
    count = int(baseline["refinement_grid_xyz"])
    for _ in range(int(baseline["refinement_levels"])):
        low = np.maximum(best[1] - steps, bounds[:3, 0])
        high = np.minimum(best[1] + steps, bounds[:3, 1])
        candidates = spatial_grid(np.stack((low, high), axis=1), (count,) * 3)
        losses, origins = score_candidates(
            fields, candidates, corrected_picks, spacing,
            tuple(baseline["origin_time_bounds_s"]))
        index = int(np.argmin(losses))
        if losses[index] < best[0]:
            best = (float(losses[index]), candidates[index].copy(),
                    float(origins[index]))
        steps *= 2.0 / (count - 1)
        assessed += len(candidates)
    return {"estimated_source_m_s": np.r_[best[1], best[2]].tolist(),
            "arrival_rmse_s": float(np.sqrt(best[0])),
            "candidate_count": assessed}


def report(path: Path, result: dict) -> None:
    summary, gates = result["summary"], result["confirmation_gate_results"]
    lines = [
        "# FMM-aligned MoE prospective confirmation", "",
        "The frozen method was evaluated once on eight source positions absent from all earlier data.", "",
        "| Metric | Waveform only | Hybrid | FMM baseline |", "|---|---:|---:|---:|",
        f"| Median location error (m) | {summary['waveform_only']['median_location_error_m']:.1f} | {summary['hybrid']['median_location_error_m']:.1f} | {summary['fmm']['median_location_error_m']:.1f} |",
        f"| Mean location error (m) | {summary['waveform_only']['mean_location_error_m']:.1f} | {summary['hybrid']['mean_location_error_m']:.1f} | {summary['fmm']['mean_location_error_m']:.1f} |",
        f"| Median origin-time error (s) | {summary['waveform_only']['median_origin_time_error_s']:.4f} | {summary['hybrid']['median_origin_time_error_s']:.4f} | {summary['fmm']['median_origin_time_error_s']:.4f} |",
        f"| Events below 500 m | {summary['waveform_only']['events_below_500_m']}/8 | {summary['hybrid']['events_below_500_m']}/8 | {summary['fmm']['events_below_500_m']}/8 |",
        "", "## Predeclared gates", "", "| Gate | Result |", "|---|---|",
    ]
    lines.extend(f"| {key} | {'PASS' if value else 'FAIL'} |"
                 for key, value in gates.items())
    lines.extend(["", f"All gates passed: **{result['all_confirmation_gates_passed']}**.",
                  "", f"Limitations: {result['limitations']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if not spec["protocol_written_before_fd_generation"]:
        raise ValueError("Confirmation protocol was not registered before FD generation")
    localization = verify_frozen_inputs(spec)
    output = ROOT / spec["output_result"]
    markdown = ROOT / spec["output_report"]
    if output.exists() or markdown.exists():
        raise FileExistsError("Refusing to overwrite prospective confirmation")
    target = (ROOT / spec["confirmation_directory"]).resolve()
    if not target.exists() or not list(target.glob("*.npz")):
        verify_unused_sources(spec, target)
    if args.preflight:
        print("Preflight passed: frozen inputs match and all eight sources are unused")
        return

    paths = prepare_events(spec)
    first = load_event(paths[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain = load_model(localization, first["receivers"], device)
    fields = np.load(ROOT / spec["receiver_fmm_fields"], mmap_mode="r")
    bounds = np.asarray(localization["source_bounds_m_s"], dtype=np.float32)
    search = localization["search"]
    arrival_spec = localization["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    total_search_seconds = 0.0
    for path in paths:
        event = load_event(path, first["receivers"])
        raw_picks, corrected_picks = pick_arrivals(event, arrival_spec)
        valid_count = int(np.isfinite(corrected_picks).sum())
        if valid_count < int(arrival_spec["minimum_valid_picks"]):
            raise ValueError(f"Only {valid_count} usable picks for {path}")
        coarse_sampled = sampled_waveforms(
            event, domain, int(search["coarse_receiver_stride"]),
            int(search["coarse_time_stride"]), device)
        refined_sampled = sampled_waveforms(
            event, domain, int(search["refine_receiver_stride"]),
            int(search["refine_time_stride"]), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        waveform = waveform_scores(model, candidates, domain, coarse_sampled,
                                   int(search["candidate_batch"]))
        arrival = arrival_scores(
            model, candidates, corrected_picks,
            float(arrival_spec["arrival_scale_s"]), int(search["candidate_batch"]))
        branches = {}
        for name, alpha in (("waveform_only", 0.0),
                            ("hybrid", float(arrival_spec["hybrid_weight"]))):
            total = waveform + alpha * arrival
            indices = choose_diverse_starts(
                candidates, total, int(search["restarts"]),
                float(search["minimum_start_separation_m"]))
            starts = [{"source_m_s": candidates[index].tolist(),
                       "coarse_total_loss": float(total[index])}
                      for index in indices]
            winner, refinements = refine(
                model, starts, domain, bounds, refined_sampled, corrected_picks,
                search, arrival_spec, alpha)
            branches[name] = {
                "estimated_source_m_s": winner["estimated_source_m_s"],
                "total_loss": winner["total_loss"],
                "waveform_normalized_mse": winner["waveform_normalized_mse"],
                "arrival_normalized_mse": winner["arrival_normalized_mse"],
                "selected_spatial_move_m": winner["spatial_move_m"],
                "selected_on_boundary": winner["on_boundary"],
                "starts": starts, "refinements": refinements}
        pure_fmm = fmm_baseline(fields, corrected_picks, bounds, spec)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        total_search_seconds += elapsed
        # Open truth only after waveform, hybrid and FMM searches are complete.
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        for branch in (*branches.values(), pure_fmm):
            estimate = np.asarray(branch["estimated_source_m_s"], dtype=np.float64)
            error = estimate - truth
            branch["component_error_m_s"] = error.tolist()
            branch["location_error_m"] = float(np.linalg.norm(error[:3]))
            branch["origin_time_error_s"] = float(abs(error[3]))
        provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        row = {"event": path.name, "observation": str(path),
               "sha256": sha256(path), "source_truth_m_s_post_search": truth.tolist(),
               "valid_raw_picks": valid_count,
               "raw_picks_s": [float(value) if np.isfinite(value) else None
                               for value in raw_picks],
               "corrected_picks_s": [float(value) if np.isfinite(value) else None
                                     for value in corrected_picks],
               "fd_generation_seconds": provenance["runtime_seconds"],
               "search_seconds": elapsed, **branches, "fmm": pure_fmm}
        rows.append(row)
        print(f"{path.name}: waveform={branches['waveform_only']['location_error_m']:.1f} m, "
              f"hybrid={branches['hybrid']['location_error_m']:.1f} m, "
              f"FMM={pure_fmm['location_error_m']:.1f} m, "
              f"et={branches['hybrid']['origin_time_error_s']:.4f} s", flush=True)

    summary = {name: summarize(rows, name)
               for name in ("waveform_only", "hybrid", "fmm")}
    improved = sum(row["hybrid"]["location_error_m"]
                   < row["waveform_only"]["location_error_m"] for row in rows)
    summary["hybrid_improved_location_events"] = int(improved)
    thresholds = spec["confirmation_gates"]
    gates = {
        "all_events_have_at_least_six_valid_picks": bool(all(
            row["valid_raw_picks"] >= int(arrival_spec["minimum_valid_picks"])
            for row in rows)),
        "hybrid_median_location_below_500m": bool(
            summary["hybrid"]["median_location_error_m"]
            < thresholds["hybrid_median_location_error_m_below"]),
        "hybrid_at_least_five_events_below_500m": bool(
            summary["hybrid"]["events_below_500_m"]
            >= thresholds["hybrid_location_error_below_500_m_events_at_least"]),
        "hybrid_median_origin_time_below_0_05s": bool(
            summary["hybrid"]["median_origin_time_error_s"]
            < thresholds["hybrid_median_origin_time_error_s_below"]),
        "hybrid_at_least_six_origin_times_below_0_05s": bool(
            summary["hybrid"]["events_origin_below_0_05_s"]
            >= thresholds["hybrid_origin_time_error_below_0_05_s_events_at_least"]),
        "hybrid_median_location_not_worse_than_waveform_only": bool(
            summary["hybrid"]["median_location_error_m"]
            <= summary["waveform_only"]["median_location_error_m"]),
        "hybrid_improves_over_waveform_in_at_least_six_events": bool(
            improved
            >= thresholds["hybrid_location_improves_over_waveform_events_at_least"]),
    }
    passed = bool(all(gates.values()))
    result = {
        "purpose": spec["purpose"], "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL), "independent_confirmation": True,
        "checkpoint_sha256": spec["model_checkpoint_sha256"],
        "event_count": len(rows), "device": str(device),
        "candidate_count_per_neural_search": len(candidates),
        "fd_generation_seconds": float(sum(row["fd_generation_seconds"] for row in rows)),
        "total_search_seconds": total_search_seconds, "summary": summary,
        "confirmation_gate_results": gates,
        "all_confirmation_gates_passed": passed, "events": rows,
        "limitations": spec["limitations"],
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report(markdown, result)
    print(json.dumps({"summary": summary, "confirmation_gate_results": gates,
                      "all_confirmation_gates_passed": passed,
                      "fd_generation_seconds": result["fd_generation_seconds"],
                      "total_search_seconds": total_search_seconds}, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
