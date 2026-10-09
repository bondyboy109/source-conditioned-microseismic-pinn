"""Stress-test the frozen FMM-aligned MoE with noise and missing receivers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.confirm_traveltime_aligned_moe import (
    fmm_baseline,
    verify_frozen_inputs,
)
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores,
    choose_diverse_starts,
    coarse_grid,
    load_event,
    load_model,
    pick_arrivals,
    refine,
    sha256,
    waveform_scores,
)

PROTOCOL = ROOT / "experiments/traveltime_aligned_moe_stress_protocol.json"


def sampled_waveforms_available(event: dict, domain: np.ndarray,
                                available_ids: np.ndarray,
                                receiver_stride: int, time_stride: int,
                                device: torch.device):
    """Sample active receivers while preserving their global receiver IDs."""
    receiver_ids = np.asarray(available_ids, dtype=np.int64)[::receiver_stride]
    time_ids = np.arange(len(event["times"]))[::time_stride]
    receivers = event["receivers"][receiver_ids]
    times = event["times"][time_ids]
    nr, nt = len(receivers), len(times)
    points = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (nr, nt, 3)),
        np.broadcast_to(times[None, :, None], (nr, nt, 1))), axis=-1)
    points = torch.as_tensor(points.reshape(-1, 4) / domain,
                             device=device, dtype=torch.float32)
    receiver_index = torch.as_tensor(
        np.repeat(receiver_ids, nt), device=device, dtype=torch.long)
    active_rms = float(np.sqrt(np.mean(
        event["data"][available_ids].astype(np.float64) ** 2)))
    if not np.isfinite(active_rms) or active_rms <= 0:
        raise ValueError("Active receiver data have invalid RMS")
    observed = torch.as_tensor(
        event["data"][np.ix_(receiver_ids, time_ids)] / active_rms,
        device=device, dtype=torch.float32).reshape(-1, 1)
    return points, receiver_index, observed


def add_exact_awgn(data: np.ndarray, snr_db: float,
                   seed: int) -> tuple[np.ndarray, float]:
    """Add Gaussian noise scaled to the requested whole-record RMS SNR."""
    signal_rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(data.shape)
    noise *= target_noise_rms / np.sqrt(np.mean(noise ** 2))
    noisy = data.astype(np.float64) + noise
    measured = 20.0 * np.log10(signal_rms / np.sqrt(np.mean(noise ** 2)))
    return noisy.astype(np.float32), float(measured)


def retained_receivers(receiver_count: int, missing_fraction: float,
                       seed: int) -> np.ndarray:
    keep_count = int(round(receiver_count * (1.0 - missing_fraction)))
    if not 1 <= keep_count <= receiver_count:
        raise ValueError("Missing-receiver fraction leaves an invalid count")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(receiver_count, keep_count, replace=False))


def branch_result(model, candidates: np.ndarray, waveform: np.ndarray,
                  arrival: np.ndarray, alpha: float, domain: np.ndarray,
                  bounds: np.ndarray, refined_sampled: tuple,
                  corrected_picks: np.ndarray, search: dict,
                  arrival_spec: dict) -> dict:
    total = waveform + alpha * arrival
    indices = choose_diverse_starts(
        candidates, total, int(search["restarts"]),
        float(search["minimum_start_separation_m"]))
    starts = [{"source_m_s": candidates[index].tolist(),
               "coarse_total_loss": float(total[index])}
              for index in indices]
    winner, _ = refine(model, starts, domain, bounds, refined_sampled,
                       corrected_picks, search, arrival_spec, alpha)
    return {
        "estimated_source_m_s": winner["estimated_source_m_s"],
        "total_loss": winner["total_loss"],
        "waveform_normalized_mse": winner["waveform_normalized_mse"],
        "arrival_normalized_mse": winner["arrival_normalized_mse"],
        "selected_spatial_move_m": winner["spatial_move_m"],
        "selected_on_boundary": winner["on_boundary"],
    }


def condition_rows(spec: dict, localization: dict, paths: list[Path],
                   model, domain: np.ndarray, fields: np.ndarray,
                   condition: dict, device: torch.device) -> list[dict]:
    bounds = np.asarray(localization["source_bounds_m_s"], dtype=np.float32)
    search = localization["search"]
    arrival_spec = localization["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    receiver_count = model.interpolator.receiver_count
    if condition["kind"] == "missing_receivers":
        available = retained_receivers(
            receiver_count, float(condition["missing_fraction"]),
            int(condition["seed"]))
    else:
        available = np.arange(receiver_count, dtype=np.int64)

    for event_index, path in enumerate(paths):
        clean = load_event(path, model.receiver_xyz_m.cpu().numpy())
        event = dict(clean)
        measured_snr = None
        if condition["kind"] == "noise":
            noisy, measured_snr = add_exact_awgn(
                clean["data"], float(condition["snr_db"]),
                int(condition["seed"]) + 1009 * event_index)
            event["data"] = noisy
        event["rms"] = float(np.sqrt(np.mean(
            event["data"].astype(np.float64) ** 2)))

        raw_picks, corrected = pick_arrivals(event, arrival_spec)
        inactive = np.ones(receiver_count, dtype=bool)
        inactive[available] = False
        corrected[inactive] = np.nan
        raw_picks[inactive] = np.nan
        valid_count = int(np.isfinite(corrected).sum())
        row = {
            "condition_id": condition["id"],
            "condition_kind": condition["kind"],
            "level": condition["level"],
            "realization": condition["realization"],
            "seed": condition["seed"],
            "event": path.name,
            "observation_sha256": sha256(path),
            "available_receiver_count": int(len(available)),
            "available_receiver_ids": available.tolist(),
            "valid_pick_count": valid_count,
            "measured_snr_db": measured_snr,
        }
        if valid_count < int(arrival_spec["minimum_valid_picks"]):
            row.update({"status": "insufficient_valid_picks",
                        "waveform_only": None, "hybrid": None, "fmm": None})
            rows.append(row)
            print(f"{condition['id']} {path.name}: insufficient picks "
                  f"({valid_count})", flush=True)
            continue

        coarse_sampled = sampled_waveforms_available(
            event, domain, available, int(search["coarse_receiver_stride"]),
            int(search["coarse_time_stride"]), device)
        refined_sampled = sampled_waveforms_available(
            event, domain, available, int(search["refine_receiver_stride"]),
            int(search["refine_time_stride"]), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        waveform = waveform_scores(model, candidates, domain, coarse_sampled,
                                   int(search["candidate_batch"]))
        arrival = arrival_scores(
            model, candidates, corrected,
            float(arrival_spec["arrival_scale_s"]),
            int(search["candidate_batch"]))
        branches = {
            "waveform_only": branch_result(
                model, candidates, waveform, arrival, 0.0, domain, bounds,
                refined_sampled, corrected, search, arrival_spec),
            "hybrid": branch_result(
                model, candidates, waveform, arrival,
                float(arrival_spec["hybrid_weight"]), domain, bounds,
                refined_sampled, corrected, search, arrival_spec),
        }
        pure_fmm = fmm_baseline(fields, corrected, bounds, spec)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        for branch in (*branches.values(), pure_fmm):
            estimate = np.asarray(branch["estimated_source_m_s"], dtype=np.float64)
            error = estimate - truth
            branch["component_error_m_s"] = error.tolist()
            branch["location_error_m"] = float(np.linalg.norm(error[:3]))
            branch["origin_time_error_s"] = float(abs(error[3]))
        row.update({"status": "ok", "source_truth_m_s_post_search": truth.tolist(),
                    "search_seconds": elapsed, **branches, "fmm": pure_fmm})
        rows.append(row)
        print(f"{condition['id']} {path.name}: "
              f"hybrid={branches['hybrid']['location_error_m']:.1f} m, "
              f"FMM={pure_fmm['location_error_m']:.1f} m, "
              f"picks={valid_count}, time={elapsed:.1f} s", flush=True)
    return rows


def summarize_branch(rows: list[dict], branch: str) -> dict:
    valid = [row[branch] for row in rows
             if row["status"] == "ok" and row[branch] is not None]
    if not valid:
        return {"successful_trials": 0, "failure_count": len(rows),
                "failure_rate": 1.0, "median_location_error_m": None,
                "mean_location_error_m": None,
                "median_origin_time_error_s": None,
                "events_below_500_m": 0}
    locations = np.asarray([row["location_error_m"] for row in valid])
    origins = np.asarray([row["origin_time_error_s"] for row in valid])
    return {
        "successful_trials": len(valid),
        "failure_count": len(rows) - len(valid),
        "failure_rate": float((len(rows) - len(valid)) / len(rows)),
        "median_location_error_m": float(np.median(locations)),
        "mean_location_error_m": float(np.mean(locations)),
        "median_origin_time_error_s": float(np.median(origins)),
        "events_below_500_m": int(np.sum(locations < 500.0)),
    }


def summarize(rows: list[dict], conditions: list[dict]) -> dict:
    output = {}
    for condition in conditions:
        subset = [row for row in rows if row["condition_id"] == condition["id"]]
        output[condition["id"]] = {
            "condition": condition,
            "trial_count": len(subset),
            "median_valid_pick_count": float(np.median(
                [row["valid_pick_count"] for row in subset])),
            **{name: summarize_branch(subset, name)
               for name in ("waveform_only", "hybrid", "fmm")},
        }
    grouped = {}
    for kind in ("noise", "missing_receivers"):
        for level in sorted({row["level"] for row in rows
                             if row["condition_kind"] == kind}):
            subset = [row for row in rows
                      if row["condition_kind"] == kind and row["level"] == level]
            grouped[f"{kind}:{level}"] = {
                "trial_count": len(subset),
                "median_valid_pick_count": float(np.median(
                    [row["valid_pick_count"] for row in subset])),
                **{name: summarize_branch(subset, name)
                   for name in ("waveform_only", "hybrid", "fmm")},
            }
    return {"by_realization": output, "by_level": grouped}


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# Frozen MoE noise and missing-receiver stress test", "",
        "The checkpoint, FMM fields, arrival correction, hybrid weight, and search settings were frozen.", "",
        "| Condition | Trials | Median picks | Waveform median (m) | Hybrid median (m) | FMM median (m) | Hybrid failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["summary"]["by_level"].items():
        lines.append(
            f"| {name} | {row['trial_count']} | {row['median_valid_pick_count']:.1f} | "
            f"{row['waveform_only']['median_location_error_m']:.1f} | "
            f"{row['hybrid']['median_location_error_m']:.1f} | "
            f"{row['fmm']['median_location_error_m']:.1f} | "
            f"{row['hybrid']['failure_count']} |")
    lines.extend(["", f"Runtime: {result['runtime_seconds']:.1f} s.", "",
                  f"Limitations: {result['limitations']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    script_path = Path(__file__).resolve()
    if sha256(script_path) != spec["implementation_sha256"]:
        raise ValueError("Stress-test implementation changed after registration")
    localization = verify_frozen_inputs(spec)
    clean_result = ROOT / spec["clean_confirmation_result"]
    if sha256(clean_result) != spec["clean_confirmation_result_sha256"]:
        raise ValueError("Clean confirmation result changed")
    paths = [ROOT / spec["confirmation_directory"] / row["file"]
             for row in spec["events"]]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("A confirmation observation is missing")
    output = ROOT / spec["output_result"]
    report_path = ROOT / spec["output_report"]
    if output.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite stress-test output")
    if args.preflight:
        print("Preflight passed: frozen assets and stress-test protocol match")
        return

    first = load_event(paths[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain = load_model(localization, first["receivers"], device)
    fields = np.load(ROOT / spec["receiver_fmm_fields"], mmap_mode="r")
    started = time.perf_counter()
    rows = []
    for condition in spec["conditions"]:
        rows.extend(condition_rows(spec, localization, paths, model, domain,
                                   fields, condition, device))
    runtime = time.perf_counter() - started
    result = {
        "purpose": spec["purpose"],
        "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "implementation_sha256": spec["implementation_sha256"],
        "checkpoint_sha256": spec["model_checkpoint_sha256"],
        "device": str(device),
        "clean_confirmation_result": str(clean_result.resolve()),
        "condition_count": len(spec["conditions"]),
        "trial_count": len(rows),
        "runtime_seconds": runtime,
        "summary": summarize(rows, spec["conditions"]),
        "records": rows,
        "limitations": spec["limitations"],
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)
    print(json.dumps(result["summary"]["by_level"], indent=2), flush=True)
    print(f"Saved {output}; runtime={runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
