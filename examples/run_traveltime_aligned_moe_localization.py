"""Run the registered development localization for the FMM-aligned MoE."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.run_holdout_noise_picker_comparison import robust_arrival
from traveltime_aligned_moe import (
    ReceiverTravelTimeInterpolator,
    TravelTimeAlignedPrototypeMoE,
)

PROTOCOL = ROOT / "experiments/traveltime_aligned_moe_localization_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coarse_grid(bounds: np.ndarray, shape: list[int]) -> np.ndarray:
    if bounds.shape != (4, 2) or len(shape) != 4 or min(shape) < 2:
        raise ValueError("Expected four bounds and a four-dimensional grid")
    axes = [np.linspace(low, high, count, dtype=np.float32)
            for (low, high), count in zip(bounds, shape)]
    return np.asarray(list(itertools.product(*axes)), dtype=np.float32)


def choose_diverse_starts(candidates: np.ndarray, scores: np.ndarray,
                          count: int, minimum_distance_m: float) -> list[int]:
    """Keep the lowest score per spatially separated coarse-grid site."""
    if len(candidates) != len(scores) or count < 1 or minimum_distance_m <= 0:
        raise ValueError("Invalid candidate selection inputs")
    selected: list[int] = []
    for index in np.argsort(scores, kind="stable"):
        if all(np.linalg.norm(candidates[index, :3] - candidates[old, :3])
               >= minimum_distance_m for old in selected):
            selected.append(int(index))
            if len(selected) == count:
                break
    if len(selected) < count:
        raise ValueError("Not enough spatially distinct candidates")
    return selected


def load_model(spec: dict, receivers: np.ndarray,
               device: torch.device) -> tuple[TravelTimeAlignedPrototypeMoE, np.ndarray]:
    checkpoint_path = ROOT / spec["model_checkpoint"]
    field_path = ROOT / spec["receiver_fmm_fields"]
    if (sha256(checkpoint_path) != spec["model_checkpoint_sha256"]
            or sha256(field_path) != spec["receiver_fmm_fields_sha256"]):
        raise ValueError("Frozen model or FMM cache changed")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_spec = payload["protocol"]["model"]
    fmm_spec = payload["protocol"]["fmm"]
    domain = np.asarray([*model_spec["spatial_lengths_m"],
                         model_spec["time_length_s"]], dtype=np.float32)
    fields = np.load(field_path, mmap_mode="r")
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)),
        float(fmm_spec["grid_spacing_m"])).to(device)
    model = TravelTimeAlignedPrototypeMoE(
        interpolator, torch.from_numpy(receivers),
        tuple(model_spec["spatial_lengths_m"]),
        time_length_s=float(model_spec["time_length_s"]),
        frequency_hz=float(model_spec["frequency_hz"]),
        expert_count=int(model_spec["expert_count"]),
        hidden_dim=int(model_spec["hidden_dim"]),
        hidden_layers=int(model_spec["hidden_layers"]),
        prototype_centers=torch.tensor(model_spec["prototype_centers"]),
        prototype_scales=tuple(model_spec["prototype_scales"]),
        temperature=float(model_spec["prototype_temperature"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval().requires_grad_(False)
    return model, domain


def load_event(path: Path, expected_receivers: np.ndarray | None = None) -> dict:
    with np.load(path) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
    if (data.shape != (len(receivers), len(times)) or not np.isfinite(data).all()
            or (expected_receivers is not None
                and not np.array_equal(receivers, expected_receivers))):
        raise ValueError(f"Invalid or changed observation geometry: {path}")
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError(f"Invalid observation RMS: {path}")
    return {"path": path, "data": data, "times": times,
            "receivers": receivers, "rms": rms, "sha256": sha256(path)}


def pick_arrivals(event: dict, arrival_spec: dict) -> tuple[np.ndarray, np.ndarray]:
    data, times = event["data"], event["times"]
    picks = np.asarray([robust_arrival(trace, times) for trace in data],
                       dtype=np.float32)
    peaks = np.max(np.abs(data), axis=1)
    valid = (np.isfinite(picks)
             & (peaks >= float(arrival_spec["minimum_receiver_peak_fraction_of_record"])
                * peaks.max())
             & (picks <= times[-1]
                - float(arrival_spec["last_record_s_excluded_for_pick"])))
    corrected = picks - float(arrival_spec["pick_bias_s"])
    corrected[~valid] = np.nan
    return picks, corrected


def sampled_waveforms(event: dict, domain: np.ndarray, receiver_stride: int,
                      time_stride: int, device: torch.device):
    receiver_ids = np.arange(len(event["receivers"]))[::receiver_stride]
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
    observed = torch.as_tensor(
        event["data"][np.ix_(receiver_ids, time_ids)] / event["rms"],
        device=device, dtype=torch.float32).reshape(-1, 1)
    return points, receiver_index, observed


def waveform_scores(model: TravelTimeAlignedPrototypeMoE, candidates: np.ndarray,
                    domain: np.ndarray, sampled: tuple, batch_size: int) -> np.ndarray:
    points, receiver_index, observed = sampled
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            batch = torch.as_tensor(candidates[start:start + batch_size] / domain,
                                    device=points.device, dtype=torch.float32)
            count = len(batch)
            prediction = model(
                points.repeat(count, 1),
                batch.repeat_interleave(len(points), dim=0),
                receiver_index.repeat(count))[0]
            prediction = prediction.reshape(count, len(points), 1)
            loss = (prediction - observed[None, :, :]).square().mean((1, 2))
            scores.extend(loss.cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def arrival_scores(model: TravelTimeAlignedPrototypeMoE, candidates: np.ndarray,
                   corrected_picks: np.ndarray, scale_s: float,
                   batch_size: int) -> np.ndarray:
    valid_ids = np.flatnonzero(np.isfinite(corrected_picks))
    if not len(valid_ids):
        return np.full(len(candidates), np.inf)
    scores: list[float] = []
    device = next(model.parameters()).device
    receiver_template = torch.as_tensor(valid_ids, device=device, dtype=torch.long)
    target = torch.as_tensor(corrected_picks[valid_ids], device=device,
                             dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            batch = torch.as_tensor(candidates[start:start + batch_size],
                                    device=device, dtype=torch.float32)
            count = len(batch)
            xyz = batch[:, None, :3].expand(count, len(valid_ids), 3).reshape(-1, 3)
            receiver_ids = receiver_template.repeat(count)
            travel = model.interpolator(xyz, receiver_ids).reshape(count, -1)
            predicted = travel + batch[:, 3:4]
            loss = ((predicted - target[None, :]) / scale_s).square().mean(1)
            scores.extend(loss.cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def objective_batch(model: TravelTimeAlignedPrototypeMoE,
                    source_normalized: torch.Tensor,
                    domain_t: torch.Tensor, sampled: tuple,
                    corrected_picks: np.ndarray,
                    arrival_scale_s: float, alpha: float):
    points, receiver_index, observed = sampled
    count = len(source_normalized)
    prediction = model(
        points.repeat(count, 1),
        source_normalized.repeat_interleave(len(points), dim=0),
        receiver_index.repeat(count))[0].reshape(count, len(points), 1)
    waveform = (prediction - observed[None, :, :]).square().mean((1, 2))
    valid_ids_np = np.flatnonzero(np.isfinite(corrected_picks))
    if len(valid_ids_np):
        valid_ids = torch.as_tensor(valid_ids_np, device=points.device,
                                    dtype=torch.long)
        xyz_m = (source_normalized[:, None, :3] * domain_t[None, None, :3])
        xyz_m = xyz_m.expand(count, len(valid_ids), 3).reshape(-1, 3)
        travel = model.interpolator(xyz_m, valid_ids.repeat(count)).reshape(count, -1)
        picks = torch.as_tensor(corrected_picks[valid_ids_np], device=points.device,
                                dtype=torch.float32)
        predicted_arrival = travel + source_normalized[:, 3:4] * domain_t[3]
        arrival = ((predicted_arrival - picks[None, :])
                   / arrival_scale_s).square().mean(1)
    else:
        arrival = torch.zeros_like(waveform)
    return waveform + alpha * arrival, waveform, arrival


def refine(model: TravelTimeAlignedPrototypeMoE, starts: list[dict],
           domain: np.ndarray, bounds: np.ndarray, sampled: tuple,
           corrected_picks: np.ndarray, search_spec: dict,
           arrival_spec: dict, alpha: float) -> tuple[dict, list[dict]]:
    device = sampled[0].device
    domain_t = torch.as_tensor(domain, device=device)
    low = torch.as_tensor(bounds[:, 0] / domain, device=device)
    high = torch.as_tensor(bounds[:, 1] / domain, device=device)
    initial_physical = np.asarray([start["source_m_s"] for start in starts],
                                  dtype=np.float32)
    initial = torch.as_tensor(initial_physical / domain, device=device)
    fraction = torch.nn.Parameter(((initial - low) / (high - low)).clamp(0, 1))
    optimizer = torch.optim.Adam(
        (fraction,),
        lr=float(search_spec["refine_adam_learning_rate_on_box_fractions"]))
    best_total = torch.full((len(starts),), float("inf"), device=device)
    best_waveform = torch.full_like(best_total, float("inf"))
    best_arrival = torch.full_like(best_total, float("inf"))
    best_source = initial.detach().clone()
    for _ in range(int(search_spec["refine_steps"])):
        optimizer.zero_grad(set_to_none=True)
        source = low + (high - low) * fraction
        total, waveform, arrival = objective_batch(
            model, source, domain_t, sampled, corrected_picks,
            float(arrival_spec["arrival_scale_s"]), alpha)
        with torch.no_grad():
            improved = total < best_total
            best_total[improved] = total[improved]
            best_waveform[improved] = waveform[improved]
            best_arrival[improved] = arrival[improved]
            best_source[improved] = source[improved]
        total.sum().backward()
        optimizer.step()
        with torch.no_grad():
            fraction.clamp_(0, 1)
    estimates = (best_source.cpu().numpy() * domain).astype(np.float64)
    results = []
    for index, (start, estimate) in enumerate(zip(starts, estimates)):
        initial_physical = np.asarray(start["source_m_s"], dtype=np.float64)
        results.append({
            "initial_source_m_s": initial_physical.tolist(),
            "coarse_total_loss": float(start["coarse_total_loss"]),
            "estimated_source_m_s": estimate.tolist(),
            "total_loss": float(best_total[index].cpu()),
            "waveform_normalized_mse": float(best_waveform[index].cpu()),
            "arrival_normalized_mse": float(best_arrival[index].cpu()),
            "spatial_move_m": float(np.linalg.norm(estimate[:3]
                                                    - initial_physical[:3])),
            "on_boundary": bool(np.any(np.isclose(
                estimate, bounds[:, 0], atol=[1, 1, 1, .001])
                | np.isclose(estimate, bounds[:, 1], atol=[1, 1, 1, .001]))),
        })
    return min(results, key=lambda row: row["total_loss"]), results


def summarize(rows: list[dict], branch: str) -> dict:
    locations = np.asarray([row[branch]["location_error_m"] for row in rows])
    origins = np.asarray([row[branch]["origin_time_error_s"] for row in rows])
    return {
        "mean_location_error_m": float(locations.mean()),
        "median_location_error_m": float(np.median(locations)),
        "events_below_500_m": int(np.sum(locations < 500)),
        "mean_origin_time_error_s": float(origins.mean()),
        "median_origin_time_error_s": float(np.median(origins)),
        "events_origin_below_0_05_s": int(np.sum(origins < .05)),
    }


def write_report(path: Path, result: dict) -> None:
    summary, gates = result["summary"], result["development_gate_results"]
    lines = [
        "# FMM-aligned MoE localization development result", "",
        "All 16 records were already used for forward development; this is not independent confirmation.", "",
        "| Metric | Waveform only | Waveform + FMM arrival |", "|---|---:|---:|",
        f"| Median location error (m) | {summary['waveform_only']['median_location_error_m']:.1f} | {summary['hybrid']['median_location_error_m']:.1f} |",
        f"| Mean location error (m) | {summary['waveform_only']['mean_location_error_m']:.1f} | {summary['hybrid']['mean_location_error_m']:.1f} |",
        f"| Median origin-time error (s) | {summary['waveform_only']['median_origin_time_error_s']:.4f} | {summary['hybrid']['median_origin_time_error_s']:.4f} |",
        f"| Events below 500 m | {summary['waveform_only']['events_below_500_m']}/16 | {summary['hybrid']['events_below_500_m']}/16 |", "",
        f"Hybrid improved spatial error in {summary['hybrid_improved_location_events']}/16 events.", "",
        "## Predeclared gates", "", "| Gate | Result |", "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |"
                 for name, passed in gates.items())
    lines.extend(["", f"All gates passed: **{result['all_development_gates_passed']}**.",
                  "", result["decision"], "", f"Limitations: {result['limitations']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["output_result"]
    report = ROOT / spec["output_report"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite registered localization output")
    if not spec["protocol_written_before_localization"]:
        raise ValueError("Localization protocol was not registered")
    forward_path = ROOT / spec["forward_development_result"]
    bias_path = ROOT / spec["arrival_constraint"]["development_bias_file"]
    if (sha256(forward_path) != spec["forward_development_result_sha256"]
            or sha256(bias_path)
            != spec["arrival_constraint"]["development_bias_file_sha256"]):
        raise ValueError("Frozen forward decision or picker-bias result changed")
    forward = json.loads(forward_path.read_text(encoding="utf-8"))
    if not forward["all_development_gates_passed"]:
        raise ValueError("Forward model did not pass all registered gates")

    paths = [path for directory in spec["development_sets"]
             for path in sorted((ROOT / directory).glob("*.npz"))]
    excluded = (ROOT / spec["excluded_confirmation_directory"]).resolve()
    if (len(paths) != int(spec["development_event_count"])
            or any(excluded in path.resolve().parents for path in paths)):
        raise ValueError("Development event set changed or confirmation leaked")
    first = load_event(paths[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain = load_model(spec, first["receivers"], device)
    bounds = np.asarray(spec["source_bounds_m_s"], dtype=np.float32)
    search_spec, arrival_spec = spec["search"], spec["arrival_constraint"]
    candidates = coarse_grid(bounds, search_spec["coarse_grid_xyzt"])
    rows = []
    total_seconds = 0.0
    for path in paths:
        event = load_event(path, first["receivers"])
        raw_picks, corrected_picks = pick_arrivals(event, arrival_spec)
        valid_count = int(np.isfinite(corrected_picks).sum())
        if valid_count < int(arrival_spec["minimum_valid_picks"]):
            raise ValueError(f"Only {valid_count} usable picks for {path}")
        coarse_sampled = sampled_waveforms(
            event, domain, int(search_spec["coarse_receiver_stride"]),
            int(search_spec["coarse_time_stride"]), device)
        refined_sampled = sampled_waveforms(
            event, domain, int(search_spec["refine_receiver_stride"]),
            int(search_spec["refine_time_stride"]), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        waveform = waveform_scores(model, candidates, domain, coarse_sampled,
                                   int(search_spec["candidate_batch"]))
        arrival = arrival_scores(
            model, candidates, corrected_picks,
            float(arrival_spec["arrival_scale_s"]),
            int(search_spec["candidate_batch"]))
        branches = {}
        for name, alpha in (("waveform_only", 0.0),
                            ("hybrid", float(arrival_spec["hybrid_weight"]))):
            total = waveform + alpha * arrival
            indices = choose_diverse_starts(
                candidates, total, int(search_spec["restarts"]),
                float(search_spec["minimum_start_separation_m"]))
            starts = [{"source_m_s": candidates[index].tolist(),
                       "coarse_total_loss": float(total[index])}
                      for index in indices]
            winner, refinements = refine(
                model, starts, domain, bounds, refined_sampled, corrected_picks,
                search_spec, arrival_spec, alpha)
            branches[name] = {
                "estimated_source_m_s": winner["estimated_source_m_s"],
                "total_loss": winner["total_loss"],
                "waveform_normalized_mse": winner["waveform_normalized_mse"],
                "arrival_normalized_mse": winner["arrival_normalized_mse"],
                "selected_spatial_move_m": winner["spatial_move_m"],
                "selected_on_boundary": winner["on_boundary"],
                "starts": starts, "refinements": refinements}
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        # Truth is deliberately read only after both searches have finished.
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        for branch in branches.values():
            estimate = np.asarray(branch["estimated_source_m_s"], dtype=np.float64)
            error = estimate - truth
            branch["component_error_m_s"] = error.tolist()
            branch["location_error_m"] = float(np.linalg.norm(error[:3]))
            branch["origin_time_error_s"] = float(abs(error[3]))
        row = {"event": f"{path.parent.name}/{path.name}",
               "observation": str(path.resolve()), "sha256": event["sha256"],
               "source_truth_m_s_post_search": truth.tolist(),
               "valid_raw_picks": valid_count,
               "raw_picks_s": [float(value) if np.isfinite(value) else None
                               for value in raw_picks],
               "corrected_picks_s": [float(value) if np.isfinite(value) else None
                                     for value in corrected_picks],
               "search_seconds": elapsed, **branches}
        rows.append(row)
        print(f"{row['event']}: waveform={branches['waveform_only']['location_error_m']:.1f} m, "
              f"hybrid={branches['hybrid']['location_error_m']:.1f} m, "
              f"time={branches['hybrid']['origin_time_error_s']:.4f} s, "
              f"picks={valid_count}, runtime={elapsed:.1f} s", flush=True)

    summary = {name: summarize(rows, name)
               for name in ("waveform_only", "hybrid")}
    improved = sum(row["hybrid"]["location_error_m"]
                   < row["waveform_only"]["location_error_m"] for row in rows)
    moved = sum(row["hybrid"]["selected_spatial_move_m"] >= 25 for row in rows)
    summary["hybrid_improved_location_events"] = int(improved)
    summary["hybrid_refinement_moved_25m_events"] = int(moved)
    thresholds = spec["development_gates"]
    gate_results = {
        "all_events_have_at_least_six_valid_picks": bool(all(
            row["valid_raw_picks"] >= int(arrival_spec["minimum_valid_picks"])
            for row in rows)),
        "hybrid_median_location_below_750m": bool(
            summary["hybrid"]["median_location_error_m"]
            < thresholds["hybrid_median_location_error_m_below"]),
        "hybrid_median_origin_time_below_0_10s": bool(
            summary["hybrid"]["median_origin_time_error_s"]
            < thresholds["hybrid_median_origin_time_error_s_below"]),
        "hybrid_median_location_not_worse_than_waveform_only": bool(
            summary["hybrid"]["median_location_error_m"]
            <= summary["waveform_only"]["median_location_error_m"]),
        "hybrid_improves_at_least_nine_events": bool(
            improved >= thresholds["hybrid_location_improves_events_at_least"]),
        "hybrid_refinement_moves_25m_in_at_least_twelve_events": bool(
            moved >= thresholds["hybrid_refinement_moves_spatially_by_25m_events_at_least"]),
    }
    passed = bool(all(gate_results.values()))
    result = {
        "purpose": spec["purpose"], "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "checkpoint_sha256": spec["model_checkpoint_sha256"],
        "independent_confirmation_used": False, "event_count": len(rows),
        "device": str(device), "candidate_count_per_event": len(candidates),
        "total_search_seconds": total_seconds, "summary": summary,
        "development_gate_results": gate_results,
        "all_development_gates_passed": passed, "events": rows,
        "decision": ("Freeze the method and register a new prospective confirmation."
                     if passed else
                     "Stop before confirmation and diagnose the failed localization gates using development data only."),
        "limitations": spec["limitations"],
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report, result)
    print(json.dumps({"summary": summary,
                      "development_gate_results": gate_results,
                      "all_development_gates_passed": passed,
                      "total_search_seconds": total_seconds}, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
