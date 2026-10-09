"""Prospective cross-geology audit for the frozen traveltime-aligned MoE.

The waveform surrogate is trained on Overthrust and is never retrained here.
The independent layered model is used only to generate new FD observations
and a new reciprocal FMM field cache.  Results therefore measure zero-shot
transfer and are not presented as a new training or accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.invert_fmm_receiver_fields import receiver_fields, score_candidates
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores,
    choose_diverse_starts,
    coarse_grid,
    load_event,
    pick_arrivals,
    refine,
    sampled_waveforms,
    waveform_scores,
)
from traveltime_aligned_moe import (
    ReceiverTravelTimeInterpolator,
    TravelTimeAlignedPrototypeMoE,
)


PROTOCOL_PATH = ROOT / "experiments/cross_model_geology_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_layered_model(shape: tuple[int, int, int], spacing: float) -> np.ndarray:
    """Build a deterministic model with undulating, independent interfaces."""
    x = np.arange(shape[0], dtype=np.float32)[:, None, None] * spacing
    y = np.arange(shape[1], dtype=np.float32)[None, :, None] * spacing
    z = np.arange(shape[2], dtype=np.float32)[None, None, :] * spacing
    interface_1 = 620.0 + 110.0 * np.sin(2.0 * np.pi * x / 8000.0) * np.cos(2.0 * np.pi * y / 12000.0)
    interface_2 = 1420.0 + 160.0 * np.cos(2.0 * np.pi * x / 10000.0 + 0.4) * np.sin(2.0 * np.pi * y / 9000.0)
    interface_3 = 2500.0 + 180.0 * np.sin(2.0 * np.pi * (x + y) / 14000.0)
    volume = np.where(z < interface_1, 2350.0,
             np.where(z < interface_2, 3250.0,
             np.where(z < interface_3, 4450.0, 5650.0))).astype(np.float32)
    anomaly = 260.0 * np.exp(-(((x - 9800.0) / 1700.0) ** 2
                               + ((y - 6800.0) / 1900.0) ** 2
                               + ((z - 1850.0) / 550.0) ** 2))
    volume = volume + anomaly.astype(np.float32)
    if volume.shape != shape or not np.isfinite(volume).all():
        raise ValueError("Independent model construction failed")
    return volume


def prepare(spec: dict) -> None:
    model_path = ROOT / spec["velocity_file"]
    config_path = ROOT / spec["fd_config"]
    output_dir = ROOT / spec["observation_directory"]
    if model_path.exists() or config_path.exists() or any(output_dir.glob("*.npz")):
        raise FileExistsError("Cross-model audit inputs already exist; refusing overwrite")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_layered_model(tuple(spec["velocity_shape"]), float(spec["spacing_m"]))
    np.save(model_path, model)
    spec["velocity_sha256"] = sha256(model_path)
    config = {
        "status": "registered_cross_model_fd_configuration",
        "velocity_file": str(model_path),
        "velocity_shape": spec["velocity_shape"],
        "grid_spacing_m": [spec["spacing_m"]] * 3,
        "domain_extent_m": [spec["spacing_m"] * (n - 1) for n in spec["velocity_shape"]],
        "receivers_file": str(ROOT / "data/observations/receiver_coords.npy"),
        "receiver_count": 25,
        "source_location_bounds_m": [[2000.0, 2000.0, 100.0], [14000.0, 14000.0, 1500.0]],
        "origin_time_bounds_s": [0.05, 0.50],
        "time_step_s": 0.002,
        "record_length_s": 2.0,
        "boundary_condition": "zero Dirichlet on all six faces",
        "frequency_validity": "5 Hz; minimum wavelength exceeds ten grid cells",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    for event in spec["events"]:
        path = output_dir / event["file"]
        subprocess.run([
            sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
            "--config", str(config_path), "--output", str(path),
            "--source", *(str(value) for value in event["source_m_s"]),
        ], cwd=ROOT, check=True)
        with np.load(path) as archive:
            if not np.allclose(archive["source_truth"], event["source_m_s"], atol=1e-5):
                raise ValueError(f"Generated source mismatch: {path}")
        event["sha256"] = sha256(path)
    PROTOCOL_PATH.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Prepared independent model {model_path} and {len(spec['events'])} FD events")


def load_zero_shot_model(checkpoint_path: Path, fields: np.ndarray,
                         receivers: np.ndarray, spacing_m: float,
                         device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_spec = payload["protocol"]["model"]
    fmm_spec = payload["protocol"]["fmm"]
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)), spacing_m).to(device)
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
    return model.eval().requires_grad_(False), model_spec


def fmm_locate(fields: np.ndarray, event: dict, corrected: np.ndarray,
               bounds: np.ndarray, spacing_m: float, search: dict,
               arrival_spec: dict) -> dict:
    valid = np.isfinite(corrected)
    candidates = np.asarray(np.meshgrid(
        *[np.linspace(low, high, n) for (low, high), n in zip(bounds, search["coarse_grid_xyz"])],
        indexing="ij")).reshape(3, -1).T
    losses, origins = score_candidates(fields, candidates, corrected,
                                        spacing_m, tuple(arrival_spec["origin_time_bounds_s"]))
    index = int(np.argmin(losses))
    best = (float(losses[index]), candidates[index].copy(), float(origins[index]))
    steps = (bounds[:, 1] - bounds[:, 0]) / (np.asarray(search["coarse_grid_xyz"]) - 1)
    for _ in range(int(search["refinement_levels"])):
        low = np.maximum(best[1] - steps, bounds[:, 0])
        high = np.minimum(best[1] + steps, bounds[:, 1])
        local = np.asarray(np.meshgrid(
            *[np.linspace(lo, hi, search["refinement_grid_xyz"]) for lo, hi in zip(low, high)],
            indexing="ij")).reshape(3, -1).T
        local_losses, local_origins = score_candidates(
            fields, local, corrected, spacing_m,
            tuple(arrival_spec["origin_time_bounds_s"]))
        local_index = int(np.argmin(local_losses))
        if local_losses[local_index] < best[0]:
            best = (float(local_losses[local_index]), local[local_index].copy(),
                    float(local_origins[local_index]))
        steps *= 2.0 / (search["refinement_grid_xyz"] - 1)
    estimate = np.r_[best[1], best[2]]
    truth = np.asarray(event["truth"], dtype=np.float64)
    error = estimate - truth
    return {"estimated_source_m_s": estimate.tolist(),
            "location_error_m": float(np.linalg.norm(error[:3])),
            "origin_time_error_s": float(abs(error[3])),
            "valid_picks": int(valid.sum()), "arrival_rmse_s": float(np.sqrt(best[0]))}


def evaluate(spec: dict) -> None:
    checkpoint = ROOT / spec["checkpoint"]
    model_path = ROOT / spec["velocity_file"]
    if sha256(model_path) != spec["velocity_sha256"]:
        raise ValueError("Independent velocity model hash changed")
    receivers = np.load(ROOT / "data/observations/receiver_coords.npy").astype(np.float32)
    model = np.load(model_path, mmap_mode="r")
    stride = int(spec["fmm_stride"])
    downsampled = np.asarray(model[::stride, ::stride, ::stride], dtype=np.float32)
    spacing = float(spec["spacing_m"] * stride)
    started = time.perf_counter()
    fields = receiver_fields(downsampled, receivers, spacing)
    fmm_seconds = time.perf_counter() - started
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zero_shot, model_spec = load_zero_shot_model(checkpoint, fields, receivers, spacing, device)
    bounds = np.asarray(spec["source_bounds_m_s"][:3], dtype=np.float32)
    search = spec["search"]
    arrival_spec = spec["arrival_constraint"]
    candidates = coarse_grid(np.asarray(spec["source_bounds_m_s"], dtype=np.float32),
                             search["coarse_grid_xyzt"])
    rows = []
    for event_spec in spec["events"]:
        path = ROOT / spec["observation_directory"] / event_spec["file"]
        if sha256(path) != event_spec["sha256"]:
            raise ValueError(f"Observation hash changed: {path}")
        event = load_event(path, receivers)
        event["truth"] = np.asarray(event_spec["source_m_s"], dtype=np.float64)
        _, corrected = pick_arrivals(event, arrival_spec)
        if int(np.isfinite(corrected).sum()) < int(arrival_spec["minimum_valid_picks"]):
            raise ValueError(f"Insufficient picks in {path}")
        coarse_sampled = sampled_waveforms(event, np.asarray(model_spec["spatial_lengths_m"] + [model_spec["time_length_s"]], dtype=np.float32),
                                           int(search["coarse_receiver_stride"]), int(search["coarse_time_stride"]), device)
        refined_sampled = sampled_waveforms(event, np.asarray(model_spec["spatial_lengths_m"] + [model_spec["time_length_s"]], dtype=np.float32),
                                            int(search["refine_receiver_stride"]), int(search["refine_time_stride"]), device)
        waveform = waveform_scores(zero_shot, candidates,
                                   np.asarray(model_spec["spatial_lengths_m"] + [model_spec["time_length_s"]], dtype=np.float32),
                                   coarse_sampled, int(search["candidate_batch"]))
        arrival = arrival_scores(zero_shot, candidates, corrected,
                                 float(arrival_spec["arrival_scale_s"]), int(search["candidate_batch"]))
        branches = {}
        for name, alpha in (("waveform_only", 0.0), ("hybrid", float(arrival_spec["hybrid_weight"]))):
            total = waveform + alpha * arrival
            indices = choose_diverse_starts(candidates, total, int(search["restarts"]),
                                             float(search["minimum_start_separation_m"]))
            starts = [{"source_m_s": candidates[i].tolist(), "coarse_total_loss": float(total[i])}
                      for i in indices]
            best, _ = refine(zero_shot, starts,
                             np.asarray(model_spec["spatial_lengths_m"] + [model_spec["time_length_s"]], dtype=np.float32),
                             np.asarray(spec["source_bounds_m_s"], dtype=np.float32), refined_sampled,
                             corrected, search, arrival_spec, alpha)
            estimate = np.asarray(best["estimated_source_m_s"])
            error = estimate - event["truth"]
            branches[name] = {"estimated_source_m_s": estimate.tolist(),
                              "location_error_m": float(np.linalg.norm(error[:3])),
                              "origin_time_error_s": float(abs(error[3])),
                              "forward_waveform_mse": float(best["waveform_normalized_mse"]),
                              "arrival_mse": float(best["arrival_normalized_mse"])}
        branches["fmm"] = fmm_locate(fields, event, corrected, bounds, spacing, search, arrival_spec)
        rows.append({"event": event_spec["name"], "truth_m_s": event_spec["source_m_s"],
                     "valid_picks": int(np.isfinite(corrected).sum()), "branches": branches})
        print(event_spec["name"], {key: round(value["location_error_m"], 1)
                                    for key, value in branches.items()}, flush=True)
    summary = {}
    for branch in ("waveform_only", "hybrid", "fmm"):
        loc = np.asarray([row["branches"][branch]["location_error_m"] for row in rows])
        origin = np.asarray([row["branches"][branch]["origin_time_error_s"] for row in rows])
        summary[branch] = {"median_location_error_m": float(np.median(loc)),
                           "mean_location_error_m": float(np.mean(loc)),
                           "median_origin_time_error_s": float(np.median(origin)),
                           "events_below_500_m": int(np.sum(loc < 500.0))}
    result = {"method": "zero_shot_cross_geology_audit", "model": "independent_layered_undulating",
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "velocity_sha256": spec["velocity_sha256"], "fmm_spacing_m": spacing,
              "fmm_setup_seconds": fmm_seconds, "events": rows, "summary": summary,
              "interpretation": "Zero-shot transfer audit only; no retraining and no claim of cross-geology generalization.",
              "limitations": "The Overthrust-trained waveform surrogate is evaluated on a distinct synthetic geology; the FMM fields use the independent model, while picker bias and search settings remain frozen from Overthrust development."}
    out = ROOT / spec["output_result"]
    report = ROOT / spec["output_report"]
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Cross-geology zero-shot audit", "",
             "The frozen Overthrust-trained MoE was evaluated without retraining on an independent layered model.", "",
             "| Metric | Waveform only | Hybrid | FMM baseline |", "|---|---:|---:|---:|"]
    for metric, key, fmt in (("Median location error (m)", "median_location_error_m", ".1f"),
                             ("Mean location error (m)", "mean_location_error_m", ".1f"),
                             ("Median origin-time error (s)", "median_origin_time_error_s", ".4f"),
                             ("Events below 500 m", "events_below_500_m", "d")):
        values = [summary[name][key] for name in ("waveform_only", "hybrid", "fmm")]
        lines.append(f"| {metric} | " + " | ".join(format(value, fmt) for value in values) + " |")
    lines.extend(["", f"FMM field setup: {fmm_seconds:.2f} s.", "", result["interpretation"], "", result["limitations"]])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out} and {report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.evaluate:
        parser.error("Choose exactly one of --prepare or --evaluate")
    if args.prepare:
        spec = {
            "purpose": "registered cross-geology zero-shot audit",
            "protocol_written_before_fd_generation": False,
            "velocity_file": "data/velocity/independent_layered_undulating.npy",
            "fd_config": "data/observations/independent_layered_undulating/fd_config.json",
            "observation_directory": "data/observations/independent_layered_undulating",
            "velocity_shape": [401, 401, 94], "spacing_m": 40.0, "fmm_stride": 2,
            "checkpoint": "outputs/traveltime_aligned_moe_development/traveltime_aligned_moe.pth",
            "source_bounds_m_s": [[2000.0, 14000.0], [2000.0, 14000.0], [100.0, 1500.0], [0.05, 0.50]],
            "search": {"coarse_grid_xyzt": [11, 11, 8, 6], "coarse_grid_xyz": [11, 11, 8],
                       "coarse_receiver_stride": 2, "coarse_time_stride": 8,
                       "candidate_batch": 64, "restarts": 5, "minimum_start_separation_m": 200.0,
                       "refine_receiver_stride": 1, "refine_time_stride": 4, "refine_steps": 250,
                       "refine_adam_learning_rate_on_box_fractions": 0.0025,
                       "refinement_grid_xyz": 7, "refinement_levels": 4},
            "arrival_constraint": {"picker": "robust_mad_6sigma", "minimum_valid_picks": 6,
                                   "minimum_receiver_peak_fraction_of_record": 0.01,
                                   "last_record_s_excluded_for_pick": 0.15,
                                   "pick_bias_s": -0.26279182618600316, "arrival_scale_s": 0.10,
                                   "hybrid_weight": 0.25,
                                   "origin_time_bounds_s": [0.05, 0.50]},
            "events": [
                {"name": "layered_west_shallow", "file": "cross_01_west_shallow.npz", "source_m_s": [3000.0, 12000.0, 300.0, 0.16]},
                {"name": "layered_center_deep", "file": "cross_02_center_deep.npz", "source_m_s": [8000.0, 8000.0, 1300.0, 0.28]},
                {"name": "layered_east_mid", "file": "cross_03_east_mid.npz", "source_m_s": [13000.0, 4500.0, 800.0, 0.37]},
                {"name": "layered_north_deep", "file": "cross_04_north_deep.npz", "source_m_s": [6000.0, 13500.0, 1450.0, 0.43]},
            ],
            "output_result": "experiments/cross_model_geology_results.json",
            "output_report": "experiments/cross_model_geology_results.md",
        }
        prepare(spec)
    else:
        spec = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        evaluate(spec)


if __name__ == "__main__":
    main()
