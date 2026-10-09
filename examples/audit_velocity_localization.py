"""Evaluate learned velocity fields in the established FMM localization baseline.

This is a velocity-sensitivity diagnostic, not a waveform-PINN inversion.
Coarse traveltime fields are reused across events for each velocity model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from audit_velocity_transfer import VelocityNet
from audit_fmm_traveltime import first_arrival
from invert_fmm_traveltime import fmm_receiver_times


def reconstruct_volume(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["target_shape"]) != expected_shape:
        raise ValueError(f"Checkpoint grid mismatch: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VelocityNet(int(checkpoint["width"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output = np.empty(expected_shape, dtype=np.float32)
    plane = np.stack(np.meshgrid(np.arange(expected_shape[1]),
                                 np.arange(expected_shape[2]), indexing="ij"), axis=-1)
    with torch.inference_mode():
        for ix in range(expected_shape[0]):
            coords = np.empty((*plane.shape[:2], 3), dtype=np.float32)
            coords[..., 0] = ix
            coords[..., 1:] = plane
            coords = torch.from_numpy(coords.reshape(-1, 3)).to(device)
            coords = 2.0 * coords / coords.new_tensor(np.array(expected_shape) - 1) - 1.0
            output[ix] = (model(coords).reshape(expected_shape[1:]).cpu().numpy()
                          * checkpoint["velocity_scale_m_s"]
                          + checkpoint["velocity_offset_m_s"])
    if not np.all(np.isfinite(output)) or np.min(output) <= 0:
        raise ValueError(f"Nonpositive/nonfinite reconstructed velocity: {path}")
    return output


def locate(velocity: np.ndarray, observations: list[dict], stride: int,
           axes: list[np.ndarray], refine_grid: int) -> tuple[list[dict], float]:
    decimated = np.asarray(velocity[::stride, ::stride, ::stride], dtype=np.float64)
    spacing = 40.0 * stride
    candidates = np.asarray(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    coarse_step = np.array([axis[1] - axis[0] for axis in axes])
    started = time.perf_counter()
    # Receiver geometry is identical across these FD observations.
    receiver_times = [fmm_receiver_times(decimated, xyz, observations[0]["receivers"],
                                         spacing) for xyz in candidates]
    rows = []
    for observation in observations:
        picks, valid = observation["picks"], observation["valid"]

        def evaluate(xyz: np.ndarray, travel: np.ndarray, current):
            origin = float(np.clip(np.median(picks[valid] - travel[valid]), 0.05, 0.50))
            residual = origin + travel[valid] - picks[valid]
            misfit = float(np.mean(residual ** 2))
            if current is None or misfit < current[0]:
                return (misfit, xyz.copy(), origin)
            return current

        best = None
        for xyz, travel in zip(candidates, receiver_times):
            best = evaluate(xyz, travel, best)
        if refine_grid >= 3:
            local_axes = [np.linspace(best[1][i] - coarse_step[i],
                                      best[1][i] + coarse_step[i], refine_grid)
                          for i in range(3)]
            local = np.asarray(np.meshgrid(*local_axes, indexing="ij")).reshape(3, -1).T
            local = np.clip(local, [2000.0, 2000.0, 100.0],
                            [14000.0, 14000.0, 1500.0])
            for xyz in local:
                best = evaluate(xyz, fmm_receiver_times(decimated, xyz,
                                observation["receivers"], spacing), best)
        estimate = np.r_[best[1], best[2]]
        difference = estimate - observation["truth"]
        rows.append({"observation": observation["path"],
                     "truth": observation["truth"].tolist(),
                     "estimate": estimate.tolist(),
                     "location_error_m": float(np.linalg.norm(difference[:3])),
                     "origin_time_error_s": float(abs(difference[3])),
                     "arrival_rmse_s": float(np.sqrt(best[0])),
                     "valid_picks": int(valid.sum())})
    return rows, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--truth-velocity", type=Path, required=True)
    parser.add_argument("--observation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--grid", type=int, nargs=3, default=[8, 8, 5])
    parser.add_argument("--refine-grid", type=int, default=3)
    args = parser.parse_args()
    if args.stride < 1 or any(n < 2 for n in args.grid) or args.refine_grid < 0:
        parser.error("Invalid stride or grid size")
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    with args.truth_velocity.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != report["target_sha256"]:
        parser.error("Reference velocity file does not match the training audit")
    selected = next((row for row in report["results"] if row["seed"] == args.seed), None)
    if selected is None:
        parser.error(f"Seed {args.seed} is absent from audit")
    observations = []
    for path in args.observation:
        with np.load(path) as archive:
            data = np.asarray(archive["data"])
            times = np.asarray(archive["times"])
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        picks = np.asarray([first_arrival(trace, times) for trace in data])
        valid = np.isfinite(picks)
        if not np.any(valid):
            raise ValueError(f"No valid arrivals in {path}")
        if observations and not np.array_equal(receivers, observations[0]["receivers"]):
            raise ValueError("Receiver geometry differs across events")
        observations.append({"path": str(path), "receivers": receivers,
                             "truth": truth, "picks": picks, "valid": valid})
    reference = np.load(args.truth_velocity, mmap_mode="r")
    axes = [np.linspace(2000.0, 14000.0, args.grid[0]),
            np.linspace(2000.0, 14000.0, args.grid[1]),
            np.linspace(100.0, 1500.0, args.grid[2])]
    output = {"method": "velocity_reconstruction_sensitivity_via_FMM",
              "scope": "single-seed diagnostic; no waveform PINN retraining",
              "seed": args.seed, "stride": args.stride, "grid": args.grid,
              "refine_grid": args.refine_grid,
              "observations": [str(path) for path in args.observation], "models": {}}
    models = {name: info.get("checkpoint") for name, info in selected["arms"].items()}
    models["true_velocity"] = None
    for name, checkpoint in models.items():
        if name != "true_velocity" and not checkpoint:
            parser.error(f"Missing checkpoint for {name}; rerun audit with --checkpoint-dir")
        volume = (reference if checkpoint is None else
                  reconstruct_volume(Path(checkpoint), reference.shape))
        rows, elapsed = locate(volume, observations, args.stride, axes, args.refine_grid)
        output["models"][name] = {
            "checkpoint": checkpoint,
            "velocity_min_max_m_s": [float(np.min(volume)), float(np.max(volume))],
            "mean_location_error_m": float(np.mean([r["location_error_m"] for r in rows])),
            "mean_origin_time_error_s": float(np.mean([r["origin_time_error_s"] for r in rows])),
            "runtime_s": elapsed, "events": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"{name}: mean eloc={output['models'][name]['mean_location_error_m']:.1f} m, "
              f"mean et={output['models'][name]['mean_origin_time_error_s']:.3f} s, "
              f"FMM runtime={elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
