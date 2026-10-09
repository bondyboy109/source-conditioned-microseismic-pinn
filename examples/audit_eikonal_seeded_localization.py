"""Development-only FMM-seeded localization audit for the eikonal carrier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from confirm_depth_gap_25 import refine
from run_fmm_seeded_waveform_development import (
    choose_spatial_starts, fine_scores, proposals, hash_file,
)
from source_conditioned_pinn import load_conditioned_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/eikonal_coverage_29_hybrid_1000/conditioned_5Hz.pth")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/eikonal_seeded_localization_development.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Localization output already exists: {args.output}")
    protocol = json.loads((ROOT / "experiments/fmm_seeded_waveform_development_protocol.json").read_text())
    comparison = json.loads((ROOT / "experiments/cross_region_fmm_comparison.json").read_text())
    fmm = json.loads((ROOT / "experiments/cross_region_fmm_baseline.json").read_text())
    checkpoint = args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False)
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    bounds = np.asarray(protocol["refinement"]["bounds_m_s"], dtype=np.float32)
    origins = np.linspace(protocol["origin_time_grid_s"]["minimum"],
                          protocol["origin_time_grid_s"]["maximum"], 46,
                          dtype=np.float32)
    score = protocol["fine_scoring"]
    config = {**json.loads((ROOT / "experiments/cross_region_fmm_protocol.json").read_text())["pinn_inversion"],
              "refine_steps": protocol["refinement"]["steps"],
              "refine_adam_learning_rate_on_box_fractions": protocol["refinement"]["adam_learning_rate_on_box_fractions"]}
    rows = []
    for base, fmm_row in zip(comparison["events"], fmm["events"]):
        path = Path(base["observation"])
        if fmm_row["sha256"] != base["sha256"] or hash_file(path) != base["sha256"]:
            raise ValueError(f"Record mismatch: {path}")
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float32)
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
        candidates = proposals(np.asarray(fmm_row["estimated_source_m_s"][:3]), bounds,
                               protocol["spatial_offsets_m_each_axis"], origins)
        losses = fine_scores(model, receivers, times, data, scale, rms, candidates,
                             score["receiver_stride"], score["time_stride"],
                             score["candidate_batch"])
        indices = choose_spatial_starts(candidates, losses, 12,
                                        protocol["initialization"]["minimum_spatial_distance_m"])
        starts = [{"source_m_s": candidates[index].tolist(),
                   "coarse_normalized_mse": float(losses[index])} for index in indices]
        winner, refinements, _, _, _ = refine(
            model, receivers, times, torch.as_tensor(data, device=device), scale,
            bounds, config, rms, starts)
        estimate = np.asarray(winner["estimated_source_m_s"])
        row = {"event": base["event"], "truth_source_m_s": truth.tolist(),
               "fmm_location_error_m": fmm_row["location_error_m"],
               "eikonal_location_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
               "eikonal_origin_time_error_s": float(abs(estimate[3] - truth[3])),
               "eikonal_normalized_mse": winner["normalized_mse"],
               "eikonal_estimated_source_m_s": estimate.tolist(),
               "candidate_count": len(candidates)}
        rows.append(row)
        print(f"{base['event']}: FMM={row['fmm_location_error_m']:.1f} m, "
              f"eikonal hybrid={row['eikonal_location_error_m']:.1f} m", flush=True)
    result = {"purpose": "post-hoc eikonal FMM-seeded localization audit; not confirmation",
              "checkpoint_sha256": hash_file(checkpoint), "rows": rows,
              "fmm_median_location_m": float(np.median([row["fmm_location_error_m"] for row in rows])),
              "eikonal_median_location_m": float(np.median([row["eikonal_location_error_m"] for row in rows]))}
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
