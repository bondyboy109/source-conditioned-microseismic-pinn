"""Diagnose whether the original coarse screen discards near-truth starts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from confirm_depth_gap_25 import coarse_search
from invert_source_conditioned import waveform_mse
from run_cross_region_fmm_comparison import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint


def main() -> None:
    original = json.loads((ROOT / "experiments/cross_region_fmm_comparison.json").read_text())
    protocol = json.loads((ROOT / "experiments/cross_region_fmm_protocol.json").read_text())
    pilot = json.loads((ROOT / "experiments/cross_region_coverage_development_results.json").read_text())
    checkpoint = ROOT / "outputs/cross_region_coverage_29_hybrid_1000/conditioned_5Hz.pth"
    if hash_file(checkpoint) != pilot["augmented_sha256"]:
        raise ValueError("Checkpoint changed after pilot")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_conditioned_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False)
    scale = np.asarray([meta["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    bounds = np.asarray(protocol["shared_source_bounds_m_s"], dtype=np.float32)
    config = protocol["pinn_inversion"]
    count = int(np.prod(config["coarse_grid_xyzt"]))
    all_starts_config = {**config, "restarts": count}
    results = []
    selected = {"region_04.npz", "region_07.npz", "region_08.npz", "central_gap.npz"}
    for row in original["events"]:
        if row["event"] not in selected:
            continue
        path = Path(row["observation"])
        if hash_file(path) != row["sha256"]:
            raise ValueError(f"FD file changed: {path}")
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float32)
        if not np.allclose(truth, row["source_truth_m_s"], rtol=0, atol=1e-5):
            raise ValueError("Truth changed")
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
        ranked = coarse_search(model, receivers, times, torch.as_tensor(data, device=device),
                               scale, bounds, all_starts_config, rms)
        # Truth is used only here, after the complete coarse grid has been ranked.
        distances = [float(np.linalg.norm((np.asarray(item["source_m_s"]) - truth)
                                          / (bounds[:, 1] - bounds[:, 0])))
                     for item in ranked]
        nearest_rank = int(np.argmin(distances))
        nearest = ranked[nearest_rank]
        fine_xyz = torch.as_tensor(receivers / scale[:3], device=device)
        fine_times = torch.as_tensor(times[::config["refine_time_stride"]] / scale[3],
                                     device=device)
        fine_observed = torch.as_tensor(data[:, ::config["refine_time_stride"]],
                                        device=device)
        with torch.inference_mode():
            fine_at_nearest = float(waveform_mse(
                model, fine_xyz, fine_times, fine_observed,
                torch.as_tensor(np.asarray(nearest["source_m_s"], dtype=np.float32)
                                / scale, device=device), rms))
            fine_at_truth = float(waveform_mse(
                model, fine_xyz, fine_times, fine_observed,
                torch.as_tensor(truth / scale, device=device), rms))
        result = {"event": row["event"], "candidate_count": count,
                  "nearest_grid_to_truth_rank_one_based": nearest_rank + 1,
                  "nearest_grid_to_truth_source_m_s": nearest["source_m_s"],
                  "nearest_grid_to_truth_coarse_mse": nearest["coarse_normalized_mse"],
                  "nearest_grid_to_truth_fine_mse": fine_at_nearest,
                  "continuous_truth_fine_mse": fine_at_truth,
                  "top_five_coarse": ranked[:5]}
        results.append(result)
        print(f"{row['event']}: nearest coarse rank {nearest_rank+1}/{count}, "
              f"MSE grid coarse/fine={nearest['coarse_normalized_mse']:.3f}/"
              f"{fine_at_nearest:.3f}, true fine={fine_at_truth:.3f}", flush=True)
    if len(results) != 4:
        raise ValueError("Missing development event")
    target = ROOT / "experiments/cross_region_coarse_ranking.json"
    target.write_text(json.dumps({"purpose": "post-hoc coarse-rank audit, not localization",
                                  "checkpoint_sha256": hash_file(checkpoint),
                                  "truth_use": "nearest-grid rank and fine loss only, after ranking all candidates",
                                  "results": results}, indent=2), encoding="utf-8")
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
