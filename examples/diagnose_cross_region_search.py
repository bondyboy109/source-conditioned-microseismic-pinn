"""Development-only search audit on two already-inspected cross-region events."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from confirm_depth_gap_25 import inversion_without_truth, refine
from invert_source_conditioned import waveform_mse
from run_cross_region_fmm_comparison import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint


def main() -> None:
    protocol = json.loads((ROOT / "experiments/cross_region_fmm_protocol.json").read_text())
    baseline = json.loads((ROOT / "experiments/cross_region_fmm_comparison.json").read_text())
    checkpoint = ROOT / protocol["fixed_pinn_checkpoint"]
    if (hash_file(checkpoint) != protocol["fixed_pinn_sha256"]
            or baseline["pinn_checkpoint_sha256"] != hash_file(checkpoint)):
        raise ValueError("Frozen model differs from registered comparison")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False)
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    bounds = np.asarray(protocol["shared_source_bounds_m_s"], dtype=np.float32)
    original = protocol["pinn_inversion"]
    variants = {
        "more_starts": {**original, "restarts": 20},
        "denser_grid": {**original, "coarse_grid_xyzt": [17, 17, 11, 9],
                        "restarts": 20},
    }
    records = []
    for entry in baseline["events"]:
        if entry["event"] not in {"region_04.npz", "region_07.npz"}:
            continue
        path = Path(entry["observation"])
        if hash_file(path) != entry["sha256"]:
            raise ValueError(f"Observation changed: {path}")
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
        row = {"event": entry["event"], "observation_sha256": entry["sha256"],
               "original_search": entry["pinn"], "variants": {}}
        for name, config in variants.items():
            result, (rms, xyz, time_t, selected) = inversion_without_truth(
                model, data, times, receivers, {"bounds_m_s": bounds.tolist()},
                config, scale)
            row["variants"][name] = {
                "estimated_source_m_s": result["estimated_source_m_s"],
                "normalized_mse": result["final_normalized_mse"],
                "grid_candidates": result["grid_candidates"],
                "coarse_starts": result["coarse_starts"],
                "refinements": result["refinements"],
                "runtime_seconds": result["runtime_seconds"],
            }
            print(f"{entry['event']} {name}: MSE="
                  f"{result['final_normalized_mse']:.4f}", flush=True)

        # Truth is opened only after both truth-independent searches finish.
        # This oracle start measures a local basin; it is never an inversion.
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float32)
        if not np.allclose(truth, entry["source_truth_m_s"], rtol=0, atol=1e-5):
            raise ValueError("Source truth differs from original comparison")
        with torch.inference_mode():
            true_mse = float(waveform_mse(
                model, xyz, time_t, selected,
                torch.as_tensor(truth / scale, device=device), rms))
        oracle, _, _, _, _ = refine(
            model, receivers, times, torch.as_tensor(data, device=device),
            scale, bounds, original, rms,
            [{"source_m_s": truth.tolist(), "coarse_normalized_mse": None}])
        row["truth_source_m_s_post_search"] = truth.tolist()
        row["true_source_normalized_mse"] = true_mse
        row["oracle_truth_initialized_refinement_not_an_inversion"] = oracle
        for item in row["variants"].values():
            estimate = np.asarray(item["estimated_source_m_s"])
            item["location_error_m_post_search"] = float(np.linalg.norm(estimate[:3] - truth[:3]))
            item["origin_time_error_s_post_search"] = float(abs(estimate[3] - truth[3]))
        print(f"{entry['event']} oracle MSE={oracle['normalized_mse']:.4f}", flush=True)
        records.append(row)
    if len(records) != 2:
        raise ValueError("Missing one of the two inspected diagnostic events")
    report = {"purpose": "development-only frozen-model search audit; not confirmation",
              "baseline_protocol": "experiments/cross_region_fmm_protocol.json",
              "baseline_result": "experiments/cross_region_fmm_comparison.json",
              "checkpoint_sha256": hash_file(checkpoint),
              "unchanged_original_inversion": original,
              "diagnostic_variants": variants,
              "events": records}
    target = ROOT / "experiments/cross_region_search_diagnostic.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
