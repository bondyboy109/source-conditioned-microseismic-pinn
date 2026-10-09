"""Check whether the already-inspected cross-region locations improve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from confirm_depth_gap_25 import inversion_without_truth
from run_cross_region_fmm_comparison import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint


def main() -> None:
    protocol = json.loads((ROOT / "experiments/cross_region_fmm_protocol.json").read_text())
    original = json.loads((ROOT / "experiments/cross_region_fmm_comparison.json").read_text())
    pilot = json.loads((ROOT / "experiments/cross_region_coverage_development_results.json").read_text())
    checkpoint = ROOT / "outputs/cross_region_coverage_29_hybrid_1000/conditioned_5Hz.pth"
    if hash_file(checkpoint) != pilot["augmented_sha256"]:
        raise ValueError("Expanded checkpoint changed after development pilot")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False)
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    training = metadata["training_record"]["receiver_observations"]
    if len(training) != 29:
        raise ValueError("Expected the fixed 29-event checkpoint")
    trained_hashes = {row["sha256"] for row in training}
    rows = []
    for old in original["events"]:
        path = Path(old["observation"])
        if hash_file(path) != old["sha256"] or old["sha256"] in trained_hashes:
            raise ValueError(f"Test observation differs or was trained on: {path}")
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        estimate, _ = inversion_without_truth(
            model, data, times, receivers,
            {"bounds_m_s": protocol["shared_source_bounds_m_s"]},
            protocol["pinn_inversion"], scale)
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float32)
        if not np.allclose(truth, old["source_truth_m_s"], atol=1e-5, rtol=0):
            raise ValueError(f"Source truth changed: {path}")
        recovered = np.asarray(estimate["estimated_source_m_s"])
        row = {"event": old["event"],
               "control_location_m": old["pinn"]["location_error_m"],
               "augmented_location_m": float(np.linalg.norm(recovered[:3] - truth[:3])),
               "control_time_s": old["pinn"]["origin_time_error_s"],
               "augmented_time_s": float(abs(recovered[3] - truth[3])),
               "estimate_m_s": recovered.tolist(),
               "normalized_mse": estimate["final_normalized_mse"],
               "runtime_seconds": estimate["runtime_seconds"]}
        rows.append(row)
        print(f"{old['event']}: {row['control_location_m']:.1f} -> "
              f"{row['augmented_location_m']:.1f} m", flush=True)
    if len(rows) != 8:
        raise ValueError("Expected all eight inspected cross-region events")
    report = {"purpose": "post-hoc localization development diagnostic; not independent confirmation",
              "checkpoint_sha256": hash_file(checkpoint), "baseline_protocol": protocol["purpose"],
              "baseline_gates_are_not_retested_as_independent_validation": True,
              "rows": rows,
              "control_median_m": float(np.median([row["control_location_m"] for row in rows])),
              "augmented_median_m": float(np.median([row["augmented_location_m"] for row in rows]))}
    path = ROOT / "experiments/cross_region_coverage_localization_development.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
