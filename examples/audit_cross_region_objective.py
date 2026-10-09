"""Development-only audit of the unmodified waveform objective at known sources."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source_conditioned_pinn import conditioned_prediction, load_conditioned_checkpoint

EVENTS = ("region_04.npz", "region_07.npz", "region_08.npz", "central_gap.npz")
WINDOWS = ((0., .5), (.5, 1.), (1., 1.5), (1.5, 2.001))


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loss_parts(observed: np.ndarray, truth_pred: np.ndarray,
               estimate_pred: np.ndarray, times: np.ndarray, coords: np.ndarray,
               full_record_rms: float) -> dict:
    """Sum squared errors / (full-record RMS**2 * number of selected samples)."""
    if observed.shape != truth_pred.shape or observed.shape != estimate_pred.shape:
        raise ValueError("All selected receiver/time arrays must have the same shape")
    normalizer = observed.size * full_record_rms ** 2
    truth_error = (truth_pred - observed) ** 2 / normalizer
    estimate_error = (estimate_pred - observed) ** 2 / normalizer
    observed_energy = float(np.sum(observed ** 2))
    receivers = [{"index": index, "xyz_m": coords[index].tolist(),
                  "observed_energy_fraction": float(np.sum(observed[index] ** 2) / observed_energy),
                  "true_source_mse_contribution": float(truth_error[index].sum()),
                  "returned_source_mse_contribution": float(estimate_error[index].sum()),
                  "returned_minus_truth": float((estimate_error[index] - truth_error[index]).sum())}
                 for index in range(len(coords))]
    windows = []
    for left, right in WINDOWS:
        mask = (times >= left) & (times < right)
        windows.append({"time_s": [left, min(right, 2.)],
                        "observed_energy_fraction": float(np.sum(observed[:, mask] ** 2) / observed_energy),
                        "true_source_mse_contribution": float(truth_error[:, mask].sum()),
                        "returned_source_mse_contribution": float(estimate_error[:, mask].sum()),
                        "returned_minus_truth": float((estimate_error[:, mask] - truth_error[:, mask]).sum())})
    return {"true_source_normalized_mse": float(truth_error.sum()),
            "returned_source_normalized_mse": float(estimate_error.sum()),
            "returned_minus_truth": float((estimate_error - truth_error).sum()),
            "receivers": receivers, "windows": windows}


def main() -> None:
    protocol = json.loads((ROOT / "experiments/cross_region_fmm_protocol.json").read_text())
    baseline = json.loads((ROOT / "experiments/cross_region_fmm_comparison.json").read_text())
    pilot = json.loads((ROOT / "experiments/cross_region_coverage_development_results.json").read_text())
    augmented = json.loads((ROOT / "experiments/cross_region_coverage_localization_development.json").read_text())
    checkpoints = {
        "control25": (ROOT / protocol["fixed_pinn_checkpoint"], protocol["fixed_pinn_sha256"]),
        "augmented29": (ROOT / "outputs/cross_region_coverage_29_hybrid_1000/conditioned_5Hz.pth",
                        pilot["augmented_sha256"]),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name, (checkpoint, digest) in checkpoints.items():
        if hash_file(checkpoint) != digest:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        model, metadata = load_conditioned_checkpoint(checkpoint, device)
        model.eval().requires_grad_(False)
        scale = np.asarray([metadata["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        models[name] = model, scale, metadata
    original = {row["event"]: row for row in baseline["events"]}
    later = {row["event"]: row for row in augmented["rows"]}
    results = []
    for name in EVENTS:
        event = original[name]
        path = Path(event["observation"])
        if hash_file(path) != event["sha256"]:
            raise ValueError(f"FD file hash mismatch: {path}")
        with np.load(path) as archive:
            full_observation = np.asarray(archive["data"], dtype=np.float32)
            observation = full_observation[:, ::4]
            times = np.asarray(archive["times"][::4], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float32)
        if not np.allclose(truth, event["source_truth_m_s"], atol=1e-5, rtol=0):
            raise ValueError("Truth differs from registered FD source")
        row = {"event": name, "sha256": event["sha256"],
               "source_truth_m_s_used_only_for_post_search_audit": truth.tolist(),
               "models": {}}
        for label, (model, scale, metadata) in models.items():
            if any(event["sha256"] == training["sha256"] or
                   np.allclose(truth, training["source_truth"], atol=1e-3, rtol=0)
                   for training in metadata["training_record"]["receiver_observations"]):
                raise ValueError(f"Event overlaps {label} training set")
            estimate = (event["pinn"]["estimated_source_m_s"] if label == "control25"
                        else later[name]["estimate_m_s"])
            xyz_t = torch.as_tensor(receivers / scale[:3], device=device)
            time_t = torch.as_tensor(times / scale[3], device=device)
            with torch.inference_mode():
                predictions = [conditioned_prediction(
                    model, xyz_t, time_t, torch.as_tensor(
                        np.asarray(source, dtype=np.float32) / scale, device=device))
                    .cpu().numpy().astype(np.float64)
                    for source in (truth, estimate)]
            parts = loss_parts(observation.astype(np.float64), *predictions, times,
                               receivers, float(np.sqrt(np.mean(
                                   full_observation.astype(np.float64) ** 2))))
            parts["returned_source_m_s"] = estimate
            parts["largest_receiver_differences"] = sorted(
                parts["receivers"], key=lambda station: abs(station["returned_minus_truth"]),
                reverse=True)[:6]
            if label == "control25" and not np.isclose(
                    parts["true_source_normalized_mse"],
                    event["pinn"]["true_source_normalized_mse"], rtol=5e-4):
                raise ValueError("Objective decomposition differs from registered baseline")
            row["models"][label] = parts
            print(f"{name} {label}: MSE true={parts['true_source_normalized_mse']:.4f}, "
                  f"returned={parts['returned_source_normalized_mse']:.4f}", flush=True)
        results.append(row)
    report = {"purpose": "post-hoc original-objective loss attribution, no tuning or inference",
              "selection_note": "Returned sources were selected on these same records; per-window/station differences are descriptive, not unbiased validation of a new loss.",
              "checkpoint_sha256": {key: value[1] for key, value in checkpoints.items()},
              "receiver_stride": 1, "time_stride": 4,
              "normalization": "MSE divided by full unstrided 2-s observation RMS squared; partitions sum over selected samples",
              "events": results}
    target = ROOT / "experiments/cross_region_objective_audit.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
