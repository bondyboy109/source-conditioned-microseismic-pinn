"""Profile regional waveform misfit over depth and origin on spent data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import fine_scores, hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_depth33_region08_depth_profile_protocol.json"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    result_path = ROOT / protocol["output"]
    if result_path.exists():
        raise FileExistsError(result_path)
    for name, digest in (("observation", "observation_sha256"),
                         ("comparison_result", "comparison_sha256"),
                         ("old_checkpoint", "old_sha256"),
                         ("new_checkpoint", "new_sha256"),
                         ("gated_checkpoint", "gated_sha256")):
        if hash_file(ROOT / protocol[name]) != protocol[digest]:
            raise ValueError(f"Registered input changed: {protocol[name]}")
    with np.load(ROOT / protocol["observation"]) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        truth = np.asarray(archive["source_truth"], dtype=np.float32)
    if data.shape != (25, 1001) or truth.shape != (4,) or not np.isfinite(data).all():
        raise ValueError("Unexpected regional FD data")
    compared = json.loads((ROOT / protocol["comparison_result"]).read_text())
    reference = next(row for row in compared["rows"] if row["event"] == "region_08.npz")
    if not np.isclose(truth[2], 700, atol=1e-5):
        raise ValueError("Regional source depth changed")
    depth_spec = protocol["depth_m"]
    origin_spec = protocol["origin_s"]
    depths = np.arange(depth_spec["minimum"], depth_spec["maximum"] + 1,
                       depth_spec["step"], dtype=np.float32)
    origins = np.linspace(origin_spec["minimum"], origin_spec["maximum"],
                          round((origin_spec["maximum"] - origin_spec["minimum"])
                                / origin_spec["step"]) + 1, dtype=np.float32)
    if not np.any(depths == truth[2]) or len(depths) != 36 or len(origins) != 46:
        raise ValueError("Fixed depth/time profile does not cover source truth")
    candidates = np.asarray([(truth[0], truth[1], z, t)
                             for z in depths for t in origins], dtype=np.float32)
    rms = float(np.sqrt(np.mean(data.astype(np.float64)**2)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {}
    for name in ("old", "new", "gated"):
        checkpoint = ROOT / protocol[f"{name}_checkpoint"]
        model, metadata = load_conditioned_checkpoint(checkpoint, device)
        model.eval().requires_grad_(False)
        scale = np.asarray([metadata["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        values = fine_scores(model, receivers, times, data, scale, rms, candidates,
                             protocol["score_receiver_stride"],
                             protocol["score_time_stride"],
                             protocol["score_candidate_batch"]).reshape(len(depths), len(origins))
        winners = np.argmin(values, axis=1)
        best = values[np.arange(len(depths)), winners]
        search = json.loads((ROOT / protocol[f"{name}_search_result"]).read_text())
        if search["checkpoint_sha256"] != protocol[f"{name}_sha256"]:
            raise ValueError(f"Previous {name} regional search used a different checkpoint")
        winner = next(row for row in search["rows"] if row["event"] == "region_08.npz")
        truth_score, winner_score = fine_scores(
            model, receivers, times, data, scale, rms,
            np.asarray([truth, winner["eikonal_estimated_source_m_s"]], dtype=np.float32),
            protocol["score_receiver_stride"], protocol["score_time_stride"],
            protocol["score_candidate_batch"])
        if not np.isclose(winner_score, winner["eikonal_normalized_mse"], rtol=2e-4, atol=1e-4):
            raise ValueError(f"Previous {name} winner's waveform objective changed")
        best_index = int(np.argmin(best))
        truth_index = int(np.where(depths == truth[2])[0][0])
        rows[name] = {
            "best_depth_m_at_true_xy": float(depths[best_index]),
            "best_time_s_at_true_xy": float(origins[winners[best_index]]),
            "minimum_mse_at_true_xy": float(best[best_index]),
            "mse_at_true_depth_profiled_time": float(best[truth_index]),
            "mse_at_100m_profiled_time": float(best[0]),
            "mse_at_exact_truth": float(truth_score),
            "mse_at_reported_winner": float(winner_score),
            "reported_winner_depth_m": float(winner["eikonal_estimated_source_m_s"][2]),
            "profile": [{"depth_m": float(depth),
                         "best_origin_s": float(origins[time_index]),
                         "best_normalized_mse": float(score)}
                        for depth, time_index, score in zip(depths, winners, best)]}
        print(f"{name}: best depth at true x/y={depths[best_index]:.0f} m, "
              f"truth MSE={truth_score:.4f}, selected MSE={winner_score:.4f}",
              flush=True)
    result = {"role": protocol["role"], "protocol_sha256": hash_file(PROTOCOL),
              "truth_source_m_s": truth.tolist(), "scoring": {
                  "receiver_stride": protocol["score_receiver_stride"],
                  "time_stride": protocol["score_time_stride"],
                  "normalization": "full-record observation RMS fixed for all candidates"},
              "models": rows, "interpretation": protocol["interpretation"]}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
