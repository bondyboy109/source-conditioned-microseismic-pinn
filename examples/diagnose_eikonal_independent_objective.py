"""Post-hoc loss-landscape audit of the eight spent independent events."""

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

PROTOCOL = ROOT / "experiments/eikonal_independent_protocol.json"
RESULT = ROOT / "experiments/eikonal_independent_results.json"
OUTPUT = ROOT / "experiments/eikonal_independent_objective_diagnostic.json"


def main() -> None:
    registered = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    search = json.loads((ROOT / registered["prior_fmm_seeded_settings"]).read_text())
    checkpoint = ROOT / registered["eikonal_checkpoint"]
    if (hash_file(checkpoint) != registered["eikonal_sha256"]
            or result["eikonal_sha256"] != registered["eikonal_sha256"]
            or len(result["events"]) != len(registered["events"]) == 8):
        raise ValueError("Frozen operator or independent result changed")
    model, metadata = load_conditioned_checkpoint(
        checkpoint, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().requires_grad_(False)
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    time_spec = search["origin_time_grid_s"]
    origins = np.linspace(time_spec["minimum"], time_spec["maximum"],
                          round((time_spec["maximum"] - time_spec["minimum"])
                                / time_spec["step"]) + 1, dtype=np.float32)
    scoring = search["fine_scoring"]
    fractions = np.linspace(0, 1, 11, dtype=np.float32)
    rows = []
    for item, entry in zip(registered["events"], result["events"]):
        path = ROOT / "data/observations/eikonal_independent" / item["file"]
        if (item["file"] != entry["event"] or hash_file(path) != entry["sha256"]
                or entry["eikonal_hybrid"] is None or entry["fmm"]["status"] != "ok"):
            raise ValueError(f"Incomplete or changed independent event: {path}")
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float32)
        if (not np.allclose(truth, item["source_m_s"], atol=1e-5, rtol=0)
                or not np.allclose(truth, entry["truth_post_search_m_s"], atol=1e-5, rtol=0)):
            raise ValueError(f"Truth differs from the completed independent run: {path}")
        fmm = np.asarray(entry["fmm"]["estimated_source_m_s"], dtype=np.float32)
        winner = np.asarray(entry["eikonal_hybrid"]["estimated_source_m_s"], dtype=np.float32)
        positions = fmm[None, :3] + fractions[:, None] * (truth[:3] - fmm[:3])[None, :]
        candidates = np.asarray([(*xyz, float(origin)) for xyz in positions
                                 for origin in origins], dtype=np.float32)
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))

        def evaluate(points: np.ndarray) -> np.ndarray:
            return fine_scores(model, receivers, times, data, scale, rms, points,
                               scoring["receiver_stride"], scoring["time_stride"],
                               scoring["candidate_batch"])

        profile = evaluate(candidates).reshape(len(fractions), len(origins))
        best_times = np.argmin(profile, axis=1)
        other = evaluate(np.stack((truth, fmm, winner)))
        best_line = int(np.argmin(profile[np.arange(len(fractions)), best_times]))
        if not np.isclose(other[2], entry["eikonal_hybrid"]["normalized_mse"], rtol=2e-4, atol=1e-4):
            raise ValueError(f"Diagnostic and recorded waveform objectives differ: {path}")
        row = {
            "event": item["file"], "sha256": entry["sha256"],
            "truth_used_only_in_post_hoc_diagnostic": True,
            "location_error_fmm_m": entry["fmm"]["location_error_m"],
            "location_error_hybrid_m": entry["eikonal_hybrid"]["location_error_m"],
            "source_m_s": {"truth": truth.tolist(), "fmm": fmm.tolist(),
                           "hybrid": winner.tolist()},
            "loss_at_true_space_and_true_time": float(other[0]),
            "loss_at_fmm_space_and_fmm_time": float(other[1]),
            "loss_at_hybrid_output": float(other[2]),
            "time_profiled_fmm_space": {"time_s": float(origins[best_times[0]]),
                                        "normalized_mse": float(profile[0, best_times[0]])},
            "time_profiled_true_space": {"time_s": float(origins[best_times[-1]]),
                                         "normalized_mse": float(profile[-1, best_times[-1]])},
            "fmm_to_truth_line": [{"fraction_toward_truth": float(fraction),
                                   "best_time_s": float(origins[index]),
                                   "best_normalized_mse": float(profile[k, index])}
                                  for k, (fraction, index) in enumerate(zip(fractions, best_times))],
            "line_minimum_fraction_toward_truth": float(fractions[best_line]),
        }
        rows.append(row)
        print(f"{item['file']}: true={other[0]:.4f}, FMM profiled="
              f"{row['time_profiled_fmm_space']['normalized_mse']:.4f}, "
              f"true profiled={row['time_profiled_true_space']['normalized_mse']:.4f}, "
              f"hybrid={other[2]:.4f}, line-min={fractions[best_line]:.1f}", flush=True)
    output = {"purpose": "post-hoc diagnostic on eight already-spent independent events; not a new test",
              "original_protocol": str(PROTOCOL.resolve()),
              "original_results": str(RESULT.resolve()),
              "checkpoint_sha256": registered["eikonal_sha256"],
              "receiver_stride": scoring["receiver_stride"],
              "time_stride": scoring["time_stride"],
              "normalization": "full unstrided 2-s FD observation RMS, identical to local waveform refinement",
              "origin_time_grid_s": {"minimum": float(origins[0]),
                                     "maximum": float(origins[-1]),
                                     "count": len(origins)},
              "line_samples": len(fractions), "events": rows}
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
