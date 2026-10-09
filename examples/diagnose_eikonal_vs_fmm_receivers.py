"""Compare frozen eikonal and 80 m FMM receiver times on spent FD events."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import skfmm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_fmm_traveltime import trilinear
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_independent_protocol.json"
INDEPENDENT = ROOT / "experiments/eikonal_independent_results.json"
OUTPUT = ROOT / "experiments/eikonal_independent_carrier_vs_fmm.json"


def main() -> None:
    registered = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    original = json.loads(INDEPENDENT.read_text(encoding="utf-8"))
    if (hash_file(ROOT / registered["eikonal_checkpoint"]) != registered["eikonal_sha256"]
            or hash_file(ROOT / registered["velocity_file"]) != registered["velocity_sha256"]
            or len(original["events"]) != len(registered["events"]) == 8):
        raise ValueError("Frozen model, velocity or independent events changed")
    model, payload = load_conditioned_checkpoint(
        ROOT / registered["eikonal_checkpoint"],
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().requires_grad_(False)
    if not hasattr(model, "eikonal_model"):
        raise ValueError("Frozen PINN has no eikonal travel-time carrier")
    scale = np.asarray([payload["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m")], dtype=np.float32)
    original_fmm = json.loads((ROOT / registered["prior_search_settings"]).read_text())["fmm_baseline"]
    stride = original_fmm["velocity_grid_stride"]
    original_grid = np.load(ROOT / registered["velocity_file"], mmap_mode="r")
    velocity = np.ascontiguousarray(original_grid[::stride, ::stride, ::stride],
                                    dtype=np.float32)
    spacing_m = 40.0 * stride
    rows = []
    for entry, result in zip(registered["events"], original["events"]):
        path = ROOT / "data/observations/eikonal_independent" / entry["file"]
        if result["event"] != entry["file"] or hash_file(path) != result["sha256"]:
            raise ValueError(f"Event differs from independent run: {path}")
        with np.load(path) as record:
            receivers = np.asarray(record["receiver_coords"], dtype=np.float64)
            source = np.asarray(record["source_truth"], dtype=np.float64)
        if not np.allclose(source, entry["source_m_s"], atol=1e-5, rtol=0):
            raise ValueError(f"Source differs from preregistration: {path}")
        grid_cell = np.rint(source[:3] / spacing_m).astype(int)
        level_set = np.ones(velocity.shape, dtype=np.float64)
        level_set[tuple(grid_cell)] = -1.0
        travel = np.asarray(skfmm.travel_time(level_set, speed=velocity, dx=spacing_m),
                            dtype=np.float32)
        fmm_times = trilinear(travel, receivers, spacing_m)
        device = next(model.parameters()).device
        with torch.inference_mode():
            receiver_t = torch.as_tensor(receivers.astype(np.float32) / scale,
                                         device=device)
            source_t = torch.as_tensor(np.broadcast_to(
                source[:3].astype(np.float32) / scale, (len(receivers), 3)).copy(),
                device=device)
            neural_times = model.eikonal_model(receiver_t, source_t).squeeze(-1).cpu().numpy()
        difference = neural_times.astype(np.float64) - fmm_times
        row = {"event": entry["file"], "sha256": result["sha256"],
               "source_used_only_for_post_hoc_diagnostic_m_s": source.tolist(),
               "fmm_source_grid_center_m": (grid_cell * spacing_m).tolist(),
               "fmm_source_grid_snap_distance_m": float(np.linalg.norm(
                   grid_cell * spacing_m - source[:3])),
               "eikonal_minus_fmm_mean_s": float(np.mean(difference)),
               "eikonal_vs_fmm_rmse_s": float(np.sqrt(np.mean(difference ** 2))),
               "demeaned_eikonal_vs_fmm_rmse_s": float(np.std(difference)),
               "fmm_travel_time_s": fmm_times.tolist(),
               "eikonal_travel_time_s": neural_times.tolist()}
        rows.append(row)
        print(f"{entry['file']}: eikonal-FMM mean={row['eikonal_minus_fmm_mean_s']:.4f}s, "
              f"RMSE={row['eikonal_vs_fmm_rmse_s']:.4f}s, "
              f"demeaned RMSE={row['demeaned_eikonal_vs_fmm_rmse_s']:.4f}s", flush=True)
    output = {"purpose": "post-hoc carrier vs numerical FMM, spent events only",
              "eikonal_sha256": registered["eikonal_sha256"],
              "velocity_sha256": registered["velocity_sha256"],
              "fmm_grid_spacing_m": spacing_m,
              "fmm_source_grid_rule": "nearest grid cell; quantify source snapping per event",
              "receiver_interpolation": "trilinear",
              "mean_event_rmse_s": float(np.mean([r["eikonal_vs_fmm_rmse_s"] for r in rows])),
              "mean_event_demeaned_rmse_s": float(np.mean([
                  r["demeaned_eikonal_vs_fmm_rmse_s"] for r in rows])),
              "events": rows}
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
