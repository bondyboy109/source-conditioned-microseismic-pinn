"""Save a reproducible receiver layout and a draft acquisition configuration.

This script does not simulate waveforms or assert that FD data are accurate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "observations"
MODEL = ROOT / "data" / "velocity" / "vp_model.npy"
DOMAIN_M = (16000.0, 16000.0, 3720.0)
SOURCE_LOW = (2000.0, 2000.0, 100.0)
SOURCE_HIGH = (14000.0, 14000.0, 1500.0)
ORIGIN_TIME_MAX_S = 0.5
DT_S = 0.002
RECORD_LENGTH_S = 9.0


def main() -> None:
    velocity = np.load(MODEL, mmap_mode="r")
    if velocity.shape != (401, 401, 94):
        raise ValueError(f"Expected the 40 m velocity export, got {velocity.shape}")
    v_min = float(velocity.min())
    v_max = float(velocity.max())
    if not (0 < v_min <= v_max and np.isfinite(v_max)):
        raise ValueError("Velocity must be finite and positive")

    # A two-dimensional surface array gives coverage along both horizontal axes.
    x, y = np.meshgrid(
        np.linspace(2000.0, 14000.0, 5),
        np.linspace(2000.0, 14000.0, 5),
        indexing="ij",
    )
    receivers = np.column_stack(
        (x.ravel(), y.ravel(), np.full(x.size, 100.0))
    ).astype(np.float32)
    if not np.all((receivers >= 0) & (receivers <= np.array(DOMAIN_M))):
        raise ValueError("Receiver outside the modeled domain")

    # Worst-case straight-line distance among the source-box corners; this is
    # a conservative planning estimate, not an eikonal or FD travel time.
    corners = np.array(
        [[sx, sy, sz]
         for sx in (SOURCE_LOW[0], SOURCE_HIGH[0])
         for sy in (SOURCE_LOW[1], SOURCE_HIGH[1])
         for sz in (SOURCE_LOW[2], SOURCE_HIGH[2])],
        dtype=np.float64,
    )
    farthest_m = float(np.linalg.norm(
        receivers[:, None, :] - corners[None, :, :], axis=-1
    ).max())
    latest_estimate_s = ORIGIN_TIME_MAX_S + farthest_m / v_min
    if latest_estimate_s >= RECORD_LENGTH_S:
        raise ValueError("Record is shorter than the conservative arrival estimate")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT / "receiver_coords.npy", receivers)
    configuration = {
        "status": "draft_geometry_only_no_observed_waveforms",
        "velocity_file": str(MODEL),
        "velocity_shape": list(velocity.shape),
        "grid_spacing_m": [40.0, 40.0, 40.0],
        "domain_extent_m": list(DOMAIN_M),
        "velocity_min_max_m_per_s": [v_min, v_max],
        "receivers_file": str(OUTPUT / "receiver_coords.npy"),
        "receiver_count": int(len(receivers)),
        "receiver_layout": "5 x 5 surface grid; x,y=2000..14000 m; z=100 m",
        "source_location_bounds_m": [list(SOURCE_LOW), list(SOURCE_HIGH)],
        "origin_time_bounds_s": [0.05, ORIGIN_TIME_MAX_S],
        "time_step_s": DT_S,
        "record_length_s": RECORD_LENGTH_S,
        "n_time_samples": int(round(RECORD_LENGTH_S / DT_S)) + 1,
        "conservative_latest_arrival_s": latest_estimate_s,
        "frequency_validity": "60 Hz requires a refined-grid dispersion study",
        "boundary_condition": "PINN zero Dirichlet; FD matching pending",
    }
    path = OUTPUT / "experiment_draft.json"
    path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(f"Saved {len(receivers)} receivers: {OUTPUT / 'receiver_coords.npy'}")
    print(f"Draft configuration: {path}")
    print(f"Latest straight-line estimate: {latest_estimate_s:.2f} s")
    print("No observation waveforms were generated.")


if __name__ == "__main__":
    main()
