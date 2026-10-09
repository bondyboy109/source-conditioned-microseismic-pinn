"""Build a uniform FD config and generate two separated source events."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "uniform_calibration_5hz.json"
EXPERIMENT = ROOT / "data" / "observations" / "uniform_3500"
VELOCITY = ROOT / "data" / "velocity" / "uniform_3500.npy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Run the two independent FD simulations")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    shape = tuple(protocol["grid_shape"])
    speed = float(protocol["velocity_m_s"])
    if not VELOCITY.is_file():
        VELOCITY.parent.mkdir(parents=True, exist_ok=True)
        np.save(VELOCITY, np.full(shape, speed, dtype=np.float32))
    velocity = np.load(VELOCITY, mmap_mode="r")
    if velocity.shape != shape or velocity.dtype != np.float32 or np.any(velocity != speed):
        raise ValueError("Existing uniform velocity file differs from registered experiment")
    original = json.loads((ROOT / "data" / "observations" / "experiment_draft.json")
                          .read_text(encoding="utf-8"))
    if original["velocity_shape"] != list(shape):
        raise ValueError("Uniform grid differs from receiver geometry")
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    original.update(velocity_file=str(VELOCITY),
                    velocity_min_max_m_per_s=[speed, speed],
                    boundary_condition=protocol["boundary"],
                    status="development_uniform_3500_5hz")
    config = EXPERIMENT / "fd_config.json"
    config.write_text(json.dumps(original, indent=2), encoding="utf-8")
    for label in ("train", "test"):
        source = protocol[f"{label}_source_m_s"]
        path = EXPERIMENT / f"{label}.npz"
        if not path.exists():
            if not args.generate:
                print(f"Missing: {path}", flush=True)
                continue
            command = [sys.executable, str(ROOT / "examples" / "generate_fd_observation.py"),
                       "--config", str(config), "--source", *(str(v) for v in source),
                       "--output", str(path), "--sponge-width-m",
                       str(protocol["sponge_width_m"]), "--sponge-max-damping-s",
                       str(protocol["sponge_max_damping_s"])]
            print(f"Generating uniform {label} event", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        with np.load(path) as data:
            truth = np.asarray(data["source_truth"], dtype=np.float64)
            waveform = np.asarray(data["data"])
            times = np.asarray(data["times"])
        fd = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if not np.allclose(truth, source, rtol=0, atol=1e-5):
            raise ValueError(f"Uniform source mismatch: {path}")
        if waveform.shape != (25, 4501) or not np.isfinite(waveform).all():
            raise ValueError(f"Uniform waveform invalid: {path}")
        if not np.isclose(times[-1], protocol["record_length_s"]):
            raise ValueError(f"Uniform time axis invalid: {path}")
        if (fd.get("boundary") != protocol["boundary"]
                or fd.get("sponge_width_m") != protocol["sponge_width_m"]
                or fd.get("frequency_hz") != protocol["frequency_hz"]):
            raise ValueError(f"Uniform FD provenance differs from protocol: {path}")
        print(f"Verified uniform {label}: {waveform.shape}, "
              f"peak={np.max(np.abs(waveform)):.3e}", flush=True)


if __name__ == "__main__":
    main()
