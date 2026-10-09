"""Reproduce the registered paired 29-source onset-gate training run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import hash_file

PROTOCOL = ROOT / "experiments/trained_onset_gate_development_protocol.json"


def check_inputs(protocol: dict) -> list[Path]:
    registered = (("baseline_checkpoint", "baseline_sha256"),
                  ("baseline_training_summary", "baseline_training_summary_sha256"),
                  ("eikonal_checkpoint", "eikonal_sha256"),
                  ("velocity_file", "velocity_sha256"),
                  ("development_baseline", "development_baseline_sha256"))
    for filename, digest in registered:
        if hash_file(ROOT / protocol[filename]) != protocol[digest]:
            raise ValueError(f"Registered input changed: {protocol[filename]}")
    velocity_max = float(np.max(np.load(ROOT / protocol["velocity_file"], mmap_mode="r")))
    if velocity_max != protocol["onset_gate"]["maximum_speed_m_s"]:
        raise ValueError("Registered onset maximum speed differs from velocity grid")
    baseline = json.loads((ROOT / protocol["development_baseline"]).read_text())
    if (baseline["checkpoint_sha256"]["eikonal"] != protocol["baseline_sha256"]
            or baseline["maximum_speed_m_s"] != velocity_max):
        raise ValueError("Development comparison does not match registered inputs")
    run = json.loads((ROOT / protocol["baseline_training_summary"]).read_text())["run"]
    if (run["seed"] != 42 or run["architecture"] != "eikonal_travel_time_carrier"
            or run["eikonal_checkpoint"] != str((ROOT / protocol["eikonal_checkpoint"]).resolve())
            or run["receiver_events_per_batch"] != 29):
        raise ValueError("Control training configuration differs from registered design")
    paths = [Path(item["path"]) for item in run["receiver_observations"]]
    if len(paths) != 29 or len(set(paths)) != 29:
        raise ValueError("Expected 29 distinct FD training observations")
    for item, path in zip(run["receiver_observations"], paths):
        if hash_file(path) != item["sha256"]:
            raise ValueError(f"Training observation changed: {path}")
    return paths


def train(args: list[str]) -> None:
    subprocess.run([sys.executable, str(ROOT / "train_source_conditioned_pinn.py"),
                    *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    training = check_inputs(protocol)
    receiver_dir = ROOT / protocol["receiver_output"]
    hybrid_dir = ROOT / protocol["hybrid_output"]
    if receiver_dir.exists() or hybrid_dir.exists():
        raise FileExistsError("Gate run output already exists; inspect before rerunning")
    print(f"Verified {len(training)} training events and all registered hashes", flush=True)
    if args.preflight:
        return
    gate = protocol["onset_gate"]
    common = ["--frequencies", "5", "--seed", "42", "--travel-time-carrier",
              "--eikonal-checkpoint", str(ROOT / protocol["eikonal_checkpoint"]),
              "--reference-velocity", "3500", "--distance-core-m", "20",
              "--wavefield-scale", "1e-5", "--onset-gate",
              "--onset-max-speed", str(gate["maximum_speed_m_s"]),
              "--onset-support-m", str(gate["support_m"]),
              "--onset-lead-s", str(gate["lead_s"]),
              "--onset-rise-s", str(gate["rise_s"]),
              "--receiver-observations", *(str(path) for path in training)]
    train([*common, "--output", str(receiver_dir), "--receiver-only",
           "--epochs", "5000", "--save-every", "5001",
           "--learning-rate", "1e-4", "--receiver-loss-weight", "0.1",
           "--receiver-batch-size", "3712", "--receiver-events-per-batch", "29"])
    initial = receiver_dir / "conditioned_5Hz.pth"
    train([*common, "--output", str(hybrid_dir), "--initial-checkpoint", str(initial),
           "--epochs", "1000", "--save-every", "1001",
           "--learning-rate", "1e-5", "--receiver-loss-weight", "0.03",
           "--receiver-batch-size", "1856", "--receiver-events-per-batch", "29",
           "--n-phys", "256", "--n-ic", "128", "--n-bc-per-face", "16"])
    print(f"New frozen candidate: {hybrid_dir / 'conditioned_5Hz.pth'}", flush=True)


if __name__ == "__main__":
    main()
