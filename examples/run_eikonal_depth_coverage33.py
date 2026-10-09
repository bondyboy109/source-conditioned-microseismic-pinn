"""Registered 29-versus-33 source depth coverage development run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_eikonal_depth_sweep import DATA, events_from_protocol
from examples.run_fmm_seeded_waveform_development import hash_file

PROTOCOL = ROOT / "experiments/eikonal_depth_coverage33_development_protocol.json"


def preflight(protocol: dict) -> tuple[list[Path], list[dict]]:
    for key, digest in (("baseline_checkpoint", "baseline_sha256"),
                        ("baseline_training_summary", "baseline_training_summary_sha256"),
                        ("eikonal_checkpoint", "eikonal_sha256"),
                        ("velocity_file", "velocity_sha256"),
                        ("original_development_result", "original_development_sha256")):
        if hash_file(ROOT / protocol[key]) != protocol[digest]:
            raise ValueError(f"Registered input changed: {protocol[key]}")
    training_run = json.loads((ROOT / protocol["baseline_training_summary"]).read_text())["run"]
    if (training_run["seed"] != 42
            or training_run["architecture"] != "eikonal_travel_time_carrier"
            or training_run["receiver_events_per_batch"] != 29
            or training_run["eikonal_checkpoint"]
            != str((ROOT / protocol["eikonal_checkpoint"]).resolve())):
        raise ValueError("Baseline configuration differs from registered comparison")
    original = [Path(record["path"]) for record in training_run["receiver_observations"]]
    if len(original) != 29 or len(set(original)) != 29:
        raise ValueError("Original manifest must contain 29 distinct sources")
    for item, path in zip(training_run["receiver_observations"], original):
        if hash_file(path) != item["sha256"]:
            raise ValueError(f"Original training observation changed: {path}")
    depth = json.loads((ROOT / protocol["depth_protocol"]).read_text())
    prior = json.loads((ROOT / protocol["original_development_result"]).read_text())
    events = events_from_protocol(depth)
    if (len(events) != 12 or len(prior["events"]) != 12
            or {e["anchor"] for e in events} != set(protocol["anchors"])
            or protocol["training_offset_m"] != [-160, 160]
            or protocol["evaluation_offset_m"] != [-80, 80, -280, 280]):
        raise ValueError("Unexpected development split")
    added = []
    evaluated = []
    original_sources = [tuple(np.round(item["source_truth"], 4))
                        for item in training_run["receiver_observations"]]
    for event, old in zip(events, prior["events"]):
        path = DATA / event["file"]
        if event["file"] != old["file"] or hash_file(path) != old["sha256"]:
            raise ValueError(f"Registered depth record changed: {path}")
        with np.load(path) as archive:
            if not np.allclose(archive["source_truth"], event["source_m_s"], rtol=0, atol=1e-5):
                raise ValueError(f"Source coordinates changed: {path}")
        if tuple(np.round(event["source_m_s"], 4)) in original_sources:
            raise ValueError(f"Development record was in original training: {path}")
        if event["offset_m"] in protocol["training_offset_m"]:
            added.append(path)
        elif event["offset_m"] in protocol["evaluation_offset_m"]:
            evaluated.append(event)
        else:
            raise ValueError(f"Unassigned depth record: {path}")
    if len(added) != 4 or len(evaluated) != 8 or set(added) & {DATA / r["file"] for r in evaluated}:
        raise ValueError("Training/evaluation split is not disjoint 4/8")
    print(f"Verified 29 original plus 4 depth training records; "
          f"8 disjoint spent development records", flush=True)
    return original + added, evaluated


def train(args: list[str]) -> None:
    subprocess.run([sys.executable, str(ROOT / "train_source_conditioned_pinn.py"), *args],
                   cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    training, _ = preflight(protocol)
    receiver_dir = ROOT / protocol["receiver_output"]
    hybrid_dir = ROOT / protocol["hybrid_output"]
    if receiver_dir.exists() or hybrid_dir.exists():
        raise FileExistsError("Depth-coverage output already exists; inspect before rerunning")
    if args.preflight:
        return
    common = ["--frequencies", "5", "--seed", "42", "--travel-time-carrier",
              "--eikonal-checkpoint", str(ROOT / protocol["eikonal_checkpoint"]),
              "--reference-velocity", "3500", "--distance-core-m", "20",
              "--wavefield-scale", "1e-5", "--receiver-observations",
              *(str(path) for path in training)]
    train([*common, "--output", str(receiver_dir), "--receiver-only",
           "--epochs", "5000", "--save-every", "5001", "--learning-rate", "1e-4",
           "--receiver-loss-weight", "0.1", "--receiver-batch-size", "4224",
           "--receiver-events-per-batch", "33"])
    train([*common, "--output", str(hybrid_dir),
           "--initial-checkpoint", str(receiver_dir / "conditioned_5Hz.pth"),
           "--epochs", "1000", "--save-every", "1001", "--learning-rate", "1e-5",
           "--receiver-loss-weight", "0.03", "--receiver-batch-size", "2112",
           "--receiver-events-per-batch", "33", "--n-phys", "256",
           "--n-ic", "128", "--n-bc-per-face", "16"])
    print(f"Saved candidate: {hybrid_dir / 'conditioned_5Hz.pth'}", flush=True)


if __name__ == "__main__":
    main()
