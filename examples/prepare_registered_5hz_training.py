"""Prepare only the training sources from the registered 5 Hz protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "5hz_protocol.json"
OUTPUT = ROOT / "data" / "observations"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Generate missing FD TRAINING sources; blind files are never generated")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    training = protocol["data"]["training_sources"]
    blind = protocol["data"]["blind_sources"]
    all_files = [item["file"] for item in training + blind]
    if len(set(all_files)) != len(all_files):
        raise ValueError("Duplicate experiment file names")
    all_sources = [tuple(item["source"]) for item in training + blind]
    if len(set(all_sources)) != len(all_sources):
        raise ValueError("Training and blind sources overlap")
    present_blind = [item["file"] for item in blind if (OUTPUT / item["file"]).exists()]
    if present_blind:
        raise RuntimeError(f"Blind files already exist before model freeze: {present_blind}")

    for item in training:
        path = OUTPUT / item["file"]
        if not path.exists():
            if not args.generate:
                print(f"Missing training file: {path}")
                continue
            command = [sys.executable, str(ROOT / "examples" / "generate_fd_observation.py"),
                       "--output", str(path), "--source", *(str(v) for v in item["source"])]
            print(f"Generating {item['file']}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
            data = np.asarray(archive["data"])
        if not np.allclose(truth, item["source"], rtol=0, atol=1e-5):
            raise ValueError(f"Source does not match registered protocol: {path}")
        if data.shape != (25, 4501) or not np.isfinite(data).all():
            raise ValueError(f"Invalid training waveform: {path}")
        report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if report.get("frequency_hz") != protocol["frequency_hz"]:
            raise ValueError(f"Wrong FD frequency for {path}")
        print(f"Verified {item['file']}: source={truth.tolist()}, peak={np.abs(data).max():.3e}",
              flush=True)
    print("Blind source waveforms were not generated or opened.")


if __name__ == "__main__":
    main()
