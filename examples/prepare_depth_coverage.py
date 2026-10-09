"""Generate the registered 5 Hz local-depth training and development events."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "depth_coverage_5hz.json"
OUTPUT = ROOT / "data" / "observations" / "depth_coverage"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Generate missing registered events with the FD solver")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    entries = protocol["added_training_sources"] + protocol["development_sources"]
    sources = [tuple(item["source_m_s"]) for item in entries]
    files = [item["file"] for item in entries]
    if len(set(sources)) != len(sources) or len(set(files)) != len(files):
        raise ValueError("Registered sources and filenames must be unique")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for item in entries:
        path = OUTPUT / item["file"]
        if not path.exists():
            if not args.generate:
                print(f"Missing: {path}")
                continue
            command = [
                sys.executable, str(ROOT / "examples" / "generate_fd_observation.py"),
                "--duration", "2", "--output", str(path),
                "--source", *(str(value) for value in item["source_m_s"]),
            ]
            print(f"Generating {item['role']} event {path.name}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        with np.load(path) as archive:
            data = np.asarray(archive["data"])
            times = np.asarray(archive["times"], dtype=np.float32)
            source = np.asarray(archive["source_truth"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"])
        if data.shape != (25, 1001) or not np.isfinite(data).all():
            raise ValueError(f"Invalid waveform array: {path}")
        if not np.allclose(source, item["source_m_s"], rtol=0, atol=1e-5):
            raise ValueError(f"Source truth mismatch: {path}")
        if not np.array_equal(times, np.arange(1001, dtype=np.float32) * 0.002):
            raise ValueError(f"Time axis mismatch: {path}")
        if receivers.shape != (25, 3):
            raise ValueError(f"Receiver geometry mismatch: {path}")
        report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if report.get("frequency_hz") != 5.0:
            raise ValueError(f"Wrong source frequency: {path}")
        print(f"Verified {path.name}: {source.tolist()}, peak={np.abs(data).max():.3e}",
              flush=True)


if __name__ == "__main__":
    main()
