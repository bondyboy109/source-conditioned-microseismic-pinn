"""Generate and crop the eight registered source-coverage expansion events."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "coverage_expansion_5hz.json"
OBS = ROOT / "data" / "observations"
WINDOW = OBS / "window_0to2s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for item in protocol["sources"]:
        full = OBS / item["file"]
        cropped = WINDOW / f"{full.stem}_0to2s.npz"
        if not full.exists():
            if not args.generate:
                print(f"Missing: {full}")
                continue
            command = [sys.executable, str(ROOT / "examples" / "generate_fd_observation.py"),
                       "--source", *(str(v) for v in item["source_m_s"]),
                       "--output", str(full)]
            print(f"Generating {full.name}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        if not cropped.exists():
            command = [sys.executable, str(ROOT / "examples" / "crop_observations.py"),
                       "--input", str(full), "--output-dir", str(WINDOW), "--end-s", "2"]
            subprocess.run(command, cwd=ROOT, check=True)
        with np.load(full) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
            data = np.asarray(archive["data"])
        if not np.allclose(truth, item["source_m_s"], atol=1e-5, rtol=0):
            raise ValueError(f"Source mismatch: {full}")
        if data.shape != (25, 4501) or not np.isfinite(data).all():
            raise ValueError(f"Invalid FD data: {full}")
        with np.load(cropped) as archive:
            if archive["data"].shape != (25, 1001):
                raise ValueError(f"Invalid cropped data: {cropped}")
        print(f"Verified {item['file']}", flush=True)


if __name__ == "__main__":
    main()
