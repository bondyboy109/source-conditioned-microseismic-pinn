"""Generate the locked FD source set only after development selection is saved."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/arrival_constrained_fresh_test_protocol.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    decision = ROOT / spec["development_result"]
    if not decision.is_file():
        raise FileNotFoundError("Finish and lock development selection before generating test records")
    selection = json.loads(decision.read_text(encoding="utf-8"))
    if selection["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Protocol changed after development selection")
    target_dir = ROOT / spec["test_directory"]
    events = []
    for entry in spec["test_events"]:
        path = target_dir / (entry["id"] + ".npz")
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(f"Refusing to overwrite an existing FD event: {path}")
        cmd = [sys.executable, "-u", str(ROOT / "examples/generate_fd_observation.py"),
               "--config", str(ROOT / spec["fd_config"]), "--output", str(path),
               "--source", *(str(x) for x in entry["source_m_s"])]
        print(f"Generating {entry['id']}: {entry['source_m_s']}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        with np.load(path) as archive:
            if not np.allclose(archive["source_truth"], entry["source_m_s"], atol=1e-5):
                raise ValueError(f"FD source differs from protocol: {path}")
        events.append({"id": entry["id"], "source_m_s": entry["source_m_s"],
                       "observation": str(path), "sha256": digest(path),
                       "fd_report_sha256": digest(path.with_suffix(".json"))})
    manifest = {"protocol_sha256": digest(PROTOCOL),
                "development_decision_sha256": digest(decision),
                "events": events}
    destination = ROOT / "experiments/arrival_constrained_fresh_test_manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
