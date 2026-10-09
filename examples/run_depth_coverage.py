"""Run the registered 20-vs-24 source depth-coverage development comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / "data" / "observations"
DEPTH = OBS / "depth_coverage"
PROTOCOL = ROOT / "experiments" / "depth_coverage_5hz.json"
OLD_SUMMARY = ROOT / "outputs" / "local_density_aug20_scratch_hybrid_1000" / "training_summary.json"
BASE = "depth_coverage"


def execute(*args: str) -> None:
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def verify_sources(paths: list[Path], expected: list[dict], role: str) -> None:
    for path, item in zip(paths, expected):
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise FileNotFoundError(f"Missing {role} event and FD provenance: {path}")
        with np.load(path) as archive:
            source = archive["source_truth"]
            data = archive["data"]
        if (not np.allclose(source, item["source_m_s"], atol=1e-5, rtol=0)
                or data.shape != (25, 1001)):
            raise ValueError(f"Unexpected {role} event: {path}")


def train(group: str, paths: list[Path], protocol: dict) -> Path:
    count = len(paths)
    if count not in (20, 24):
        raise ValueError("Expected either the 20-event control or 24-event depth group")
    stem = f"{BASE}_{group}"
    receiver_dir = ROOT / "outputs" / f"{stem}_receiver_5000"
    hybrid_dir = ROOT / "outputs" / f"{stem}_hybrid_1000"
    base_args = ["--velocity", str(ROOT / "data" / "velocity" / "vp_model.npy"),
                 "--frequencies", "5", "--seed", "42", "--travel-time-carrier",
                 "--reference-velocity", "3500", "--distance-core-m", "20",
                 "--wavefield-scale", "1e-5", "--receiver-observations",
                 *(str(path) for path in paths)]
    receiver = receiver_dir / "conditioned_5Hz.pth"
    if not receiver.exists():
        if receiver_dir.exists():
            raise RuntimeError(f"Incomplete output directory, inspect before retrying: {receiver_dir}")
        run = protocol["receiver_pretraining"]
        execute("train_source_conditioned_pinn.py", *base_args,
                "--output", str(receiver_dir), "--receiver-only",
                "--epochs", str(run["updates"]), "--save-every", str(run["updates"] + 1),
                "--learning-rate", str(run["learning_rate"]),
                "--receiver-loss-weight", str(run["receiver_loss_weight"]),
                "--receiver-batch-size", str(run[f"batch_size_{count}"]),
                "--receiver-events-per-batch", str(count))
    final = hybrid_dir / "conditioned_5Hz.pth"
    if not final.exists():
        if hybrid_dir.exists():
            raise RuntimeError(f"Incomplete output directory, inspect before retrying: {hybrid_dir}")
        run = protocol["hybrid_training"]
        execute("train_source_conditioned_pinn.py", *base_args,
                "--output", str(hybrid_dir), "--initial-checkpoint", str(receiver),
                "--epochs", str(run["updates"]), "--save-every", str(run["updates"] + 1),
                "--learning-rate", str(run["learning_rate"]),
                "--receiver-loss-weight", str(run["receiver_loss_weight"]),
                "--receiver-batch-size", str(run[f"batch_size_{count}"]),
                "--receiver-events-per-batch", str(count),
                "--n-phys", str(run["n_phys"]), "--n-ic", str(run["n_ic"]),
                "--n-bc-per-face", str(run["n_bc_per_face"]))
    summary = json.loads((hybrid_dir / "training_summary.json").read_text(encoding="utf-8"))
    recorded = summary["run"]["receiver_observations"]
    if [Path(row["path"]).resolve() for row in recorded] != [p.resolve() for p in paths]:
        raise ValueError(f"Checkpoint training records differ from registered {group} group")
    return final


def audit(group: str, checkpoint: Path, dev_paths: list[Path],
          anchors: list[Path], train_paths: list[Path]) -> None:
    prefix = ROOT / "experiments" / f"{BASE}_{group}"
    execute("examples/audit_forward_error_decomposition.py", checkpoint, *dev_paths,
            "--evaluation-label", "development", "--output", f"{prefix}_development.json")
    execute("examples/audit_forward_error_decomposition.py", checkpoint,
            *anchors, *(train_paths if group == "aug24" else []),
            "--evaluation-label", "training-diagnostic",
            "--output", f"{prefix}_training_audit.json")
    for anchor, displaced, label in ((anchors[0], dev_paths[:2], "01"),
                                     (anchors[1], dev_paths[2:], "09")):
        execute("examples/audit_spatial_source_response.py", checkpoint, anchor, *displaced,
                "--output", f"{prefix}_response_{label}.json")


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    old = json.loads(OLD_SUMMARY.read_text(encoding="utf-8"))
    base = [Path(row["path"]) for row in old["run"]["receiver_observations"]]
    if len(base) != 20 or len(set(base)) != 20:
        raise ValueError("Original training manifest must have exactly 20 distinct FD events")
    added = [DEPTH / row["file"] for row in protocol["added_training_sources"]]
    dev = [DEPTH / row["file"] for row in protocol["development_sources"]]
    verify_sources(added, protocol["added_training_sources"], "training")
    verify_sources(dev, protocol["development_sources"], "development")
    sources = []
    for path in [*base, *added, *dev]:
        with np.load(path) as archive:
            sources.append(tuple(np.round(archive["source_truth"], 4)))
    if len(set(sources)) != len(sources):
        raise ValueError("Training and development source coordinates overlap")
    anchors = [OBS / "window_0to2s" / f"train_{label}_0to2s.npz"
               for label in ("01", "09")]
    for group, paths in (("control20", base), ("aug24", [*base, *added])):
        final = train(group, paths, protocol)
        audit(group, final, dev, anchors, added)


if __name__ == "__main__":
    main()
