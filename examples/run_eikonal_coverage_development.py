"""Train and audit a 29-source eikonal-carrier development model."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/eikonal_coverage_development_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args) -> None:
    subprocess.run([sys.executable, *(str(value) for value in args)], cwd=ROOT,
                   check=True)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    control = ROOT / protocol["control_checkpoint"]
    if sha256(control) != protocol["control_sha256"]:
        raise ValueError("The registered straight-carrier control changed")
    summary = json.loads((control.parent / "training_summary.json").read_text())
    training = [Path(row["path"]) for row in
                summary["run"]["receiver_observations"]]
    if len(training) != 29 or len(set(training)) != 29:
        raise ValueError("Control manifest must contain exactly 29 distinct events")
    for path in training:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as archive:
            if archive["data"].shape != (25, 1001):
                raise ValueError(f"Unexpected training record shape: {path}")
    eikonal = ROOT / protocol["eikonal_checkpoint"]
    if not eikonal.is_file():
        raise FileNotFoundError(eikonal)
    receiver_dir = ROOT / "outputs/eikonal_coverage_29_receiver_5000"
    hybrid_dir = ROOT / "outputs/eikonal_coverage_29_hybrid_1000"
    receiver = receiver_dir / "conditioned_5Hz.pth"
    final = hybrid_dir / "conditioned_5Hz.pth"
    common = ["--frequencies", 5, "--seed", 42, "--travel-time-carrier",
              "--eikonal-checkpoint", eikonal, "--reference-velocity", 3500,
              "--distance-core-m", 20, "--wavefield-scale", "1e-5",
              "--receiver-observations", *training]
    if not receiver.exists():
        if receiver_dir.exists():
            raise RuntimeError(f"Partial output requires inspection: {receiver_dir}")
        run("train_source_conditioned_pinn.py", *common, "--output", receiver_dir,
            "--receiver-only", "--epochs", 5000, "--save-every", 5001,
            "--learning-rate", "1e-4", "--receiver-loss-weight", 0.1,
            "--receiver-batch-size", 3712, "--receiver-events-per-batch", 29)
    if not final.exists():
        if hybrid_dir.exists():
            raise RuntimeError(f"Partial output requires inspection: {hybrid_dir}")
        run("train_source_conditioned_pinn.py", *common, "--output", hybrid_dir,
            "--initial-checkpoint", receiver, "--epochs", 1000, "--save-every", 1001,
            "--learning-rate", "1e-5", "--receiver-loss-weight", 0.03,
            "--receiver-batch-size", 1856, "--receiver-events-per-batch", 29,
            "--n-phys", 256, "--n-ic", 128, "--n-bc-per-face", 16)
    final_summary = json.loads((hybrid_dir / "training_summary.json").read_text())
    recorded = [Path(row["path"]) for row in
                final_summary["run"]["receiver_observations"]]
    if [path.resolve() for path in recorded] != [path.resolve() for path in training]:
        raise ValueError("Eikonal checkpoint manifest differs from control")
    observations = sorted((ROOT / "data/observations/fmm_seeded_independent").glob("*.npz"))
    if len(observations) != 8:
        raise ValueError("Expected eight previously registered evaluation records")
    control_report = ROOT / "experiments/eikonal_coverage_control_forward.json"
    eikonal_report = ROOT / "experiments/eikonal_coverage_augmented_forward.json"
    run("examples/audit_forward_error_decomposition.py", control, *observations,
        "--evaluation-label", "independent-reaudit", "--output", control_report)
    run("examples/audit_forward_error_decomposition.py", final, *observations,
        "--evaluation-label", "independent-reaudit", "--output", eikonal_report)
    before = json.loads(control_report.read_text())
    after = json.loads(eikonal_report.read_text())
    rows = []
    for old, new in zip(before["events"], after["events"]):
        if old["sha256"] != new["sha256"]:
            raise ValueError("Control and eikonal reports are not paired")
        rows.append({"event": Path(old["observation"]).name,
                     "control_raw_l2": old["original_amplitude_relative_l2"],
                     "eikonal_raw_l2": new["original_amplitude_relative_l2"],
                     "control_shift_gain_l2": old["common_time_shift_and_gain_diagnostic_only"]["relative_l2"],
                     "eikonal_shift_gain_l2": new["common_time_shift_and_gain_diagnostic_only"]["relative_l2"]})
    result = {"purpose": "post-hoc eikonal-carrier forward development audit",
              "protocol": str(PROTOCOL.resolve()), "control_sha256": sha256(control),
              "eikonal_checkpoint_sha256": sha256(final),
              "control_training_seconds": json.loads((control.parent / "training_summary.json").read_text())["5Hz"]["seconds"],
              "receiver_training_seconds": final_summary["5Hz"]["seconds"],
              "rows": rows,
              "control_mean_raw_l2": float(np.mean([row["control_raw_l2"] for row in rows])),
              "eikonal_mean_raw_l2": float(np.mean([row["eikonal_raw_l2"] for row in rows])),
              "control_mean_shift_gain_l2": float(np.mean([row["control_shift_gain_l2"] for row in rows])),
              "eikonal_mean_shift_gain_l2": float(np.mean([row["eikonal_shift_gain_l2"] for row in rows]))}
    target = ROOT / "experiments/eikonal_coverage_development_results.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
