"""Development-only leave-one-source-out test on the eight registered training events."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "5hz_protocol.json"
OBSERVATIONS = ROOT / "data" / "observations"


def run(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(command, cwd=ROOT, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "dev_loo_relative_carrier_5hz")
    parser.add_argument("--window-dir", type=Path, default=None,
                        help="Use registered cropped files named <stem>_0to2s.npz")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    training = protocol["data"]["training_sources"]
    if len(training) != 8 or len({item["file"] for item in training}) != 8:
        raise ValueError("Expected eight distinct registered training files")
    if args.window_dir is None:
        paths = [OBSERVATIONS / item["file"] for item in training]
    else:
        paths = [args.window_dir / f"{Path(item['file']).stem}_0to2s.npz"
                 for item in training]
    sources = []
    for item, path in zip(training, paths):
        with np.load(path) as archive:
            source = np.asarray(archive["source_truth"], dtype=np.float64)
        if source.shape != (4,) or not np.allclose(
                source, item["source"], rtol=0, atol=1e-5):
            raise ValueError(f"Source does not match protocol: {path}")
        sources.append(source)
    output.mkdir(parents=True, exist_ok=True)
    results = []

    for index, (heldout, path) in enumerate(zip(training, paths)):
        fold = output / f"leave_{index + 1:02d}"
        fold.mkdir()
        others = [p for j, p in enumerate(paths) if j != index]
        train_command = [
            sys.executable, "-u", str(ROOT / "train_source_conditioned_pinn.py"),
            "--frequencies", "5", "--epochs", "5000", "--save-every", "5000",
            "--seed", "42", "--travel-time-carrier", "--reference-velocity", "3500",
            "--hard-initial", "--wavefield-scale", "0.00001", "--receiver-only",
            "--receiver-observations", *(str(p) for p in others),
            "--receiver-batch-size", "1024", "--receiver-loss-weight", "0.1",
            "--output", str(fold),
        ]
        print(f"Fold {index + 1}/8: train on seven, evaluate {heldout['file']}",
              flush=True)
        run(train_command, fold / "train.log")
        report_path = fold / "audit.json"
        run([sys.executable, str(ROOT / "examples" / "audit_forward_waveforms.py"),
             str(fold / "conditioned_5Hz.pth"), str(path),
             "--output", str(report_path)], fold / "audit.log")
        audit = json.loads(report_path.read_text(encoding="utf-8"))
        if audit["source_in_receiver_training_set"] or audit["file_in_receiver_training_set"]:
            raise RuntimeError(f"Held-out event leaked into fold {index + 1}")
        recorded = {Path(entry["path"]).name for entry in audit["checkpoint_training_sources"]}
        if recorded != {p.name for p in others}:
            raise RuntimeError(f"Training source set is wrong for fold {index + 1}")

        distances = [float(np.linalg.norm(sources[index][:3] - source[:3]))
                     if j != index else float("inf")
                     for j, source in enumerate(sources)]
        nearest = int(np.argmin(distances))
        result = {
            "held_out": heldout["file"],
            "source_m_s": sources[index].tolist(),
            "nearest_training": training[nearest]["file"],
            "nearest_distance_m": distances[nearest],
            "nearest_depth_difference_m": abs(sources[index][2] - sources[nearest][2]),
            "nearest_origin_time_difference_s": abs(sources[index][3] - sources[nearest][3]),
            "relative_l2": audit["relative_l2"],
            "fd_peak_abs": audit["fd_peak_abs"],
            "predicted_peak_abs": audit["pinn_peak_abs"],
            "training_seconds": json.loads((fold / "training_summary.json").read_text(
                encoding="utf-8"))["5Hz"]["seconds"],
        }
        results.append(result)
        (output / "summary.json").write_text(json.dumps({
            "purpose": "development-only leave-one-out supervised representation diagnostic",
            "architecture": "travel_time_carrier", "reference_velocity_m_s": 3500,
            "steps_per_fold": 5000, "completed_folds": len(results),
            "zero_prediction_relative_l2": 1.0, "results": results,
        }, indent=2), encoding="utf-8")
        print(f"  nearest={distances[nearest]:.0f} m, "
              f"relative L2={result['relative_l2']:.4f}", flush=True)

    print(f"Passed below 1: {sum(r['relative_l2'] < 1 for r in results)}/8", flush=True)
    print(f"Saved {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
