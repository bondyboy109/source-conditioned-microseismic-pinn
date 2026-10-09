"""Development-only 25-vs-29 source forward-model coverage pilot."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_cross_region_fmm_comparison import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/cross_region_coverage_development_protocol.json"
DATA = ROOT / "data/observations/cross_region_coverage"
EXPERIMENTS = ROOT / "experiments"


def execute(*arguments) -> None:
    subprocess.run([sys.executable, *(str(argument) for argument in arguments)],
                   cwd=ROOT, check=True)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    control = ROOT / protocol["control_checkpoint"]
    if hash_file(control) != protocol["control_sha256"]:
        raise ValueError("Control checkpoint differs from the registered model")
    _, metadata = load_conditioned_checkpoint(control, torch.device("cpu"))
    original = [Path(row["path"]) for row in
                metadata["training_record"]["receiver_observations"]]
    if len(original) != 25 or len(set(original)) != 25:
        raise ValueError("Control must contain exactly 25 training events")
    new_entries = protocol["training_sources"] + protocol["development_sources"]
    desired = [np.asarray(row["source_m_s"], dtype=np.float32) for row in new_entries]
    if (len(new_entries) != 8 or len({row["file"] for row in new_entries}) != 8
            or any(np.allclose(a, b, rtol=0, atol=1e-4) for i, a in enumerate(desired)
                   for b in desired[i+1:])):
        raise ValueError("New files and source coordinates must be distinct")
    for row in metadata["training_record"]["receiver_observations"]:
        if any(np.allclose(row["source_truth"], source, atol=1e-4, rtol=0)
               for source in desired):
            raise ValueError("New data would overlap control training")
    # Inspect archived coordinates before generating this pilot's FD records.
    for path in (ROOT / "data/observations").rglob("*.npz"):
        if DATA in path.parents:
            continue
        with np.load(path) as archive:
            if "source_truth" not in archive.files:
                continue
            previous = archive["source_truth"]
        if any(np.allclose(previous, source, atol=1e-4, rtol=0)
               for source in desired):
            raise ValueError(f"New source previously archived: {path}")
    DATA.mkdir(parents=True, exist_ok=True)
    for row in new_entries:
        path = DATA / row["file"]
        if not path.is_file():
            print(f"Generating {row['file']}", flush=True)
            execute("examples/generate_fd_observation.py", "--duration", 2,
                    "--source", *row["source_m_s"], "--output", path)
        with np.load(path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.allclose(archive["source_truth"], row["source_m_s"],
                                       rtol=0, atol=1e-5)):
                raise ValueError(f"Invalid FD record: {path}")
        provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (provenance["frequency_hz"] != 5 or provenance["duration_s"] != 2
                or provenance["boundary"] != "zero Dirichlet on all six faces"):
            raise ValueError(f"FD setup differs: {path}")
    train_additions = [DATA / entry["file"] for entry in protocol["training_sources"]]
    development = [DATA / entry["file"] for entry in protocol["development_sources"]]
    all_training = [*original, *train_additions]
    receiver_dir = ROOT / "outputs/cross_region_coverage_29_receiver_5000"
    hybrid_dir = ROOT / "outputs/cross_region_coverage_29_hybrid_1000"
    receiver = receiver_dir / "conditioned_5Hz.pth"
    final = hybrid_dir / "conditioned_5Hz.pth"
    base_args = ("--frequencies", 5, "--seed", 42, "--travel-time-carrier",
                 "--reference-velocity", 3500, "--distance-core-m", 20,
                 "--wavefield-scale", "1e-5", "--receiver-observations", *all_training)
    for output, checkpoint, settings, is_receiver in (
            (receiver_dir, None, protocol["receiver_training"], True),
            (hybrid_dir, receiver, protocol["hybrid_training"], False)):
        target = output / "conditioned_5Hz.pth"
        if target.is_file():
            continue
        if output.exists():
            raise RuntimeError(f"Partial training run requires inspection: {output}")
        args = ["train_source_conditioned_pinn.py", *base_args,
                "--output", output, "--epochs", settings["updates"],
                "--save-every", settings["updates"] + 1,
                "--learning-rate", settings["learning_rate"],
                "--receiver-loss-weight", settings["weight"],
                "--receiver-batch-size", settings["samples_per_event"] * 29,
                "--receiver-events-per-batch", 29]
        if is_receiver:
            args.append("--receiver-only")
        else:
            args.extend(("--initial-checkpoint", checkpoint,
                         "--n-phys", settings["n_phys"],
                         "--n-ic", settings["n_ic"],
                         "--n-bc-per-face", settings["n_bc_per_face"]))
        print(f"Training {output.name}", flush=True)
        execute(*args)
    final_summary = json.loads((hybrid_dir / "training_summary.json").read_text())
    recorded = final_summary["run"]["receiver_observations"]
    if ([Path(row["path"]).resolve() for row in recorded]
            != [path.resolve() for path in all_training]):
        raise ValueError("New checkpoint has the wrong training manifest")
    reports = {}
    for label, checkpoint in (("control25", control), ("aug29", final)):
        output = EXPERIMENTS / f"cross_region_coverage_{label}_forward.json"
        execute("examples/audit_forward_error_decomposition.py", checkpoint,
                *development, "--evaluation-label", "development", "--output", output)
        reports[label] = json.loads(output.read_text(encoding="utf-8"))
    rows = []
    for old, new in zip(reports["control25"]["events"],
                        reports["aug29"]["events"]):
        if old["sha256"] != new["sha256"]:
            raise ValueError("Control and augmented evaluation files differ")
        rows.append({"file": Path(old["observation"]).name, "sha256": old["sha256"],
                     "control_raw_l2": old["original_amplitude_relative_l2"],
                     "augmented_raw_l2": new["original_amplitude_relative_l2"]})
    control_mean = float(np.mean([row["control_raw_l2"] for row in rows]))
    augmented_mean = float(np.mean([row["augmented_raw_l2"] for row in rows]))
    improvement_count = sum(row["augmented_raw_l2"] < row["control_raw_l2"]
                            for row in rows)
    gates = protocol["development_gates"]
    checks = {
        "mean_improvement": augmented_mean / control_mean <
                            gates["new_event_mean_raw_l2_fraction_of_control_below"],
        "improvement_count": improvement_count >=
                             gates["new_event_improvement_count_at_least"],
        "all_augmented_below": all(row["augmented_raw_l2"] <
                                   gates["new_event_all_raw_l2_below"] for row in rows),
    }
    report = {"purpose": "source-coverage development pilot, not independent confirmation",
              "protocol": str(PROTOCOL), "control_sha256": hash_file(control),
              "augmented_sha256": hash_file(final),
              "receiver_seconds": json.loads((receiver_dir / "training_summary.json").read_text())["5Hz"]["seconds"],
              "hybrid_seconds": final_summary["5Hz"]["seconds"],
              "added_fd_seconds": sum(json.loads(path.with_suffix(".json").read_text())["runtime_seconds"]
                                      for path in train_additions),
              "rows": rows, "control_mean_l2": control_mean,
              "augmented_mean_l2": augmented_mean,
              "gates": checks, "all_gates_passed": all(checks.values())}
    target = EXPERIMENTS / "cross_region_coverage_development_results.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {target}; gates={checks}", flush=True)


if __name__ == "__main__":
    main()
