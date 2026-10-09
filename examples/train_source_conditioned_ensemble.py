"""Train four replications of the frozen 5-Hz source-conditioned workflow.

Seed 42 already exists.  The new members use the identical 25 observations,
architecture, and two-stage optimization budget; only random initialization
and mini-batch sampling change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 123, 2028, 4096)
BASELINE = ROOT / "outputs/depth_gap_25_hybrid_1000/conditioned_5Hz.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observations() -> list[Path]:
    paths = [ROOT / "data/observations/window_0to2s" /
             f"train_{index:02d}_0to2s.npz" for index in range(1, 17)]
    paths.extend((
        ROOT / "data/observations/local_density/train_01_x3300.npz",
        ROOT / "data/observations/local_density/train_01_x3700.npz",
        ROOT / "data/observations/local_density/train_09_x6300.npz",
        ROOT / "data/observations/local_density/train_09_x6700.npz",
        ROOT / "data/observations/depth_coverage/train_01_z340.npz",
        ROOT / "data/observations/depth_coverage/train_01_z660.npz",
        ROOT / "data/observations/depth_coverage/train_09_z540.npz",
        ROOT / "data/observations/depth_coverage/train_09_z860.npz",
        ROOT / "data/observations/depth_gap_900/train_09_z900.npz",
    ))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing receiver observations: {missing}")
    return paths


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "experiments/source_conditioned_ensemble_members.json")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed == 42 for seed in args.seeds):
        parser.error("Replication seeds must be unique and must not include existing seed 42")
    records = observations()
    common = [
        "--velocity", str(ROOT / "data/velocity/vp_model.npy"),
        "--frequencies", "5",
        "--travel-time-carrier",
        "--distance-core-m", "20",
        "--wavefield-scale", "1e-5",
        "--receiver-observations", *(str(path) for path in records),
        "--receiver-events-per-batch", "25",
    ]

    members = [{"seed": 42, "checkpoint": str(BASELINE.resolve()),
                "sha256": sha256(BASELINE), "existing": True}]
    for seed in args.seeds:
        receiver_dir = ROOT / f"outputs/source_conditioned_ensemble_seed{seed}_receiver_5000"
        hybrid_dir = ROOT / f"outputs/source_conditioned_ensemble_seed{seed}_hybrid_1000"
        receiver_checkpoint = receiver_dir / "conditioned_5Hz.pth"
        hybrid_checkpoint = hybrid_dir / "conditioned_5Hz.pth"
        if not receiver_checkpoint.is_file():
            run([
                sys.executable, str(ROOT / "train_source_conditioned_pinn.py"),
                *common, "--output", str(receiver_dir), "--epochs", "5000",
                "--learning-rate", "1e-4", "--receiver-only",
                "--receiver-batch-size", "3200", "--save-every", "5000",
                "--seed", str(seed),
            ])
        if not hybrid_checkpoint.is_file():
            run([
                sys.executable, str(ROOT / "train_source_conditioned_pinn.py"),
                *common, "--output", str(hybrid_dir), "--epochs", "1000",
                "--learning-rate", "1e-5", "--receiver-batch-size", "1600",
                "--receiver-loss-weight", "0.03", "--n-phys", "256",
                "--n-ic", "128", "--n-bc-per-face", "16",
                "--initial-checkpoint", str(receiver_checkpoint),
                "--save-every", "1000", "--seed", str(seed),
            ])
        receiver_summary = json.loads(
            (receiver_dir / "training_summary.json").read_text(encoding="utf-8"))
        hybrid_summary = json.loads(
            (hybrid_dir / "training_summary.json").read_text(encoding="utf-8"))
        members.append({
            "seed": seed,
            "checkpoint": str(hybrid_checkpoint.resolve()),
            "sha256": sha256(hybrid_checkpoint),
            "existing": False,
            "receiver_training_seconds": receiver_summary["5Hz"]["seconds"],
            "hybrid_training_seconds": hybrid_summary["5Hz"]["seconds"],
            "total_training_seconds": (receiver_summary["5Hz"]["seconds"]
                                       + hybrid_summary["5Hz"]["seconds"]),
        })
        print(f"Completed ensemble member seed {seed}", flush=True)

    output = {
        "purpose": "five-member source-conditioned deep-ensemble training manifest",
        "ensemble_size": len(members),
        "seeds": [member["seed"] for member in members],
        "only_random_seed_changes": True,
        "training_observation_count": len(records),
        "receiver_stage": {"updates": 5000, "learning_rate": 1e-4,
                           "batch_size": 3200, "events_per_batch": 25},
        "hybrid_stage": {"updates": 1000, "learning_rate": 1e-5,
                         "batch_size": 1600, "events_per_batch": 25,
                         "receiver_loss_weight": 0.03, "n_phys": 256,
                         "n_ic": 128, "n_bc_per_face": 16},
        "members": members,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
