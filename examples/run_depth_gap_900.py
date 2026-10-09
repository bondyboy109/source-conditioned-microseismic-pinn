"""Generate and run registered 24-vs-25 local-depth development pilot."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnose_depth_inverse import depth_scan
from invert_source_conditioned import waveform_mse
from source_conditioned_pinn import load_conditioned_checkpoint


PROTOCOL = ROOT / "experiments/depth_gap_900_protocol.json"
DATA = ROOT / "data/observations/depth_gap_900"
BASE = ROOT / "data/observations/depth_coverage"
EXPERIMENTS = ROOT / "experiments"


def run(*args) -> None:
    command = [sys.executable, *(str(value) for value in args)]
    print("Running:", command[1], "with", len(command) - 2, "arguments", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify(path: Path, source: list[float]) -> None:
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise FileNotFoundError(f"FD data or metadata missing: {path}")
    with np.load(path) as archive:
        if (not np.allclose(archive["source_truth"], source, atol=1e-5, rtol=0)
                or archive["data"].shape != (25, 1001)):
            raise ValueError(f"FD source or sample shape mismatch: {path}")
    report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (report["frequency_hz"] != 5.0 or report["duration_s"] != 2.0
            or report["boundary"] != "zero Dirichlet on all six faces"):
        raise ValueError(f"FD settings mismatch: {path}")


def prepare(protocol: dict) -> tuple[Path, list[Path]]:
    DATA.mkdir(parents=True, exist_ok=True)
    entries = [protocol["training_addition"], *protocol["new_development_sources"]]
    if len({tuple(item["source_m_s"]) for item in entries}) != len(entries):
        raise ValueError("Duplicate registered source")
    for item in entries:
        path = DATA / item["file"]
        if not path.exists():
            run("examples/generate_fd_observation.py", "--duration", "2",
                "--source", *item["source_m_s"], "--output", path)
        verify(path, item["source_m_s"])
    return DATA / entries[0]["file"], [DATA / item["file"] for item in entries[1:]]


def train(paths: list[Path], protocol: dict) -> Path:
    receiver_dir = ROOT / "outputs/depth_gap_25_receiver_5000"
    hybrid_dir = ROOT / "outputs/depth_gap_25_hybrid_1000"
    receiver = receiver_dir / "conditioned_5Hz.pth"
    final = hybrid_dir / "conditioned_5Hz.pth"
    common = ("--frequencies", "5", "--seed", "42", "--travel-time-carrier",
              "--reference-velocity", "3500", "--distance-core-m", "20",
              "--wavefield-scale", "1e-5", "--receiver-observations", *paths)
    if not receiver.exists():
        if receiver_dir.exists():
            raise RuntimeError(f"Partial run requires inspection: {receiver_dir}")
        config = protocol["receiver_pretraining"]
        run("train_source_conditioned_pinn.py", *common, "--output", receiver_dir,
            "--receiver-only", "--epochs", config["epochs"],
            "--save-every", config["epochs"] + 1,
            "--learning-rate", config["learning_rate"],
            "--receiver-loss-weight", config["receiver_loss_weight"],
            "--receiver-batch-size", config["batch_size"],
            "--receiver-events-per-batch", len(paths))
    if not final.exists():
        if hybrid_dir.exists():
            raise RuntimeError(f"Partial run requires inspection: {hybrid_dir}")
        config = protocol["hybrid_training"]
        run("train_source_conditioned_pinn.py", *common, "--output", hybrid_dir,
            "--initial-checkpoint", receiver, "--epochs", config["epochs"],
            "--save-every", config["epochs"] + 1,
            "--learning-rate", config["learning_rate"],
            "--receiver-loss-weight", config["receiver_loss_weight"],
            "--receiver-batch-size", config["batch_size"],
            "--receiver-events-per-batch", len(paths),
            "--n-phys", config["n_phys"], "--n-ic", config["n_ic"],
            "--n-bc-per-face", config["n_bc_per_face"])
    summary = json.loads((hybrid_dir / "training_summary.json").read_text(encoding="utf-8"))
    if [Path(row["path"]).resolve() for row in summary["run"]["receiver_observations"]] != [
        path.resolve() for path in paths
    ]:
        raise ValueError("Trained checkpoint manifest does not match the registered source list")
    return final


def evaluate(group: str, checkpoint: Path, observations: list[Path],
             reference: Path, training_anchors: list[Path]) -> None:
    prefix = EXPERIMENTS / f"depth_gap_{group}"
    run("examples/audit_forward_error_decomposition.py", checkpoint, *observations,
        "--evaluation-label", "development", "--output", f"{prefix}_development.json")
    run("examples/audit_forward_error_decomposition.py", checkpoint, *training_anchors,
        "--evaluation-label", "training-diagnostic", "--output", f"{prefix}_training.json")
    run("examples/audit_spatial_source_response.py", checkpoint,
        reference, *observations, "--output", f"{prefix}_response.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False)
    cfg = metadata["config"]
    scale = np.asarray([cfg["x_length_m"], cfg["y_length_m"],
                        cfg["z_length_m"], cfg["time_length_s"]], dtype=np.float32)
    scans = []
    for path in observations:
        with np.load(path) as archive:
            fd = np.asarray(archive["data"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float32)
        xyz_t = torch.as_tensor(receivers / scale[:3], device=device)
        time_t = torch.as_tensor(times[::4] / scale[3], device=device)
        fd_t = torch.as_tensor(fd[:, ::4], device=device)
        rms = float(np.sqrt(np.mean(fd.astype(np.float64) ** 2)))
        scans.append({"observation": str(path),
                      "scan": depth_scan(model, xyz_t, time_t, fd_t, truth, scale, rms)})
    path = Path(f"{prefix}_depth_scans.json")
    path.write_text(json.dumps({"purpose": "fixed-true-xyz-time depth scan, development only",
                                "checkpoint": str(checkpoint),
                                "events": scans}, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    control = ROOT / protocol["fixed_control_checkpoint"]
    if hashlib.sha256(control.read_bytes()).hexdigest() != protocol["fixed_control_sha256"]:
        raise ValueError("Fixed control checkpoint changed since registering the experiment")
    _, payload = load_conditioned_checkpoint(control, torch.device("cpu"))
    base = [Path(item["path"]) for item in payload["training_record"]["receiver_observations"]]
    if len(base) != 24 or len(set(base)) != 24:
        raise ValueError("Control must have exactly 24 unique training sources")
    old_dev = BASE / "dev_train_09_z940.npz"
    new_sources = [protocol["training_addition"], *protocol["new_development_sources"]]
    existing = [tuple(np.round(row["source_truth"], 4))
                for row in payload["training_record"]["receiver_observations"]]
    if (len(set(existing + [tuple(row["source_m_s"]) for row in new_sources])) != 27
            or any(np.allclose(np.asarray(row["source_m_s"]),
                               [6500, 3500, 940, .2], atol=1e-4)
                   for row in new_sources)):
        raise ValueError("Training/development source coordinates must not overlap")
    addition, new_dev = prepare(protocol)
    training = [*base, addition]
    checkpoint = train(training, protocol)
    observations = [new_dev[0], old_dev, new_dev[1]]
    reference = BASE / "train_09_z860.npz"
    anchor_dir = ROOT / "data/observations/window_0to2s"
    anchors = [anchor_dir / "train_01_0to2s.npz",
               anchor_dir / "train_09_0to2s.npz"]
    evaluate("control24", control, observations, reference, anchors)
    evaluate("aug25", checkpoint, observations, reference, [*anchors, addition])


if __name__ == "__main__":
    main()
