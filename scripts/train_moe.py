"""Train a new MoE from the archived training data without legacy comparisons.

This is a current-version rerun. It never overwrites the archived checkpoint,
protocols, results, or observations and is not an independent confirmation.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.run_traveltime_aligned_moe_development import (
    aligned_prediction,
    build_or_load_fields,
    interpolate_fields,
    load_observations,
    sample_training_batch,
)
from traveltime_aligned_moe import (
    ReceiverTravelTimeInterpolator,
    TravelTimeAlignedPrototypeMoE,
)

PROTOCOL = ROOT / "experiments/traveltime_aligned_moe_development_protocol.json"
TRAINING_MANIFEST = ROOT / "data/training_manifest.json"
DEVELOPMENT_MANIFEST = ROOT / "data/development_manifest.json"
EXPECTED_PROTOCOL_SHA256 = "8cd6bf5e1c2fca0774e153ea65cdcb7ceeca7b2d016a4eabf38fa5d7cd2140b5"
EXPECTED_CACHE_SHA256 = "f0023bb953e8d94ef5184293c0393331232f7fe93207525a7242d5dfffbf3a36"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def manifest_paths(path: Path, count: int) -> list[Path]:
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document.get("records", [])
    if len(records) != count:
        raise ValueError(f"Expected {count} observations in {path}, found {len(records)}")
    paths: list[Path] = []
    for row in records:
        relative = Path(row["path"])
        if relative.is_absolute():
            raise ValueError(f"Manifest paths must be repository-relative: {relative}")
        observation = (ROOT / relative).resolve()
        observation.relative_to((ROOT / "data/observations").resolve())
        if (observation in paths or observation.suffix != ".npz"
                or "confirmation" in str(relative).lower()):
            raise ValueError(f"Duplicate, invalid, or confirmation observation: {relative}")
        if sha256(observation) != row["sha256"]:
            raise ValueError(f"Observation hash changed: {relative}")
        paths.append(observation)
    return paths


def verify_inputs() -> tuple[dict, list[dict], list[dict], np.ndarray, np.ndarray, dict]:
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Archived training protocol hash changed")
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    training_paths = manifest_paths(TRAINING_MANIFEST, int(spec["training_observation_count"]))
    development_paths = manifest_paths(DEVELOPMENT_MANIFEST, int(spec["development_event_count"]))
    if set(training_paths) & set(development_paths):
        raise ValueError("Training and development observations overlap")
    expected_development = []
    for directory in spec["development_sets"]:
        expected_development.extend(sorted((ROOT / directory).glob("*.npz")))
    if {path.resolve() for path in expected_development} != set(development_paths):
        raise ValueError("Development manifest differs from the registered source sets")
    training_events, receivers, times = load_observations(training_paths)
    development_events, dev_receivers, dev_times = load_observations(development_paths)
    if (not np.array_equal(receivers, dev_receivers)
            or not np.array_equal(times, dev_times)):
        raise ValueError("Training and development acquisition differ")
    if (receivers.shape != (25, 3) or times.shape != (1001,)
            or not np.isfinite(receivers).all()
            or not np.allclose(times, np.arange(1001) * 0.002, atol=2e-7, rtol=0)):
        raise ValueError("Expected 25 receivers and 1001 samples from 0 to 2 s")

    cache = ROOT / spec["fmm"]["cache_file"]
    if cache.is_file():
        if sha256(cache) != EXPECTED_CACHE_SHA256:
            raise ValueError("Archived reciprocal travel-time cache hash changed")
        fields = np.load(cache, mmap_mode="r")
        if fields.shape != (25, 201, 201, 47) or fields.dtype != np.float32:
            raise ValueError("Unexpected reciprocal travel-time cache geometry")
        cache_status = "verified_existing_cache; external velocity not required"
    else:
        velocity = ROOT / spec["velocity_file"]
        if not velocity.is_file():
            raise FileNotFoundError(
                "Install the receiver travel-time cache Release asset, or provide "
                f"the original velocity export at {velocity} to regenerate it")
        if sha256(velocity) != spec["velocity_sha256"]:
            raise ValueError("External velocity array does not match the archived export")
        cache_status = "cache_missing; verified external velocity will regenerate it during training"

    report = {
        "status": "input_checks_passed",
        "training_observations": len(training_events),
        "development_observations": len(development_events),
        "receiver_count": len(receivers),
        "sample_count": len(times),
        "time_step_s": 0.002,
        "time_window_s": [0.0, 2.0],
        "source_protocol_sha256": sha256(PROTOCOL),
        "training_manifest_sha256": sha256(TRAINING_MANIFEST),
        "development_manifest_sha256": sha256(DEVELOPMENT_MANIFEST),
        "cache_status": cache_status,
        "expected_cache_sha256": EXPECTED_CACHE_SHA256,
        "independent_confirmation": False,
    }
    return spec, training_events, development_events, receivers, times, report


def make_model(spec: dict, fields: np.ndarray, receivers: np.ndarray,
               device: torch.device) -> TravelTimeAlignedPrototypeMoE:
    settings = spec["model"]
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)),
        float(spec["fmm"]["grid_spacing_m"])).to(device)
    return TravelTimeAlignedPrototypeMoE(
        interpolator, torch.from_numpy(receivers),
        tuple(settings["spatial_lengths_m"]),
        time_length_s=float(settings["time_length_s"]),
        frequency_hz=float(settings["frequency_hz"]),
        expert_count=int(settings["expert_count"]),
        hidden_dim=int(settings["hidden_dim"]),
        hidden_layers=int(settings["hidden_layers"]),
        prototype_centers=torch.tensor(settings["prototype_centers"]),
        prototype_scales=tuple(settings["prototype_scales"]),
        temperature=float(settings["prototype_temperature"])).to(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true",
                        help="Verify manifests, observations, acquisition, protocol and cache; do not train")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--output", type=Path,
                        help="New directory under outputs/new_training/; existing directories are refused")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    spec, training_events, development_events, receivers, times, checks = verify_inputs()
    print(json.dumps(checks, indent=2), flush=True)
    if args.check_inputs:
        return

    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(f"outputs/new_training/run_{timestamp}_seed{args.seed}_u{args.updates}")
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    output.relative_to((ROOT / "outputs/new_training").resolve())
    output.mkdir(parents=True, exist_ok=False)

    run_spec = copy.deepcopy(spec)
    for key in ("training_manifest_checkpoint", "training_manifest_checkpoint_sha256",
                "development_gates", "decision_rule"):
        run_spec.pop(key, None)
    run_spec["purpose"] = "current-version MoE training rerun on archived training and development observations"
    run_spec["training_manifest_file"] = "data/training_manifest.json"
    run_spec["development_manifest_file"] = "data/development_manifest.json"
    run_spec["training"]["updates"] = args.updates
    run_spec["training"]["seed"] = args.seed
    run_spec["output_checkpoint"] = (output / "traveltime_aligned_moe.pth").relative_to(ROOT).as_posix()
    run_spec["result_file"] = (output / "training_metrics.json").relative_to(ROOT).as_posix()
    run_spec["report_file"] = (output / "run_record.json").relative_to(ROOT).as_posix()
    run_spec["source_protocol_sha256"] = checks["source_protocol_sha256"]
    run_spec["independent_confirmation"] = False
    run_spec["limitations"] = (
        "New training rerun on the existing training and development sets. "
        "It does not replace the archived checkpoint, re-establish historical "
        "confirmation, or demonstrate an improvement over any baseline.")
    run_protocol = output / "training_protocol.json"
    write_json(run_protocol, run_spec)
    write_json(output / "input_checks.json", checks)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(args.seed)
    fields, cache_build_seconds = build_or_load_fields(run_spec, receivers)
    field_cache = ROOT / run_spec["fmm"]["cache_file"]
    cache_hash = sha256(field_cache)
    if cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError("Regenerated cache differs from the archived fields; preserve and investigate it before training")
    domain = np.asarray([*run_spec["model"]["spatial_lengths_m"],
                         run_spec["model"]["time_length_s"]], dtype=np.float32)
    source_xyz = np.stack([event["source"][:3] for event in training_events])
    arrivals = interpolate_fields(fields, source_xyz, float(run_spec["fmm"]["grid_spacing_m"]))
    model = make_model(run_spec, fields, receivers, device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(run_spec["training"]["learning_rate"]))
    history = []
    synchronize(device)
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        model.train()
        points, sources, receiver_index, target = sample_training_batch(
            training_events, receivers, times, arrivals, run_spec, rng, domain, device)
        optimizer.zero_grad(set_to_none=True)
        prediction, gate, _ = model(points, sources, receiver_index)
        waveform_loss = (prediction - target).square().mean()
        gate_mean = gate.mean(dim=0)
        balance_loss = ((gate_mean - 1.0 / model.expert_count) ** 2).mean()
        separation_loss = model.prototype_separation_loss()
        loss = (waveform_loss
                + float(run_spec["training"]["gate_balance_weight"]) * balance_loss
                + float(run_spec["training"]["prototype_separation_weight"]) * separation_loss)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite training loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                      float(run_spec["training"]["gradient_clip_norm"]))
        optimizer.step()
        low, high = run_spec["model"]["prototype_bounds_normalized"]
        with torch.no_grad():
            model.prototype_centers.clamp_(float(low), float(high))
        if update == 1 or update % 100 == 0 or update == args.updates:
            row = {"update": update, "loss": float(loss.detach()),
                   "waveform": float(waveform_loss.detach()),
                   "gate_balance": float(balance_loss.detach()),
                   "prototype_separation": float(separation_loss.detach()),
                   "mean_gate": gate_mean.detach().cpu().tolist()}
            history.append(row)
            print(json.dumps(row), flush=True)
    synchronize(device)
    training_seconds = time.perf_counter() - started
    checkpoint = output / "traveltime_aligned_moe.pth"
    torch.save({"state_dict": model.state_dict(), "protocol": run_spec,
                "protocol_sha256": sha256(run_protocol),
                "field_cache_sha256": cache_hash,
                "training_history": history,
                "training_seconds": training_seconds}, checkpoint)

    development_metrics = []
    started = time.perf_counter()
    for event in development_events:
        predicted = aligned_prediction(model, event, receivers, times, domain, device)
        observed = event["data"] / event["rms"]
        denominator = np.linalg.norm(observed.astype(np.float64))
        relative_l2 = float(np.linalg.norm(predicted.astype(np.float64) - observed) / denominator)
        development_metrics.append({
            "observation": event["path"].relative_to(ROOT).as_posix(),
            "sha256": event["sha256"], "relative_l2": relative_l2})
    synchronize(device)
    evaluation_seconds = time.perf_counter() - started
    write_json(output / "training_metrics.json", {
        "independent_confirmation": False,
        "training_history": history,
        "development_receiver_waveform_metrics": development_metrics,
        "development_mean_relative_l2": float(np.mean([row["relative_l2"] for row in development_metrics])),
        "development_median_relative_l2": float(np.median([row["relative_l2"] for row in development_metrics])),
        "interpretation": "Development diagnostics of this new checkpoint; not localization or independent confirmation results."})
    write_json(output / "run_record.json", {
        "status": "completed_new_training",
        "independent_confirmation": False,
        "seed": args.seed, "updates": args.updates,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "training_protocol_sha256": sha256(run_protocol),
        "source_protocol_sha256": checks["source_protocol_sha256"],
        "training_entrypoint_sha256": sha256(Path(__file__)),
        "field_cache_sha256": cache_hash,
        "training_seconds": training_seconds,
        "cache_build_seconds": cache_build_seconds,
        "development_evaluation_seconds": evaluation_seconds,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pytorch": torch.__version__, "cuda": torch.version.cuda},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "limitations": run_spec["limitations"],
    })
    print(f"New training saved to {output}. Archived experiment files were not overwritten.", flush=True)


if __name__ == "__main__":
    main()
