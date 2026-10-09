"""Reproducible paired source-domain pretraining and target localization pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.invert_fmm_receiver_fields import interpolate_fields, receiver_fields
from examples.run_traveltime_aligned_moe_development import (
    aligned_prediction, load_observations, relative_l2, sample_training_batch,
)
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, choose_diverse_starts, coarse_grid, load_event, pick_arrivals,
    refine, sampled_waveforms, waveform_scores,
)
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator, TravelTimeAlignedPrototypeMoE

PROTOCOL = ROOT / "experiments/source_transfer_moe_protocol.json"
OUTPUT = ROOT / "outputs/source_transfer_moe"
RESULT = ROOT / "experiments/source_transfer_moe_results.json"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_setup():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model_spec = json.loads((ROOT / spec["model_and_optimizer"]).read_text(encoding="utf-8"))
    loc_spec = json.loads((ROOT / spec["localization"]).read_text(encoding="utf-8"))
    if digest(ROOT / spec["source_velocity"]) != spec["source_velocity_sha256"]:
        raise ValueError("Source velocity changed")
    if digest(ROOT / model_spec["velocity_file"]) != model_spec["velocity_sha256"]:
        raise ValueError("Target velocity changed")
    if digest(ROOT / spec["target_fmm_fields"]) != loc_spec["receiver_fmm_fields_sha256"]:
        raise ValueError("Target FMM fields changed")
    if digest(ROOT / spec["target_manifest_checkpoint"]) != model_spec["training_manifest_checkpoint_sha256"]:
        raise ValueError("Target training manifest changed")
    manifest = torch.load(ROOT / spec["target_manifest_checkpoint"], map_location="cpu", weights_only=False)
    rows = manifest["training_record"]["receiver_observations"]
    if len(rows) != model_spec["training_observation_count"]:
        raise ValueError("Target event count changed")
    paths = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError(f"Target training record changed: {path}")
        paths.append(path)
    return spec, model_spec, loc_spec, paths


def check_new_positions(spec: dict, exclude: Path) -> None:
    proposed = [np.asarray(row["source_m_s"], dtype=np.float64) for row in spec["test_events"]]
    if len({tuple(v) for v in proposed}) != len(proposed):
        raise ValueError("Duplicate test position")
    for path in (ROOT / "data/observations").rglob("*.npz"):
        if exclude in path.resolve().parents:
            continue
        with np.load(path) as archive:
            if "source_truth" in archive.files:
                source = np.asarray(archive["source_truth"], dtype=np.float64)
                if any(np.allclose(source, v, rtol=0, atol=1e-3) for v in proposed):
                    raise ValueError(f"Test source already present: {path}")


def generate(path: Path, source: np.ndarray, config: Path | None) -> None:
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite FD record: {path}")
    command = [sys.executable, str(ROOT / "examples/generate_fd_observation.py")]
    if config is not None:
        command += ["--config", str(config)]
    command += ["--output", str(path), "--duration", "2", "--source", *map(str, source)]
    subprocess.run(command, cwd=ROOT, check=True)


def prepare(spec: dict, target_paths: list[Path]) -> None:
    source_dir = ROOT / spec["source_observation_directory"]
    test_dir = ROOT / spec["test_observation_directory"]
    source_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    # Reuse positions but not the target waveforms or velocity when building source records.
    for index in spec["source_training_indices"]:
        with np.load(target_paths[index]) as archive:
            source = np.asarray(archive["source_truth"], dtype=np.float64)
        path = source_dir / f"source_{index:02d}.npz"
        if not path.is_file():
            generate(path, source, ROOT / spec["source_fd_config"])
        with np.load(path) as archive:
            if not np.allclose(archive["source_truth"], source, rtol=0, atol=1e-5):
                raise ValueError(f"Mismatched source record: {path}")
    existing = list(test_dir.glob("*.npz"))
    if not existing:
        check_new_positions(spec, test_dir.resolve())
    for row in spec["test_events"]:
        path = test_dir / row["file"]
        if not path.is_file():
            generate(path, np.asarray(row["source_m_s"]), None)
        with np.load(path) as archive:
            if not np.allclose(archive["source_truth"], row["source_m_s"], rtol=0, atol=1e-5):
                raise ValueError(f"Mismatched test record: {path}")
    field_path = ROOT / spec["source_fmm_fields"]
    if not field_path.exists():
        receivers = np.load(ROOT / "data/observations/receiver_coords.npy")
        velocity = np.load(ROOT / spec["source_velocity"], mmap_mode="r")
        fields = receiver_fields(np.ascontiguousarray(velocity[::2, ::2, ::2]), receivers, 80.0)
        field_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(field_path, fields)
    print(f"Source fields SHA256: {digest(field_path)}", flush=True)


def new_model(fields: np.ndarray, receivers: np.ndarray, setup: dict, device: torch.device):
    m = setup["model"]
    model = TravelTimeAlignedPrototypeMoE(
        ReceiverTravelTimeInterpolator(torch.from_numpy(np.array(fields, copy=True)), 80.0),
        torch.from_numpy(receivers), tuple(m["spatial_lengths_m"]),
        time_length_s=m["time_length_s"], frequency_hz=m["frequency_hz"],
        expert_count=m["expert_count"], hidden_dim=m["hidden_dim"],
        hidden_layers=m["hidden_layers"],
        prototype_centers=torch.tensor(m["prototype_centers"]),
        prototype_scales=tuple(m["prototype_scales"]),
        temperature=m["prototype_temperature"])
    return model.to(device)


def phase(model, events, receivers, times, fields, setup, rng, updates, device):
    training = setup["training"]
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    arrivals = interpolate_fields(fields, np.stack([e["source"][:3] for e in events]), 80.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    started = time.perf_counter()
    for step in range(1, updates + 1):
        model.train()
        points, sources, receiver_ids, target = sample_training_batch(
            events, receivers, times, arrivals, setup, rng, domain, device)
        optimizer.zero_grad(set_to_none=True)
        prediction, gates, _ = model(points, sources, receiver_ids)
        balance = (gates.mean(0) - 1.0 / model.expert_count).square().mean()
        loss = ((prediction - target).square().mean()
                + training["gate_balance_weight"] * balance
                + training["prototype_separation_weight"] * model.prototype_separation_loss())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
        optimizer.step()
        with torch.no_grad():
            model.prototype_centers.clamp_(*setup["model"]["prototype_bounds_normalized"])
        if step % 500 == 0:
            print(f"  step {step}/{updates}: loss={float(loss.detach()):.5f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


def train(spec: dict, setup: dict, target_paths: list[Path]):
    source_paths = [ROOT / spec["source_observation_directory"] / f"source_{i:02d}.npz"
                    for i in spec["source_training_indices"]]
    if not all(p.is_file() for p in source_paths):
        raise FileNotFoundError("Run prepare first")
    source_events, receivers, times = load_observations(source_paths)
    target_events, target_receivers, target_times = load_observations(target_paths)
    if not np.array_equal(receivers, target_receivers) or not np.array_equal(times, target_times):
        raise ValueError("Source and target acquisition differ")
    source_fields = np.load(ROOT / spec["source_fmm_fields"], mmap_mode="r")
    target_fields = np.load(ROOT / spec["target_fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            destination = OUTPUT / f"{arm}_seed{seed}.pth"
            if destination.exists():
                print(f"Preserving existing checkpoint: {destination}", flush=True)
                continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            first_domain = "source" if arm == "source_to_target" else "target"
            initial_fields = source_fields if first_domain == "source" else target_fields
            model = new_model(initial_fields, receivers, setup, device)
            seconds = {}
            if arm != "target_only_1500":
                first_events = (source_events if first_domain == "source"
                                else [target_events[i] for i in spec["source_training_indices"]])
                seconds["first_phase"] = phase(
                    model, first_events, receivers, times, initial_fields, setup, rng,
                    spec["phase_updates"], device)
                # FMM fields are geology-specific and are intentionally absent from state_dict.
                if first_domain == "source":
                    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    model = new_model(target_fields, receivers, setup, device)
                    model.load_state_dict(weights)
            seconds["target_phase"] = phase(
                model, target_events, receivers, times, target_fields, setup, rng,
                spec["phase_updates"], device)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "arm": arm, "seed": seed, "seconds": seconds,
                        "protocol_sha256": digest(PROTOCOL),
                        "source_velocity_sha256": spec["source_velocity_sha256"],
                        "source_fields_sha256": digest(ROOT / spec["source_fmm_fields"]),
                        "target_fields_sha256": digest(ROOT / spec["target_fmm_fields"]),
                        "source_observations": {p.name: digest(p) for p in source_paths},
                        "target_observations": {p.name: digest(p) for p in target_paths}}, destination)
            print(f"Saved {destination}, seconds={seconds}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def evaluate(spec: dict, setup: dict, loc: dict):
    if RESULT.exists():
        raise FileExistsError(f"Refusing to overwrite results: {RESULT}")
    paths = [ROOT / spec["test_observation_directory"] / row["file"]
             for row in spec["test_events"]]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Run prepare first")
    receivers = np.load(ROOT / "data/observations/receiver_coords.npy").astype(np.float32)
    events, receiver_check, _ = load_observations(paths)
    if not np.array_equal(receivers, receiver_check):
        raise ValueError("Test receiver mismatch")
    fields = np.load(ROOT / spec["target_fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    bounds = np.asarray(loc["source_bounds_m_s"], dtype=np.float32)
    search, arrival = loc["search"], loc["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            checkpoint = OUTPUT / f"{arm}_seed{seed}.pth"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["protocol_sha256"] != digest(PROTOCOL):
                raise ValueError(f"Protocol changed after training: {checkpoint}")
            model = new_model(fields, receivers, setup, device)
            model.load_state_dict(payload["state_dict"])
            model.eval().requires_grad_(False)
            for event, row in zip(events, spec["test_events"]):
                observed = load_event(event["path"], receivers)
                _, picks = pick_arrivals(observed, arrival)
                valid = int(np.isfinite(picks).sum())
                if valid < arrival["minimum_valid_picks"]:
                    raise ValueError(f"Insufficient picks for {event['path']}: {valid}")
                coarse = sampled_waveforms(observed, domain,
                                           search["coarse_receiver_stride"],
                                           search["coarse_time_stride"], device)
                fine = sampled_waveforms(observed, domain,
                                         search["refine_receiver_stride"],
                                         search["refine_time_stride"], device)
                start = time.perf_counter()
                wave = waveform_scores(model, candidates, domain, coarse,
                                       search["candidate_batch"])
                fmm = arrival_scores(model, candidates, picks,
                                     arrival["arrival_scale_s"], search["candidate_batch"])
                predictions = {}
                for name, alpha in (("waveform_only", 0.0),
                                    ("hybrid", arrival["hybrid_weight"])):
                    score = wave + alpha * fmm
                    ids = choose_diverse_starts(candidates, score, search["restarts"],
                                                search["minimum_start_separation_m"])
                    starts = [{"source_m_s": candidates[i].tolist(),
                               "coarse_total_loss": float(score[i])} for i in ids]
                    best, _ = refine(model, starts, domain, bounds, fine, picks,
                                     search, arrival, alpha)
                    estimate = np.asarray(best["estimated_source_m_s"])
                    truth = np.asarray(row["source_m_s"])
                    predictions[name] = {"estimated_source_m_s": estimate.tolist(),
                                         "location_error_m": float(np.linalg.norm((estimate-truth)[:3])),
                                         "origin_time_error_s": float(abs(estimate[3]-truth[3])),
                                         "waveform_normalized_mse": best["waveform_normalized_mse"]}
                if device.type == "cuda":
                    torch.cuda.synchronize()
                full = aligned_prediction(model, event, receivers, observed["times"], domain, device)
                l2 = relative_l2(event["data"] / event["rms"], full)
                record = {"seed": seed, "arm": arm, "event": row["file"],
                          "truth_m_s": row["source_m_s"], "sha256": event["sha256"],
                          "valid_picks": valid, "full_waveform_relative_l2": l2,
                          "search_seconds": time.perf_counter() - start,
                          "predictions": predictions}
                rows.append(record)
                print(f"{seed} {arm} {row['file']}: hybrid={predictions['hybrid']['location_error_m']:.1f} m, L2={l2:.3f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = {}
    for arm in spec["arms"]:
        selected = [r for r in rows if r["arm"] == arm]
        summary[arm] = {"median_hybrid_location_error_m": float(np.median([
            r["predictions"]["hybrid"]["location_error_m"] for r in selected])),
                        "mean_full_waveform_relative_l2": float(np.mean([
            r["full_waveform_relative_l2"] for r in selected]))}
    result = {"purpose": spec["purpose"], "protocol_sha256": digest(PROTOCOL),
              "device": str(device), "rows": rows, "summary": summary,
              "interpretation": spec["interpretation"]}
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "evaluate"))
    args = parser.parse_args()
    spec, setup, loc, paths = read_setup()
    if args.stage == "prepare":
        prepare(spec, paths)
    elif args.stage == "train":
        train(spec, setup, paths)
    else:
        evaluate(spec, setup, loc)


if __name__ == "__main__":
    main()
