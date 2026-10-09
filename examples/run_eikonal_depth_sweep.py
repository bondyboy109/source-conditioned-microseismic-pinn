"""Audit frozen waveforms and picked arrivals on a preregistered depth sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_eikonal_training_generalization import evaluate
from examples.audit_fmm_traveltime import first_arrival
from examples.run_eikonal_directional_grid import checked_record, predict, response
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_depth_sweep_protocol.json"
DATA = ROOT / "data/observations/eikonal_depth_sweep"
OUTPUT = ROOT / "experiments/eikonal_depth_sweep_results.json"


def events_from_protocol(protocol: dict) -> list[dict]:
    events = []
    for anchor in protocol["anchors"]:
        for magnitude in protocol["new_depth_offset_m"]:
            for sign in protocol["signs"]:
                source = list(anchor["source_m_s"])
                source[2] += magnitude * sign
                events.append({"anchor": anchor["name"], "offset_m": magnitude * sign,
                               "file": f"{anchor['name']}_z{'p' if sign > 0 else 'm'}{magnitude}.npz",
                               "source_m_s": source})
    return events


def preflight(protocol: dict) -> tuple[list[dict], list[dict]]:
    if (len(protocol["anchors"]) != 2 or protocol["new_depth_offset_m"] != [80, 160, 280]
            or protocol["signs"] != [-1, 1]):
        raise ValueError("Unexpected registered sweep design")
    for name in ("eikonal", "straight"):
        if hash_file(ROOT / protocol[f"{name}_checkpoint"]) != protocol[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} checkpoint changed")
    if hash_file(ROOT / protocol["velocity_file"]) != protocol["velocity_sha256"]:
        raise ValueError("Frozen velocity changed")
    prior = ROOT / protocol["separate_previously_inspected_reference"]
    old = json.loads(prior.read_text(encoding="utf-8"))
    if (old["checkpoint_sha256"] != {k: protocol[f"{k}_sha256"]
                                     for k in ("eikonal", "straight")}
            or len([r for r in old["events"] if r["axis"] == "z"]) != 4):
        raise ValueError("Previous 240 m reference changed")
    manifests = []
    for name in ("eikonal", "straight"):
        _, meta = load_conditioned_checkpoint(ROOT / protocol[f"{name}_checkpoint"],
                                              torch.device("cpu"))
        if meta["frequency_hz"] != 5:
            raise ValueError("Expected 5 Hz checkpoints")
        manifests.append(meta["training_record"]["receiver_observations"])
    training = manifests[0]
    if (len(training) != 29 or [(r["sha256"], r["source_truth"]) for r in training]
            != [(r["sha256"], r["source_truth"]) for r in manifests[1]]):
        raise ValueError("Models do not share the same 29 training events")
    for anchor in protocol["anchors"]:
        path = ROOT / anchor["observation"]
        matches = [r for r in training if Path(r["path"]).resolve() == path.resolve()]
        if len(matches) != 1 or hash_file(path) != matches[0]["sha256"]:
            raise ValueError(f"Anchor not in training set: {path}")
        with np.load(path) as archive:
            if not np.allclose(archive["source_truth"], anchor["source_m_s"], rtol=0, atol=1e-5):
                raise ValueError(f"Anchor source differs: {path}")
    events = events_from_protocol(protocol)
    if len(events) != 12 or len({tuple(e["source_m_s"][:3]) for e in events}) != 12:
        raise ValueError("Depth events are not twelve distinct locations")
    archive_sources = []
    for path in sorted((ROOT / "data/observations").rglob("*.npz")):
        if DATA in path.parents:
            continue
        with np.load(path) as archive:
            if "source_truth" in archive.files:
                archive_sources.append((path, np.asarray(archive["source_truth"][:3], dtype=np.float64)))
    for event in events:
        source = np.asarray(event["source_m_s"][:3])
        if not 100 <= source[2] <= 1500:
            raise ValueError(f"Source exceeds trained depth bounds: {event['file']}")
        for path, previous in archive_sources:
            if np.allclose(source, previous, rtol=0, atol=.01):
                raise ValueError(f"New source already archived: {event['file']} at {path}")
    print(f"Pre-generation: {len(events)} new depth sources distinct from "
          f"{len(archive_sources)} archived records", flush=True)
    return events, training


def arrival_metrics(fd: np.ndarray, candidate: np.ndarray,
                    times: np.ndarray) -> dict:
    fd_picks = np.asarray([first_arrival(trace, times, threshold=.05)
                           for trace in fd], dtype=np.float64)
    model_picks = np.asarray([first_arrival(trace, times, threshold=.05)
                              for trace in candidate], dtype=np.float64)
    fd_peaks = np.max(np.abs(fd), axis=1)
    valid_fd = (np.isfinite(fd_picks) & (fd_peaks >= .01 * fd_peaks.max())
                & (fd_picks <= 1.85))
    paired = valid_fd & np.isfinite(model_picks) & (model_picks <= 1.85)
    if valid_fd.sum() < 6:
        return {"status": "insufficient_fd_picks", "valid_fd_pick_count": int(valid_fd.sum()),
                "paired_pick_count": int(paired.sum()),
                "missing_model_picks_on_valid_fd_stations": int((valid_fd & ~paired).sum()),
                "signed_mean_error_s": None, "rmse_s": None,
                "demeaned_rmse_s": None}
    if paired.sum() < 6:
        return {"status": "insufficient_paired_picks", "valid_fd_pick_count": int(valid_fd.sum()),
                "paired_pick_count": int(paired.sum()),
                "missing_model_picks_on_valid_fd_stations": int((valid_fd & ~paired).sum()),
                "signed_mean_error_s": None, "rmse_s": None,
                "demeaned_rmse_s": None}
    difference = model_picks[paired] - fd_picks[paired]
    return {"status": "ok", "valid_fd_pick_count": int(valid_fd.sum()),
            "paired_pick_count": int(paired.sum()),
            "missing_model_picks_on_valid_fd_stations": int((valid_fd & ~paired).sum()),
            "signed_mean_error_s": float(difference.mean()),
            "rmse_s": float(np.sqrt(np.mean(difference ** 2))),
            "demeaned_rmse_s": float(np.std(difference))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.preflight:
        raise FileExistsError(f"One-time result already exists: {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    events, training = preflight(protocol)
    if args.preflight:
        return
    DATA.mkdir(parents=True, exist_ok=True)
    anchors = {row["name"]: ROOT / row["observation"] for row in protocol["anchors"]}
    for event in events:
        path = DATA / event["file"]
        if not path.exists():
            subprocess.run([sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                            "--duration", "2", "--output", str(path), "--source",
                            *(str(n) for n in event["source_m_s"])], cwd=ROOT, check=True)
        checked_record(path, event["source_m_s"], anchors[event["anchor"]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name in ("eikonal", "straight"):
        model, payload = load_conditioned_checkpoint(ROOT / protocol[f"{name}_checkpoint"], device)
        model.eval().requires_grad_(False)
        scale = np.asarray([payload["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        models[name] = (model, scale)
    baseline = {}
    for item in protocol["anchors"]:
        path = anchors[item["name"]]
        record = checked_record(path, item["source_m_s"], path)
        baseline[item["name"]] = (record, {name: predict(model, scale, *record)
                                             for name, (model, scale) in models.items()})
    source_positions = np.asarray([r["source_truth"][:3] for r in training], dtype=np.float64)
    results = []
    for event in events:
        path = DATA / event["file"]
        fd, times, receivers, truth = checked_record(path, event["source_m_s"], anchors[event["anchor"]])
        anchor_record, anchor_predictions = baseline[event["anchor"]]
        nearest = float(np.min(np.linalg.norm(source_positions - truth[:3], axis=1)))
        row = {**event, "sha256": hash_file(path), "nearest_training_spatial_distance_m": nearest,
               "fd_runtime_seconds": json.loads(path.with_suffix(".json").read_text())["runtime_seconds"],
               "models": {}}
        for name, (model, scale) in models.items():
            metrics = evaluate(model, scale, path, row["sha256"])
            model_record = predict(model, scale, fd, times, receivers, truth)
            row["models"][name] = {
                "full_record_relative_l2": metrics["full_record_relative_l2"],
                "active_sample_relative_l2": metrics["active_sample_relative_l2"],
                "predicted_to_observed_peak_ratio": metrics["predicted_to_observed_peak_ratio"],
                **response(anchor_record[0], fd, anchor_predictions[name], model_record),
                "arrival": arrival_metrics(fd, model_record, times),
            }
        results.append(row)
        print(f"{event['file']}: eikonal L2={row['models']['eikonal']['full_record_relative_l2']:.3f}; "
              f"straight L2={row['models']['straight']['full_record_relative_l2']:.3f}; "
              f"paired picks={row['models']['eikonal']['arrival']['paired_pick_count']}", flush=True)
    groups = {}
    for magnitude in protocol["new_depth_offset_m"]:
        subset = [row for row in results if abs(row["offset_m"]) == magnitude]
        groups[str(magnitude)] = {name: {
            "mean_full_l2": float(np.mean([r["models"][name]["full_record_relative_l2"] for r in subset])),
            "mean_change_l2": float(np.mean([r["models"][name]["change_relative_l2"] for r in subset])),
            "mean_change_cosine": float(np.mean([r["models"][name]["change_cosine"] for r in subset])),
            "mean_change_norm_ratio": float(np.mean([
                r["models"][name]["model_to_fd_change_norm_ratio"] for r in subset])),
            "arrival_usable_count": sum(r["models"][name]["arrival"]["status"] == "ok" for r in subset),
            "mean_arrival_rmse_s": (float(np.mean([
                r["models"][name]["arrival"]["rmse_s"] for r in subset
                if r["models"][name]["arrival"]["status"] == "ok"]))
                if any(r["models"][name]["arrival"]["status"] == "ok" for r in subset) else None),
        } for name in models}
    report = {"purpose": protocol["purpose"], "role": protocol["role"],
              "protocol_sha256": hash_file(PROTOCOL),
              "checkpoint_sha256": {name: protocol[f"{name}_sha256"] for name in models},
              "velocity_sha256": protocol["velocity_sha256"],
              "arrival_rule": protocol["arrival_rule"],
              "events": results, "magnitude_means": groups,
              "limitations": protocol["interpretation"]}
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
