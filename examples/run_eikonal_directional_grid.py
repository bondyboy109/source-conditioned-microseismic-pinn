"""Run a registered directional FD development grid on two frozen operators."""

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
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_directional_grid_protocol.json"
DATA = ROOT / "data/observations/eikonal_directional_grid"
OUTPUT = ROOT / "experiments/eikonal_directional_grid_results.json"


def planned_events(protocol: dict) -> list[dict]:
    offset = protocol["offset_m"]
    events = []
    for anchor in protocol["anchors"]:
        for axis in protocol["axes"]:
            for sign in protocol["signs"]:
                source = np.asarray(anchor["source_m_s"], dtype=np.float64).copy()
                source["xyz".index(axis)] += sign * offset
                name = f"{anchor['name']}_{axis}{'p' if sign > 0 else 'm'}{offset}.npz"
                events.append({"anchor": anchor["name"], "axis": axis,
                               "sign": sign, "file": name, "source_m_s": source.tolist()})
    return events


def preflight(protocol: dict) -> tuple[list[dict], list[dict]]:
    for name in ("eikonal", "straight"):
        path = ROOT / protocol[f"{name}_checkpoint"]
        if hash_file(path) != protocol[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} checkpoint changed")
    if hash_file(ROOT / protocol["velocity_file"]) != protocol["velocity_sha256"]:
        raise ValueError("Velocity file changed")
    events = planned_events(protocol)
    if (len(protocol["anchors"]) != 2 or len(events) != 12
            or len({event["file"] for event in events}) != 12
            or len({tuple(event["source_m_s"][:3]) for event in events}) != 12):
        raise ValueError("Expected twelve distinct registered offsets")
    snapshots = []
    for name in ("eikonal", "straight"):
        _, meta = load_conditioned_checkpoint(ROOT / protocol[f"{name}_checkpoint"],
                                              torch.device("cpu"))
        training = meta["training_record"]["receiver_observations"]
        if len(training) != 29 or meta["frequency_hz"] != 5:
            raise ValueError("Frozen checkpoint must use the same 29-source 5 Hz design")
        snapshots.append(training)
    if [(r["sha256"], r["source_truth"]) for r in snapshots[0]] != [
            (r["sha256"], r["source_truth"]) for r in snapshots[1]]:
        raise ValueError("Models do not share a training manifest")
    training = snapshots[0]
    bounds = np.array([[2000., 14000.], [2000., 14000.], [100., 1500.]])
    for anchor in protocol["anchors"]:
        path = ROOT / anchor["observation"]
        matching = [record for record in training if Path(record["path"]).resolve() == path.resolve()]
        if len(matching) != 1 or hash_file(path) != matching[0]["sha256"]:
            raise ValueError(f"Anchor is not the registered training observation: {path}")
        with np.load(path) as archive:
            if not np.allclose(archive["source_truth"], anchor["source_m_s"], atol=1e-5, rtol=0):
                raise ValueError(f"Anchor truth differs from protocol: {path}")
    known_sources = []
    for path in sorted((ROOT / "data/observations").rglob("*.npz")):
        if DATA in path.parents:
            continue
        with np.load(path) as archive:
            if "source_truth" in archive.files:
                known_sources.append((path, np.asarray(archive["source_truth"][:3], dtype=np.float64)))
    for event in events:
        xyz = np.asarray(event["source_m_s"][:3], dtype=np.float64)
        if np.any(xyz < bounds[:, 0]) or np.any(xyz > bounds[:, 1]):
            raise ValueError(f"Source outside trained bounds: {event['file']}")
        for path, known in known_sources:
            if np.allclose(xyz, known, atol=0.01, rtol=0):
                raise ValueError(f"Source position previously observed: {event['file']} in {path}")
    print(f"Pre-generation audit: {len(events)} unique offset positions; "
          f"no overlap with {len(known_sources)} archived records", flush=True)
    return events, training


def checked_record(path: Path, source: list[float], reference: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        truth = np.asarray(archive["source_truth"], dtype=np.float32)
    with np.load(reference) as archive:
        if not np.array_equal(archive["times"], times) or not np.array_equal(archive["receiver_coords"], receivers):
            raise ValueError(f"Receiver geometry or times differ from anchor: {path}")
    provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (data.shape != (25, 1001) or not np.isfinite(data).all()
            or not np.allclose(truth, source, atol=1e-5, rtol=0)
            or provenance["method"] != "second_order_3d_acoustic_FD"
            or provenance["frequency_hz"] != 5 or provenance["duration_s"] != 2
            or provenance["boundary"] != "zero Dirichlet on all six faces"):
        raise ValueError(f"FD data or provenance differs from preregistration: {path}")
    return data, times, receivers, truth


def predict(model, scale: np.ndarray, data: np.ndarray, times: np.ndarray,
            receivers: np.ndarray, truth: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    source = torch.as_tensor(truth / scale, device=device)
    result = np.empty_like(data)
    with torch.inference_mode():
        for station, receiver in enumerate(receivers):
            xyzt = np.column_stack((np.broadcast_to(receiver, (len(times), 3)), times))
            result[station] = model(torch.as_tensor(xyzt / scale, device=device),
                                    source.expand(len(times), 4)).squeeze(-1).cpu().numpy()
    return result


def response(original: np.ndarray, new: np.ndarray,
             original_pred: np.ndarray, new_pred: np.ndarray) -> dict:
    fd_change = new.astype(np.float64) - original.astype(np.float64)
    model_change = new_pred.astype(np.float64) - original_pred.astype(np.float64)
    fd_norm = np.linalg.norm(fd_change)
    predicted_norm = np.linalg.norm(model_change)
    if fd_norm <= 0 or predicted_norm <= 0:
        raise ValueError("Cannot measure an empty finite source displacement")
    return {"fd_change_over_anchor_norm": float(fd_norm / np.linalg.norm(original)),
            "model_to_fd_change_norm_ratio": float(predicted_norm / fd_norm),
            "change_relative_l2": float(np.linalg.norm(model_change - fd_change) / fd_norm),
            "change_cosine": float(np.sum(fd_change * model_change) / (fd_norm * predicted_norm))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="Verify source novelty before FD generation")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.preflight:
        raise FileExistsError(f"Development report already exists: {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    events, training = preflight(protocol)
    if args.preflight:
        return
    DATA.mkdir(parents=True, exist_ok=True)
    anchor_paths = {row["name"]: ROOT / row["observation"] for row in protocol["anchors"]}
    for event in events:
        path = DATA / event["file"]
        if not path.exists():
            subprocess.run([sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                            "--duration", "2", "--output", str(path), "--source",
                            *(str(n) for n in event["source_m_s"])], cwd=ROOT, check=True)
        checked_record(path, event["source_m_s"], anchor_paths[event["anchor"]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name in ("eikonal", "straight"):
        model, meta = load_conditioned_checkpoint(ROOT / protocol[f"{name}_checkpoint"], device)
        model.eval().requires_grad_(False)
        scale = np.asarray([meta["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        models[name] = model, scale
    anchor_values = {}
    anchors = {}
    for entry in protocol["anchors"]:
        path = anchor_paths[entry["name"]]
        values = checked_record(path, entry["source_m_s"], path)
        anchor_values[entry["name"]] = values
        anchors[entry["name"]] = {"observation": str(path), "sha256": hash_file(path),
                                  "metrics": {}, "prediction": {}}
        for name, (model, scale) in models.items():
            anchors[entry["name"]]["metrics"][name] = evaluate(model, scale, path, hash_file(path))
            anchors[entry["name"]]["prediction"][name] = predict(model, scale, *values)
    training_sources = np.asarray([r["source_truth"][:3] for r in training], dtype=np.float64)
    rows = []
    for event in events:
        path = DATA / event["file"]
        values = checked_record(path, event["source_m_s"], anchor_paths[event["anchor"]])
        nearest = float(np.min(np.linalg.norm(training_sources - np.asarray(event["source_m_s"][:3]), axis=1)))
        row = {**event, "sha256": hash_file(path), "nearest_training_spatial_distance_m": nearest,
               "fd_generation_seconds": json.loads(path.with_suffix(".json").read_text())["runtime_seconds"],
               "models": {}}
        for name, (model, scale) in models.items():
            metrics = evaluate(model, scale, path, row["sha256"])
            predicted = predict(model, scale, *values)
            sensitivity = response(anchor_values[event["anchor"]][0], values[0],
                                   anchors[event["anchor"]]["prediction"][name], predicted)
            row["models"][name] = {"full_record_relative_l2": metrics["full_record_relative_l2"],
                                   "active_sample_relative_l2": metrics["active_sample_relative_l2"],
                                   "predicted_to_observed_peak_ratio": metrics["predicted_to_observed_peak_ratio"],
                                   **sensitivity}
        rows.append(row)
        print(f"{event['file']}: nearest={nearest:.0f} m; "
              f"eikonal L2={row['models']['eikonal']['full_record_relative_l2']:.3f}; "
              f"straight L2={row['models']['straight']['full_record_relative_l2']:.3f}", flush=True)
    for anchor in anchors.values():
        del anchor["prediction"]
    summary = {}
    for axis in protocol["axes"]:
        selected = [row for row in rows if row["axis"] == axis]
        summary[axis] = {name: {key: float(np.mean([row["models"][name][key] for row in selected]))
                                for key in ("full_record_relative_l2", "active_sample_relative_l2",
                                            "change_relative_l2", "change_cosine")}
                         for name in models}
    output = {"purpose": protocol["purpose"], "role": protocol["role"],
              "protocol_sha256": hash_file(PROTOCOL),
              "checkpoint_sha256": {name: protocol[f"{name}_sha256"] for name in models},
              "velocity_sha256": protocol["velocity_sha256"], "anchors": anchors,
              "events": rows, "axis_means": summary, "limitations": protocol["interpretation"]}
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
