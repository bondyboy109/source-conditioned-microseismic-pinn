"""Generate and evaluate the preregistered midpoint-teacher forward test."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_eikonal_training_generalization import evaluate
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/midpoint_teacher_confirmation_protocol.json"


def previous_source_positions(exclude: Path) -> list[tuple[Path, np.ndarray]]:
    records = []
    for path in sorted((ROOT / "data/observations").rglob("*.npz")):
        if exclude in path.parents:
            continue
        with np.load(path) as archive:
            if "source_truth" in archive.files:
                records.append((path, np.asarray(archive["source_truth"][:3], dtype=np.float64)))
    return records


def main():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    data_dir = ROOT / spec["data_directory"]
    result_path = ROOT / spec["result_file"]
    if data_dir.exists() or result_path.exists():
        raise FileExistsError("Confirmation data or result already exists; refusing to overwrite")
    registered = (("selected_checkpoint", "selected_sha256"),
                  ("control_checkpoint", "control_sha256"),
                  ("fd_config", "fd_config_sha256"))
    for path_key, digest_key in registered:
        if hash_file(ROOT / spec[path_key]) != spec[digest_key]:
            raise ValueError(f"Registered input changed: {spec[path_key]}")
    config = json.loads((ROOT / spec["fd_config"]).read_text(encoding="utf-8"))
    if (hash_file(Path(config["velocity_file"])) != spec["velocity_sha256"]
            or hash_file(Path(config["receivers_file"])) != spec["receiver_sha256"]):
        raise ValueError("FD velocity or receiver geometry changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for label in ("control", "selected"):
        model, metadata = load_conditioned_checkpoint(ROOT / spec[f"{label}_checkpoint"], device)
        model.eval().requires_grad_(False)
        records = metadata["training_record"]["receiver_observations"]
        if metadata["frequency_hz"] != spec["frequency_hz"] or len(records) != 33:
            raise ValueError(f"Unexpected {label} checkpoint")
        scale = np.asarray([metadata["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        loaded[label] = (model, scale, records)
    if ({row["sha256"] for row in loaded["control"][2]}
            != {row["sha256"] for row in loaded["selected"][2]}):
        raise ValueError("Control and selected checkpoints do not share the same 33 FD records")
    training_sources = np.asarray([row["source_truth"][:3]
                                   for row in loaded["selected"][2]], dtype=np.float64)
    sources = np.asarray([row["source_m_s"] for row in spec["events"]], dtype=np.float64)
    if len(spec["events"]) != 8 or len({tuple(row) for row in sources}) != 8:
        raise ValueError("Expected eight distinct registered sources")
    distances = np.min(np.linalg.norm(sources[:, None, :3] - training_sources[None, :, :],
                                      axis=2), axis=1)
    for row, distance in zip(spec["events"], distances):
        if (row["stratum"] == "local"
                and distance > spec["strata"]["local_max_nearest_training_distance_m"]):
            raise ValueError(f"Local source is too far from training coverage: {row}")
        if (row["stratum"] == "remote"
                and distance < spec["strata"]["remote_min_nearest_training_distance_m"]):
            raise ValueError(f"Remote source is too close to training coverage: {row}")
    previous = previous_source_positions(data_dir)
    for source in sources:
        for path, old in previous:
            if np.allclose(source[:3], old, atol=0.01, rtol=0):
                raise ValueError(f"Registered source already exists in {path}")
    print(f"Pre-generation audit passed: {len(previous)} existing observations; "
          f"nearest training distances={distances.round(1).tolist()}", flush=True)
    data_dir.mkdir(parents=True, exist_ok=False)
    paths = []
    generation_started = time.perf_counter()
    for row in spec["events"]:
        path = data_dir / row["file"]
        subprocess.run([sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                        "--config", str(ROOT / spec["fd_config"]),
                        "--duration", str(spec["duration_s"]),
                        "--output", str(path), "--source",
                        *(str(value) for value in row["source_m_s"])], cwd=ROOT, check=True)
        paths.append(path)
    generation_seconds = time.perf_counter() - generation_started
    results = []
    for row, path, distance in zip(spec["events"], paths, distances):
        with np.load(path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.allclose(archive["source_truth"], row["source_m_s"],
                                       atol=1e-5, rtol=0)):
                raise ValueError(f"Generated event differs from registration: {path}")
        fd = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (fd["method"] != "second_order_3d_acoustic_FD"
                or fd["duration_s"] != spec["duration_s"]
                or fd["boundary"] != "zero Dirichlet on all six faces"):
            raise ValueError(f"FD provenance differs: {path}")
        event = {"event": row["file"], "stratum": row["stratum"],
                 "source_m_s": row["source_m_s"], "sha256": hash_file(path),
                 "nearest_training_distance_m": float(distance),
                 "fd_runtime_seconds": fd["runtime_seconds"]}
        for label in ("control", "selected"):
            model, scale, _ = loaded[label]
            event[label] = evaluate(model, scale, path, event["sha256"])
        event["selected_improves_full_l2"] = (
            event["selected"]["full_record_relative_l2"]
            < event["control"]["full_record_relative_l2"])
        results.append(event)
        print(f"{row['file']}: control={event['control']['full_record_relative_l2']:.3f}, "
              f"selected={event['selected']['full_record_relative_l2']:.3f}", flush=True)
    selected_values = np.asarray([row["selected"]["full_record_relative_l2"]
                                  for row in results])
    local = [row for row in results if row["stratum"] == "local"]
    remote = [row for row in results if row["stratum"] == "remote"]
    gates = spec["predeclared_gates"]
    checks = {
        "all_selected_full_l2": bool(np.all(selected_values < gates["all_selected_full_l2_below"])),
        "selected_median_full_l2": float(np.median(selected_values))
        < gates["selected_median_full_l2_below"],
        "selected_local_mean_full_l2": float(np.mean([
            row["selected"]["full_record_relative_l2"] for row in local]))
        < gates["selected_local_mean_full_l2_below"],
        "selected_remote_mean_full_l2": float(np.mean([
            row["selected"]["full_record_relative_l2"] for row in remote]))
        < gates["selected_remote_mean_full_l2_below"],
        "selected_beats_control": sum(row["selected_improves_full_l2"] for row in results)
        >= gates["selected_beats_control_event_count_at_least"],
    }
    report = {"purpose": spec["purpose"], "role": spec["role"],
              "protocol_sha256": hash_file(PROTOCOL),
              "selected_sha256": spec["selected_sha256"],
              "control_sha256": spec["control_sha256"],
              "generation_seconds": generation_seconds,
              "summary": {
                  "selected_mean_full_l2": float(np.mean(selected_values)),
                  "selected_median_full_l2": float(np.median(selected_values)),
                  "control_mean_full_l2": float(np.mean([
                      row["control"]["full_record_relative_l2"] for row in results])),
                  "local_selected_mean_full_l2": float(np.mean([
                      row["selected"]["full_record_relative_l2"] for row in local])),
                  "remote_selected_mean_full_l2": float(np.mean([
                      row["selected"]["full_record_relative_l2"] for row in remote])),
                  "selected_improves_event_count": int(sum(
                      row["selected_improves_full_l2"] for row in results)),
              },
              "checks": checks, "all_gates_passed": all(checks.values()),
              "events": results, "limitations": spec["limitations"]}
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {result_path}; checks={checks}; all_pass={report['all_gates_passed']}",
          flush=True)


if __name__ == "__main__":
    main()
