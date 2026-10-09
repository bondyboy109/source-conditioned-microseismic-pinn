"""Compare frozen FD objectives on three newly generated source positions."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.audit_fd_joint_time_amplitude import score_joint
from examples.run_fd_waveform_global_refined_confirmation import global_grid, observed_record
from fd_waveform import best_origin_time, simulate_candidates

PROTOCOL = ROOT / "experiments/fd_joint_gain_fresh_source_protocol.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize(scores: list[dict], candidates: np.ndarray, truth: np.ndarray,
              seconds: float, gain_fitted: bool) -> dict:
    index = min(range(len(scores)), key=lambda i: scores[i]["normalized_mse"])
    score = scores[index]
    estimate = np.append(candidates[index], score["origin_time_s"])
    return {
        "candidate_index": index,
        "estimated_source_m_s": estimate.tolist(),
        "location_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
        "origin_time_error_s": float(abs(estimate[3] - truth[3])),
        "normalized_mse": score["normalized_mse"],
        "relative_l2": math.sqrt(score["normalized_mse"]),
        "nonnegative_gain": score["nonnegative_gain"] if gain_fitted else 1.0,
        "ranking_seconds": seconds,
    }


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path = ROOT / spec["result"]
    report_path = ROOT / spec["report"]
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Fresh-source results already exist; refusing to overwrite")
    cfg_path = ROOT / spec["fd_config"]
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    bounds = np.asarray(config["source_location_bounds_m"], dtype=np.float64).T
    candidates = global_grid(bounds, spec["spatial_grid_counts"])
    if len(candidates) != 75:
        raise ValueError("Expected the frozen 75-candidate grid")
    observation_dir = ROOT / spec["observation_directory"]
    observation_dir.mkdir(parents=True, exist_ok=True)
    observations = []
    for event in spec["observations"]:
        path = observation_dir / f"{event['id']}.npz"
        if not path.exists():
            subprocess.run([
                sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                "--config", str(cfg_path), "--output", str(path), "--duration", "2.0",
                "--source", *map(str, event["source_m_s"]),
            ], cwd=ROOT, check=True)
        observed, truth = observed_record(path, tuple(spec["waveform_window_s"]))
        if not np.allclose(truth, event["source_m_s"], atol=1e-5):
            raise ValueError(f"Observation has wrong source truth: {path}")
        observations.append((event["id"], path, observed, truth))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Synthesizing {len(candidates)} FD candidate templates on {device}", flush=True)
    templates, fd_report = simulate_candidates(
        cfg_path, candidates, spec["template_duration_s"],
        spec["reference_origin_time_s"], spec["template_batch_size"], device,
    )
    rows = []
    for name, path, observed, truth in observations:
        started = time.perf_counter()
        unit = [best_origin_time(template, observed, fd_report["time_step_s"],
                                 spec["reference_origin_time_s"],
                                 tuple(spec["origin_time_bounds_s"]))
                for template in templates]
        unit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        joint = [score_joint(template, observed, fd_report["time_step_s"],
                             spec["reference_origin_time_s"],
                             tuple(spec["origin_time_bounds_s"]))
                 for template in templates]
        joint_seconds = time.perf_counter() - started
        row = {
            "id": name, "observation": str(path.relative_to(ROOT)),
            "observation_sha256": digest(path), "truth_source_m_s": truth.tolist(),
            "unit": summarize(unit, candidates, truth, unit_seconds, False),
            "joint": summarize(joint, candidates, truth, joint_seconds, True),
        }
        rows.append(row)
        print(f"{name}: unit={row['unit']['location_error_m']:.1f} m; "
              f"joint={row['joint']['location_error_m']:.1f} m; "
              f"gain={row['joint']['nonnegative_gain']:.4f}", flush=True)
    median = lambda key, field: float(np.median([row[key][field] for row in rows]))
    result = {
        "role": spec["role"], "protocol_sha256": digest(PROTOCOL),
        "fd_config_sha256": digest(cfg_path),
        "generator_sha256": digest(ROOT / "examples/generate_fd_observation.py"),
        "solver_sha256": digest(ROOT / "fd_waveform.py"),
        "candidate_count": len(candidates), "device": str(device),
        "template_runtime_seconds": fd_report["runtime_seconds"],
        "peak_cuda_memory_mib": (fd_report["peak_cuda_memory_bytes"] / 2**20
                                 if fd_report["peak_cuda_memory_bytes"] else None),
        "events": rows, "summary": {
            key: {field: median(key, field) for field in (
                "location_error_m", "origin_time_error_s", "relative_l2", "ranking_seconds")}
            for key in ("unit", "joint")
        }, "limitations": spec["limitations"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Fresh-source FD objective check", "",
             "Three newly generated 5-Hz source positions; frozen 75-node grid, "
             "waveform window and origin-time bounds. Both objectives use identical "
             "templates and observations. Unit fits time only; joint fits time and "
             "one nonnegative amplitude per candidate.", "",
             "| Event | Grid floor (m) | Unit eloc (m) | Joint eloc (m) | "
             "Unit et (s) | Joint et (s) | Unit MSE | Joint MSE | Joint gain |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        u, j = row["unit"], row["joint"]
        floor = float(np.min(np.linalg.norm(candidates - row["truth_source_m_s"][:3], axis=1)))
        lines.append(f"| {row['id']} | {floor:.1f} | {u['location_error_m']:.1f} | "
                     f"{j['location_error_m']:.1f} | {u['origin_time_error_s']:.3f} | "
                     f"{j['origin_time_error_s']:.3f} | {u['normalized_mse']:.4f} | "
                     f"{j['normalized_mse']:.4f} | {j['nonnegative_gain']:.4f} |")
    lines.extend(["", "| Median | - | "
                  f"{result['summary']['unit']['location_error_m']:.1f} | "
                  f"{result['summary']['joint']['location_error_m']:.1f} | "
                  f"{result['summary']['unit']['origin_time_error_s']:.3f} | "
                  f"{result['summary']['joint']['origin_time_error_s']:.3f} | - | - | - |",
                  "", f"FD template synthesis: {fd_report['runtime_seconds']:.1f} s; "
                  f"median unit/joint ranking: {median('unit', 'ranking_seconds'):.3f}/"
                  f"{median('joint', 'ranking_seconds'):.3f} s.", "", spec["limitations"], ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {result_path} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
