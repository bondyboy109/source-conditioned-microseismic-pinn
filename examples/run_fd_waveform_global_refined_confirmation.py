"""Run a cached higher-resolution global FD waveform baseline.

The 75 spatial templates are generated once, without FMM initialization, and
then scored against all eight locked confirmation records. This separates FD
forward-synthesis cost from per-event candidate ranking.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fd_waveform import best_origin_time, simulate_candidates

PROTOCOL = ROOT / "experiments/fd_waveform_global_refined_confirmation_protocol.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def global_grid(bounds: np.ndarray, counts: dict[str, int]) -> np.ndarray:
    axes = [
        np.linspace(bounds[axis, 0], bounds[axis, 1], counts[name])
        for axis, name in enumerate(("x", "y", "z"))
    ]
    return np.asarray(list(itertools.product(*axes)), dtype=np.float64)


def observed_record(path: Path, window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        times = np.asarray(archive["times"], dtype=np.float64)
        selected = (times >= window[0]) & (times <= window[1] + 1e-9)
        return (np.asarray(archive["data"][:, selected], dtype=np.float32),
                np.asarray(archive["source_truth"], dtype=np.float64))


def render(result: dict) -> str:
    lines = [
        "# Cached higher-resolution global FD waveform confirmation baseline", "",
        "The FD branch searches the full registered source bounds on a fixed "
        "5 x 5 x 3 spatial grid and optimizes origin time by discrete waveform "
        "shifting. It does not use FMM estimates. The 75 templates are generated "
        "once and reused for all events.", "",
        "| Event | Grid floor (m) | FD location error (m) | FD origin error (s) | Relative L2 | Ranking (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["events"]:
        lines.append(
            f"| {row['id']} | {row['nearest_grid_error_m']:.1f} | "
            f"{row['location_error_m']:.1f} | {row['origin_time_error_s']:.4f} | "
            f"{row['relative_l2']:.4f} | {row['ranking_seconds']:.4f} |"
        )
    summary = result["summary"]
    lines.extend([
        f"| **Median** | **{summary['median_nearest_grid_error_m']:.1f}** | "
        f"**{summary['median_location_error_m']:.1f}** | "
        f"**{summary['median_origin_time_error_s']:.4f}** | "
        f"**{summary['median_relative_l2']:.4f}** | "
        f"**{summary['median_ranking_seconds']:.4f}** |",
        "",
        f"Template synthesis took {result['template_runtime_seconds']:.1f} s "
        f"for {result['candidate_count']} candidates and reached "
        f"{result['peak_cuda_memory_mib']:.1f} MiB peak CUDA allocation. "
        "The templates were reused for all events; ranking time excludes FD synthesis.",
        "",
        result["limitations"], "",
    ])
    return "\n".join(lines)


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path = ROOT / spec["result"]
    report_path = ROOT / spec["report"]
    checkpoint_path = ROOT / spec["checkpoint"]
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Confirmation result already exists; refusing to overwrite")

    config_path = ROOT / spec["fd_config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bounds = np.asarray(config["source_location_bounds_m"], dtype=np.float64).T
    candidates = global_grid(bounds, spec["spatial_grid_counts"])
    expected_count = int(np.prod([spec["spatial_grid_counts"][name]
                                  for name in ("x", "y", "z")]))
    if len(candidates) != expected_count:
        raise ValueError(f"Unexpected candidate count: {len(candidates)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating and caching {len(candidates)} FD templates on {device}...", flush=True)
    templates, fd_report = simulate_candidates(
        config_path, candidates,
        duration_s=spec["template_duration_s"],
        origin_time_s=spec["reference_origin_time_s"],
        batch_size=spec["batch_size"], device=device)
    rows = []
    window = tuple(spec["waveform_window_s"])
    for event in spec["events"]:
        observation_path = ROOT / event["observation"]
        if digest(observation_path) != event["observation_sha256"]:
            raise ValueError(f"Observation changed: {observation_path}")
        observed, truth = observed_record(observation_path, window)
        started = time.perf_counter()
        scores = [best_origin_time(
            template, observed, fd_report["time_step_s"],
            spec["reference_origin_time_s"], tuple(spec["origin_time_bounds_s"]))
            for template in templates]
        ranking_seconds = time.perf_counter() - started
        winner_index = min(range(len(scores)), key=lambda index: scores[index]["normalized_mse"])
        estimate = np.append(candidates[winner_index], scores[winner_index]["origin_time_s"])
        delta = estimate - truth
        nearest_grid_error = float(np.min(np.linalg.norm(candidates - truth[:3], axis=1)))
        winner_score = scores[winner_index]
        row = {
            "id": event["id"],
            "observation_sha256": event["observation_sha256"],
            "estimated_source_m_s": estimate.tolist(),
            "source_truth_m_s": truth.tolist(),
            "component_error_m_s": delta.tolist(),
            "nearest_grid_error_m": nearest_grid_error,
            "location_error_m": float(np.linalg.norm(delta[:3])),
            "origin_time_error_s": float(abs(delta[3])),
            "relative_l2": float(winner_score["relative_l2"]),
            "normalized_mse": float(winner_score["normalized_mse"]),
            "lag_samples": int(winner_score["lag_samples"]),
            "candidate_count": int(len(candidates)),
            "ranking_seconds": float(ranking_seconds),
        }
        rows.append(row)
        checkpoint_path.write_text(json.dumps({
            "role": spec["role"],
            "spatial_grid_counts": spec["spatial_grid_counts"],
            "candidate_count": len(candidates),
            "template_runtime_seconds": fd_report["runtime_seconds"],
            "peak_cuda_memory_mib": (fd_report["peak_cuda_memory_bytes"] / 2**20
                                      if fd_report["peak_cuda_memory_bytes"] else None),
            "events": rows,
        }, indent=2), encoding="utf-8")
        print(f"{event['id']}: FD={row['location_error_m']:.1f} m, "
              f"et={row['origin_time_error_s']:.4f} s, "
              f"ranking={row['ranking_seconds']:.3f} s", flush=True)

    result = {
        "role": spec["role"],
        "protocol_sha256": digest(PROTOCOL),
        "spatial_grid_counts": spec["spatial_grid_counts"],
        "candidate_count": len(candidates),
        "template_runtime_seconds": float(fd_report["runtime_seconds"]),
        "peak_cuda_memory_mib": (fd_report["peak_cuda_memory_bytes"] / 2**20
                                  if fd_report["peak_cuda_memory_bytes"] else None),
        "events": rows,
        "summary": {
            "event_count": len(rows),
            "median_nearest_grid_error_m": float(np.median([r["nearest_grid_error_m"] for r in rows])),
            "mean_location_error_m": float(np.mean([r["location_error_m"] for r in rows])),
            "median_location_error_m": float(np.median([r["location_error_m"] for r in rows])),
            "median_origin_time_error_s": float(np.median([r["origin_time_error_s"] for r in rows])),
            "median_relative_l2": float(np.median([r["relative_l2"] for r in rows])),
            "median_ranking_seconds": float(np.median([r["ranking_seconds"] for r in rows])),
        },
        "limitations": spec["limitations"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(render(result), encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()
