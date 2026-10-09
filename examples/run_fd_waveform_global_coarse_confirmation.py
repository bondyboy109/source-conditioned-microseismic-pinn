"""Run a coarse global FD waveform baseline on the locked confirmation events.

The search does not use FMM estimates. It evaluates a fixed global spatial grid
and optimizes only the origin-time shift for each FD candidate. The result is a
low-resolution cost pilot because the spatial grid is intentionally limited to
3 x 3 x 2 points.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fd_waveform import best_origin_time, simulate_candidates


PROTOCOL = ROOT / "experiments/fd_waveform_global_coarse_confirmation_protocol.json"


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


def render(result: dict) -> str:
    lines = [
        "# Coarse global FD waveform confirmation baseline", "",
        "The FD branch searches the full registered source bounds on a fixed "
        "3 x 3 x 2 spatial grid and optimizes origin time by discrete waveform "
        "shifting. It does not use FMM estimates. The spatial resolution is "
        "coarse and is reported explicitly as a baseline limitation.", "",
        "| Event | FD location error (m) | FD origin error (s) | FD waveform relative L2 | Candidates | Runtime (s) | Peak CUDA MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["events"]:
        memory = row["peak_cuda_memory_mib"]
        memory_text = "--" if memory is None else f"{memory:.1f}"
        lines.append(
            f"| {row['id']} | {row['location_error_m']:.1f} | "
            f"{row['origin_time_error_s']:.4f} | {row['relative_l2']:.4f} | "
            f"{row['candidate_count']} | {row['runtime_seconds']:.1f} | {memory_text} |"
        )
    summary = result["summary"]
    lines.extend([
        f"| **Median** | **{summary['median_location_error_m']:.1f}** | "
        f"**{summary['median_origin_time_error_s']:.4f}** | "
        f"**{summary['median_relative_l2']:.4f}** | -- | "
        f"**{np.median([row['runtime_seconds'] for row in result['events']]):.1f}** | -- |",
        "", f"The repeated FD template syntheses took "
        f"{summary['total_runtime_seconds']:.1f} s in total. The candidate sources "
        "are identical for every event, so the templates can be cached; the "
        "reported per-event synthesis time is not a cached online cost. "
        "Cached ranking time was not instrumented.",
        "", result["limitations"], "",
    ])
    return "\n".join(lines)


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path = ROOT / spec["result"]
    report_path = ROOT / spec["report"]
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
    rows = []
    checkpoint_path = ROOT / spec.get(
        "checkpoint", "experiments/fd_waveform_global_coarse_confirmation_checkpoint.json"
    )
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("spatial_grid_counts") != spec["spatial_grid_counts"]:
            raise ValueError("Checkpoint grid does not match protocol")
        rows.extend(checkpoint.get("events", []))
    completed = {row["id"] for row in rows}
    for event in spec["events"]:
        if event["id"] in completed:
            print(f"{event['id']}: loaded from checkpoint", flush=True)
            continue
        observation_path = ROOT / event["observation"]
        expected_hash = event["observation_sha256"]
        if digest(observation_path) != expected_hash:
            raise ValueError(f"Observation changed: {observation_path}")
        with np.load(observation_path) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            selected = ((times >= spec["waveform_window_s"][0])
                        & (times <= spec["waveform_window_s"][1] + 1e-9))
            observed = np.asarray(archive["data"][:, selected], dtype=np.float32)

        templates, fd_report = simulate_candidates(
            config_path, candidates,
            duration_s=spec["template_duration_s"],
            origin_time_s=spec["reference_origin_time_s"],
            batch_size=spec["batch_size"], device=device)
        scores = [best_origin_time(
            template, observed, fd_report["time_step_s"],
            spec["reference_origin_time_s"], tuple(spec["origin_time_bounds_s"]))
            for template in templates]
        winner_index = min(range(len(scores)), key=lambda index: scores[index]["normalized_mse"])
        estimate = np.append(candidates[winner_index], scores[winner_index]["origin_time_s"])
        with np.load(observation_path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        delta = estimate - truth
        winner_score = scores[winner_index]
        rows.append({
            "id": event["id"],
            "observation_sha256": expected_hash,
            "estimated_source_m_s": estimate.tolist(),
            "source_truth_m_s_post_ranking": truth.tolist(),
            "component_error_m_s": delta.tolist(),
            "location_error_m": float(np.linalg.norm(delta[:3])),
            "origin_time_error_s": float(abs(delta[3])),
            "relative_l2": float(winner_score["relative_l2"]),
            "normalized_mse": float(winner_score["normalized_mse"]),
            "lag_samples": int(winner_score["lag_samples"]),
            "candidate_count": int(len(candidates)),
            "candidate_grid_xyz_m": candidates.tolist(),
            "runtime_seconds": float(fd_report["runtime_seconds"]),
            "batch_seconds": fd_report["batch_seconds"],
            "peak_cuda_memory_mib": (
                fd_report["peak_cuda_memory_bytes"] / 2**20
                if fd_report["peak_cuda_memory_bytes"] else None),
        })
        print(
            f"{event['id']}: FD={rows[-1]['location_error_m']:.1f} m, "
            f"et={rows[-1]['origin_time_error_s']:.4f} s, "
            f"runtime={rows[-1]['runtime_seconds']:.1f} s",
            flush=True,
        )
        checkpoint_path.write_text(json.dumps({
            "role": spec["role"],
            "spatial_grid_counts": spec["spatial_grid_counts"],
            "events": rows,
        }, indent=2), encoding="utf-8")

    result = {
        "role": spec["role"],
        "protocol_sha256": digest(PROTOCOL),
        "spatial_grid_counts": spec["spatial_grid_counts"],
        "events": rows,
        "summary": {
            "event_count": len(rows),
            "median_location_error_m": float(np.median([row["location_error_m"] for row in rows])),
            "mean_location_error_m": float(np.mean([row["location_error_m"] for row in rows])),
            "median_origin_time_error_s": float(np.median([row["origin_time_error_s"] for row in rows])),
            "median_relative_l2": float(np.median([row["relative_l2"] for row in rows])),
            "total_runtime_seconds": float(sum(row["runtime_seconds"] for row in rows)),
        },
        "limitations": spec["limitations"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(render(result), encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()
