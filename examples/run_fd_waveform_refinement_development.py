"""Run the registered FMM-initialized local FD waveform development pilot."""

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


PROTOCOL = ROOT / "experiments/fd_waveform_refinement_development_protocol.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def candidate_grid(center: np.ndarray, offsets: dict, bounds: np.ndarray) -> np.ndarray:
    rows = []
    for offset in itertools.product(offsets["x"], offsets["y"], offsets["z"]):
        rows.append(np.clip(center + np.asarray(offset, dtype=np.float64),
                            bounds[:, 0], bounds[:, 1]))
    return np.unique(np.asarray(rows), axis=0)


def render(result: dict) -> str:
    lines = [
        "# FMM-initialized local FD waveform refinement: development pilot", "",
        "This pilot uses already-inspected events. It measures feasibility and cost; it is not an independent or global FD inversion.", "",
        "| Event | FMM error (m) | Local FD error (m) | FMM origin error (s) | Local FD origin error (s) | FD candidates | FD runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["events"]:
        lines.append(
            f"| {row['id']} | {row['fmm']['location_error_m']:.1f} | "
            f"{row['fd_waveform']['location_error_m']:.1f} | "
            f"{row['fmm']['origin_time_error_s']:.4f} | "
            f"{row['fd_waveform']['origin_time_error_s']:.4f} | "
            f"{row['candidate_count']} | {row['fd_runtime_seconds']:.1f} |"
        )
    summary = result["summary"]
    lines.extend([
        f"| **Mean** | **{summary['fmm_mean_location_error_m']:.1f}** | **{summary['fd_mean_location_error_m']:.1f}** | **{summary['fmm_mean_origin_time_error_s']:.4f}** | **{summary['fd_mean_origin_time_error_s']:.4f}** | -- | **{summary['fd_total_runtime_seconds']:.1f}** |",
        "", "The FD branch evaluates physical forward solves around the frozen FMM estimate and selects origin time by a zero-padded discrete delay. It fits no waveform gain. The timing excludes the preceding FMM field setup and search.",
        "", result["limitations"], "",
    ])
    return "\n".join(lines)


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path = ROOT / spec["result"]
    report_path = ROOT / spec["report"]
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Development result already exists; refusing to overwrite")
    frozen = ((spec["fmm_results"], spec["fmm_results_sha256"]),
              (spec["fd_config"], spec["fd_config_sha256"]),
              (spec["reference_fd_implementation"],
               spec["reference_fd_implementation_sha256"]),
              (spec["implementation"], spec["implementation_sha256"]))
    for relative, expected in frozen:
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")

    fmm = json.loads((ROOT / spec["fmm_results"]).read_text(encoding="utf-8"))
    fmm_by_id = {row["id"]: row for row in fmm["events"]}
    config = json.loads((ROOT / spec["fd_config"]).read_text(encoding="utf-8"))
    spatial_bounds = np.asarray(config["source_location_bounds_m"], dtype=np.float64).T
    if spatial_bounds.shape != (3, 2):
        raise ValueError("Invalid source bounds")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for event in spec["events"]:
        observation_path = ROOT / event["observation"]
        if digest(observation_path) != event["observation_sha256"]:
            raise ValueError(f"Observation changed: {observation_path}")
        fmm_row = fmm_by_id[event["id"]]
        center = np.asarray(fmm_row["refined"]["estimated_source_m_s"][:3],
                            dtype=np.float64)
        candidates = candidate_grid(center, spec["candidate_offsets_m"], spatial_bounds)
        with np.load(observation_path) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            selected = ((times >= spec["waveform_window_s"][0])
                        & (times <= spec["waveform_window_s"][1] + 1e-9))
            observed = np.asarray(archive["data"][:, selected], dtype=np.float32)
        templates, fd_report = simulate_candidates(
            ROOT / spec["fd_config"], candidates,
            duration_s=(spec["waveform_window_s"][1]
                        + spec["reference_origin_time_s"]
                        - spec["origin_time_bounds_s"][0]),
            origin_time_s=spec["reference_origin_time_s"],
            batch_size=spec["batch_size"], device=device)
        scores = [best_origin_time(
            template, observed, fd_report["time_step_s"],
            spec["reference_origin_time_s"], tuple(spec["origin_time_bounds_s"]))
                  for template in templates]
        winner_index = min(range(len(scores)), key=lambda index: scores[index]["normalized_mse"])
        estimate = np.append(candidates[winner_index], scores[winner_index]["origin_time_s"])
        # The source truth is read only after the fixed candidate ranking.
        with np.load(observation_path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        fmm_estimate = np.asarray(fmm_row["refined"]["estimated_source_m_s"],
                                  dtype=np.float64)
        delta = estimate - truth
        fmm_delta = fmm_estimate - truth
        rows.append({
            "id": event["id"], "observation_sha256": event["observation_sha256"],
            "source_truth_m_s_post_ranking": truth.tolist(),
            "fmm": {"estimated_source_m_s": fmm_estimate.tolist(),
                    "location_error_m": float(np.linalg.norm(fmm_delta[:3])),
                    "origin_time_error_s": float(abs(fmm_delta[3]))},
            "fd_waveform": {"estimated_source_m_s": estimate.tolist(),
                            "component_error_m_s": delta.tolist(),
                            "location_error_m": float(np.linalg.norm(delta[:3])),
                            "origin_time_error_s": float(abs(delta[3])),
                            **scores[winner_index]},
            "candidate_count": int(len(candidates)),
            "candidate_source_xyz_m": candidates.tolist(),
            "candidate_scores": scores,
            "fd_runtime_seconds": fd_report["runtime_seconds"],
            "fd_peak_cuda_memory_mib": (fd_report["peak_cuda_memory_bytes"] / 2**20
                                         if fd_report["peak_cuda_memory_bytes"] else None),
            "fd_batch_seconds": fd_report["batch_seconds"],
        })
        print(f"{event['id']}: FMM={rows[-1]['fmm']['location_error_m']:.1f} m; "
              f"FD={rows[-1]['fd_waveform']['location_error_m']:.1f} m; "
              f"runtime={fd_report['runtime_seconds']:.1f} s", flush=True)

    result = {
        "role": spec["role"], "protocol_sha256": digest(PROTOCOL),
        "events": rows,
        "summary": {
            "event_count": len(rows),
            "fmm_mean_location_error_m": float(np.mean([
                row["fmm"]["location_error_m"] for row in rows])),
            "fd_mean_location_error_m": float(np.mean([
                row["fd_waveform"]["location_error_m"] for row in rows])),
            "fmm_mean_origin_time_error_s": float(np.mean([
                row["fmm"]["origin_time_error_s"] for row in rows])),
            "fd_mean_origin_time_error_s": float(np.mean([
                row["fd_waveform"]["origin_time_error_s"] for row in rows])),
            "fd_total_runtime_seconds": float(sum(
                row["fd_runtime_seconds"] for row in rows)),
        },
        "limitations": spec["interpretation"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(render(result), encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()
