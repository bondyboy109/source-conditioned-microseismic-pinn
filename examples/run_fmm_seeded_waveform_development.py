"""Use known-velocity FMM estimates to diagnose global PINN search failure."""

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
from examples.confirm_depth_gap_25 import refine
from source_conditioned_pinn import load_conditioned_checkpoint


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proposals(center: np.ndarray, bounds: np.ndarray,
              offsets: list[float], origin_times: np.ndarray) -> np.ndarray:
    """Construct truth-independent candidates around a supplied FMM position."""
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,) or bounds.shape != (4, 2):
        raise ValueError("Expected three FMM coordinates and four search bounds")
    spatial = sorted({tuple(np.clip(center + delta, bounds[:3, 0], bounds[:3, 1]))
                      for delta in itertools.product(offsets, repeat=3)})
    return np.asarray([(*xyz, float(origin)) for xyz in spatial
                       for origin in origin_times], dtype=np.float32)


def choose_spatial_starts(candidates: np.ndarray, scores: np.ndarray,
                          count: int, minimum_distance_m: float) -> list[int]:
    """Select the best time at each distinct spatial point, then diverse sites."""
    if len(candidates) != len(scores) or count < 1 or minimum_distance_m <= 0:
        raise ValueError("Invalid proposal scores or diversity threshold")
    selected = []
    for index in np.argsort(scores, kind="stable"):
        if all(np.linalg.norm(candidates[index, :3] - candidates[old, :3])
               >= minimum_distance_m for old in selected):
            selected.append(int(index))
            if len(selected) == count:
                break
    return selected


def fine_scores(model, receivers: np.ndarray, times: np.ndarray,
                data: np.ndarray, scale: np.ndarray,
                rms: float, candidates: np.ndarray,
                receiver_stride: int, time_stride: int, batch_size: int) -> np.ndarray:
    """Use the exact waveform_mse sampling/normalization of local refinement."""
    device = next(model.parameters()).device
    xyz = torch.as_tensor(receivers[::receiver_stride] / scale[:3], device=device)
    sampled_time = torch.as_tensor(times[::time_stride] / scale[3], device=device)
    observed = torch.as_tensor(data[::receiver_stride, ::time_stride], device=device)
    nr, nt = len(xyz), len(sampled_time)
    inputs = torch.cat((xyz[:, None, :].expand(nr, nt, 3),
                        sampled_time[None, :, None].expand(nr, nt, 1)),
                       dim=-1).reshape(-1, 4)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            batch = torch.as_tensor(candidates[start:start+batch_size] / scale,
                                    device=device)
            prediction = model(inputs.repeat(len(batch), 1),
                               batch.repeat_interleave(len(inputs), dim=0))
            prediction = prediction.reshape(len(batch), nr, nt)
            loss = (((prediction - observed[None, :, :]) / rms) ** 2).mean((1, 2))
            scores.extend(loss.cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def main() -> None:
    protocol = json.loads((ROOT / "experiments/fmm_seeded_waveform_development_protocol.json").read_text())
    registered = json.loads((ROOT / "experiments/cross_region_fmm_protocol.json").read_text())
    old = json.loads((ROOT / "experiments/cross_region_fmm_comparison.json").read_text())
    prior_pinn = json.loads((ROOT / "experiments/cross_region_coverage_localization_development.json").read_text())
    fmm = json.loads((ROOT / protocol["fmm_estimates"]).read_text())
    checkpoint = ROOT / protocol["model_checkpoint"]
    if (hash_file(checkpoint) != protocol["model_sha256"]
            or hash_file(ROOT / protocol["fmm_velocity"]) != protocol["fmm_velocity_sha256"]
            or protocol["refinement"]["bounds_m_s"] != registered["shared_source_bounds_m_s"]):
        raise ValueError("Frozen input or original bounds changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False)
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    training_hashes = {row["sha256"] for row in
                       metadata["training_record"]["receiver_observations"]}
    bounds = np.asarray(protocol["refinement"]["bounds_m_s"], dtype=np.float32)
    origin_spec = protocol["origin_time_grid_s"]
    origin_times = np.linspace(origin_spec["minimum"], origin_spec["maximum"],
                               round((origin_spec["maximum"] - origin_spec["minimum"])
                                     / origin_spec["step"]) + 1, dtype=np.float32)
    scoring = protocol["fine_scoring"]
    start_spec = protocol["initialization"]
    refine_config = {**registered["pinn_inversion"],
                     "refine_steps": protocol["refinement"]["steps"],
                     "refine_adam_learning_rate_on_box_fractions":
                     protocol["refinement"]["adam_learning_rate_on_box_fractions"]}
    if not (len(old["events"]) == len(fmm["events"]) == len(prior_pinn["rows"]) == 8):
        raise ValueError("Expected eight paired records")
    rows = []
    for baseline, arrival, old29 in zip(old["events"], fmm["events"], prior_pinn["rows"]):
        path = Path(baseline["observation"])
        if (hash_file(path) != baseline["sha256"] or baseline["sha256"] in training_hashes
                or arrival["sha256"] != baseline["sha256"]
                or old29["event"] != baseline["event"] or arrival["status"] != "ok"):
            raise ValueError(f"Event pairing or training overlap failed: {path}")
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        candidates = proposals(np.asarray(arrival["estimated_source_m_s"][:3]), bounds,
                               protocol["spatial_offsets_m_each_axis"], origin_times)
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
        scores = fine_scores(model, receivers, times, data, scale, rms, candidates,
                             scoring["receiver_stride"], scoring["time_stride"],
                             scoring["candidate_batch"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        proposal_seconds = time.perf_counter() - started
        indices = choose_spatial_starts(candidates, scores, start_spec["max_restarts"],
                                         start_spec["minimum_spatial_distance_m"])
        starts = [{"source_m_s": candidates[index].tolist(),
                   "coarse_normalized_mse": float(scores[index])} for index in indices]
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        best, refinements, _, _, _ = refine(
            model, receivers, times, torch.as_tensor(data, device=device),
            scale, bounds, refine_config, rms, starts)
        if device.type == "cuda":
            torch.cuda.synchronize()
        refine_seconds = time.perf_counter() - started
        # Truth is deliberately opened only after proposal ranking and refinement.
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float32)
        if not np.allclose(truth, baseline["source_truth_m_s"], atol=1e-5, rtol=0):
            raise ValueError("Source truth changed")
        estimate = np.asarray(best["estimated_source_m_s"])
        row = {"event": baseline["event"], "observation_sha256": baseline["sha256"],
               "fmm_spatial_center_m": arrival["estimated_source_m_s"][:3],
               "fmm_location_error_m_post_search": arrival["location_error_m"],
               "fmm_search_seconds_from_baseline": arrival["search_seconds"],
               "prior_pinn29_location_error_m": old29["augmented_location_m"],
               "candidate_count": len(candidates), "starts": [
                   {"source_m_s": item["source_m_s"],
                    "fine_normalized_mse": item["coarse_normalized_mse"]} for item in starts],
               "refinements": [{"estimated_source_m_s": item["estimated_source_m_s"],
                                "fine_normalized_mse": item["normalized_mse"],
                                "on_boundary": item["on_region_boundary"]}
                               for item in refinements],
               "estimated_source_m_s": estimate.tolist(),
               "final_normalized_mse": best["normalized_mse"],
               "location_error_m_post_search": float(np.linalg.norm(estimate[:3] - truth[:3])),
               "origin_time_error_s_post_search": float(abs(estimate[3] - truth[3])),
               "proposal_seconds": proposal_seconds, "refinement_seconds": refine_seconds}
        rows.append(row)
        print(f"{row['event']}: FMM={row['fmm_location_error_m_post_search']:.1f} m, "
              f"PINN-only={row['prior_pinn29_location_error_m']:.1f} m, "
              f"hybrid={row['location_error_m_post_search']:.1f} m, "
              f"search={proposal_seconds+refine_seconds:.1f}s", flush=True)
    output = {"purpose": "FMM-seeded PINN development diagnostic, not independent confirmation",
              "protocol": str((ROOT / "experiments/fmm_seeded_waveform_development_protocol.json").resolve()),
              "checkpoint_sha256": hash_file(checkpoint),
              "fmm_setup_seconds_from_baseline": fmm["setup_seconds"],
              "fmm_receiver_fields_mib_from_baseline": fmm["receiver_field_cache_mib"],
              "location_error_median": float(np.median([r["location_error_m_post_search"] for r in rows])),
              "rows": rows}
    target = ROOT / "experiments/fmm_seeded_waveform_development_results.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
