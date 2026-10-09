"""Fixed-rule retrospective MoE versus refined FMM on one shared candidate pool."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_arrival_spatial_resolution import rank_candidates
from examples.invert_fmm_receiver_fields import receiver_fields, spatial_grid
from examples.validate_arrival_constrained import (
    digest, load_model, load_protocol, pick_receivers, score_choice, waveform_losses,
)

PROTOCOL = ROOT / "experiments/refined_fmm_moe_diagnostic_protocol.json"


def candidate_pool(fields, picks, spec, *, seeds=3, rounds=3):
    """Reconstruct all points evaluated by the frozen refined FMM search."""
    bounds = np.asarray(spec["source_bounds_m"], dtype=np.float64)
    coarse = spatial_grid(bounds, tuple(spec["coarse_grid"]))
    scores, _ = rank_candidates(fields, picks, coarse, spec)
    current = coarse[np.argsort(scores, kind="stable")[:seeds]]
    step = (bounds[:, 1] - bounds[:, 0]) / (np.asarray(spec["coarse_grid"]) - 1)
    offsets = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)))
    visited = [coarse]
    for _ in range(rounds):
        step = step / 2
        trial = np.clip(current[:, None, :] + offsets[None, :, :] * step,
                        bounds[:, 0], bounds[:, 1]).reshape(-1, 3)
        candidates = np.unique(np.concatenate((current, trial)), axis=0)
        visited.append(candidates)
        rmse, _ = rank_candidates(fields, picks, candidates, spec)
        current = candidates[np.argsort(rmse, kind="stable")[:seeds]]
    pool = np.unique(np.concatenate(visited), axis=0)
    rmse, origin = rank_candidates(fields, picks, pool, spec)
    return np.column_stack((pool, origin)), rmse


def summarize(rows, method):
    values = [row["methods"][method] for row in rows]
    return {"events": len(values),
            "mean_location_error_m": float(np.mean([v["location_error_m"] for v in values])),
            "median_location_error_m": float(np.median([v["location_error_m"] for v in values])),
            "mean_origin_time_error_s": float(np.mean([v["origin_time_error_s"] for v in values]))}


def main():
    plan = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / plan["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_plan = ROOT / plan["source_protocol"]
    source_result = ROOT / plan["source_result"]
    for path, key in ((source_plan, "source_protocol_sha256"),
                      (source_result, "source_result_sha256"),
                      (ROOT / plan["base_checkpoint"], "base_checkpoint_sha256"),
                      (ROOT / "experiments/arrival_constrained_picker_development.json",
                       "previous_alpha_decision_sha256")):
        if digest(path) != plan[key]:
            raise ValueError(f"Registered input changed: {path}")
    registered = json.loads(source_plan.read_text(encoding="utf-8"))
    previous = json.loads(source_result.read_text(encoding="utf-8"))
    manifest_path = ROOT / registered["manifest"]
    if digest(manifest_path) != plan["source_manifest_sha256"]:
        raise ValueError("Registered FD manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = json.loads((ROOT / "experiments/arrival_constrained_picker_development.json")
                          .read_text(encoding="utf-8"))["selected"]
    spec = load_protocol()
    if (spec["model_sha256"] != plan["model_checkpoint_sha256"]
            or selected["alpha"] != plan["alpha"]
            or selected["picker"] != registered["picker"]
            or spec["arrival_scale_s"] != plan["arrival_scale_s"]
            or manifest["protocol_sha256"] != digest(source_plan)
            or previous["manifest_sha256"] != digest(manifest_path)):
        raise ValueError("Frozen model, picker, alpha, or FD registration differs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(spec, device)
    velocity = np.asarray(np.load(ROOT / spec["velocity"], mmap_mode="r")
                          [::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"]],
                          dtype=np.float32)
    receiver_geometry = np.load(ROOT / spec["receivers"])
    setup_start = time.perf_counter()
    fields = receiver_fields(velocity, receiver_geometry, spec["fmm_spacing_m"])
    setup_seconds = time.perf_counter() - setup_start
    rows = []
    for event, record, locked in zip(registered["test_events"], manifest["events"], previous["events"]):
        path = ROOT / registered["test_directory"] / (event["id"] + ".npz")
        if (event["id"] != record["id"] or event["id"] != locked["id"]
                or digest(path) != record["observation_sha256"]):
            raise ValueError(f"FD observation changed: {path}")
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"])
        if not np.array_equal(receivers, receiver_geometry):
            raise ValueError(f"Receiver layout changed: {path}")
        picks = pick_receivers(data, times, registered["picker"], spec) - registered["calibration_bias_s"]
        if np.isfinite(picks).sum() < spec["min_valid_picks"]:
            raise ValueError(f"Insufficient picks: {path}")
        start = time.perf_counter()
        sources, arrival_rmse = candidate_pool(fields, picks, spec)
        search_seconds = time.perf_counter() - start
        best_arrival = int(np.argmin(arrival_rmse))
        if (not np.allclose(sources[best_arrival], locked["refined"]["estimated_source_m_s"], atol=1e-7)
                or not np.isclose(arrival_rmse[best_arrival], locked["refined"]["arrival_rmse_s"], atol=1e-9)):
            raise ValueError(f"Candidate pool does not reproduce locked FMM baseline: {path}")
        start = time.perf_counter()
        waveform = waveform_losses(model, sources, data, times, receivers, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_seconds = time.perf_counter() - start
        hybrid = waveform + plan["alpha"] * (arrival_rmse / plan["arrival_scale_s"]) ** 2
        indices = {"refined_fmm": best_arrival,
                   "moe_waveform": int(np.argmin(waveform)),
                   "moe_hybrid": int(np.argmin(hybrid))}
        # True source is loaded after ranking all branches.
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if not np.allclose(truth, event["source_m_s"], atol=1e-5):
            raise ValueError(f"FD source differs from registered truth: {path}")
        methods = {}
        for name, idx in indices.items():
            methods[name] = {**score_choice(sources[idx], truth),
                             "candidate_index": idx,
                             "arrival_rmse_s": float(arrival_rmse[idx]),
                             "waveform_loss": float(waveform[idx]),
                             "hybrid_score": float(hybrid[idx])}
        rows.append({"id": event["id"], "observation_sha256": record["observation_sha256"],
                     "candidate_count": len(sources), "valid_picks": int(np.isfinite(picks).sum()),
                     "source_truth_m_s_post_search": truth.tolist(),
                     "candidate_search_seconds": search_seconds,
                     "moe_scoring_seconds": model_seconds, "methods": methods})
        print(f"{event['id']}: FMM={methods['refined_fmm']['location_error_m']:.0f}m "
              f"MoE={methods['moe_waveform']['location_error_m']:.0f}m "
              f"hybrid={methods['moe_hybrid']['location_error_m']:.0f}m", flush=True)
    summary = {name: summarize(rows, name) for name in
               ("refined_fmm", "moe_waveform", "moe_hybrid")}
    summary["hybrid_location_win_count"] = int(sum(
        row["methods"]["moe_hybrid"]["location_error_m"] <
        row["methods"]["refined_fmm"]["location_error_m"] for row in rows))
    summary["hybrid_same_candidate_count"] = int(sum(
        row["methods"]["moe_hybrid"]["candidate_index"] ==
        row["methods"]["refined_fmm"]["candidate_index"] for row in rows))
    result = {"role": plan["role"], "protocol_sha256": digest(PROTOCOL),
              "source_manifest_sha256": plan["source_manifest_sha256"],
              "fmm_setup_seconds": setup_seconds,
              "mean_candidate_search_seconds": float(np.mean([r["candidate_search_seconds"] for r in rows])),
              "mean_moe_scoring_seconds": float(np.mean([r["moe_scoring_seconds"] for r in rows])),
              "summary": summary, "events": rows, "limits": plan["limits"]}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
