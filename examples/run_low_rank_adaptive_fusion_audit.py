"""True adaptive-weight audit for the frozen source-conditioned low-rank model."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.cross_model_geology_audit import fmm_locate
from examples.invert_fmm_receiver_fields import receiver_fields
from examples.run_low_rank_pick_perturbation_audit import (
    corrected_picks, crop_to_model_window, load_model_with_fields, run_branch,
)
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, choose_diverse_starts, coarse_grid, load_event,
    sampled_waveforms, waveform_scores,
)

ROOT_PROTOCOL = ROOT / "experiments/low_rank_adaptive_fusion_protocol.json"
BASE_PROTOCOL = ROOT / "experiments/low_rank_pick_perturbation_protocol.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def adaptive_alpha(valid_picks: int, spec: dict) -> tuple[float, str]:
    if valid_picks < int(spec["minimum_valid_picks"]):
        return 1.0, "fmm_fallback"
    alpha = float(spec["min_hybrid_weight"] +
                  (spec["max_hybrid_weight"] - spec["min_hybrid_weight"]) *
                  np.clip((valid_picks - spec["minimum_valid_picks"]) / 6.0, 0.0, 1.0))
    return alpha, "adaptive_hybrid"


def error(best: dict, truth: np.ndarray) -> dict:
    estimate = np.asarray(best["estimated_source_m_s"], dtype=np.float64)
    delta = estimate - truth
    return {"estimated_source_m_s": estimate.tolist(),
            "location_error_m": float(np.linalg.norm(delta[:3])),
            "origin_time_error_s": float(abs(delta[3])),
            "waveform_normalized_mse": float(best.get("waveform_normalized_mse", np.nan)),
            "arrival_normalized_mse": float(best.get("arrival_normalized_mse", np.nan))}


def main() -> None:
    spec = json.loads(ROOT_PROTOCOL.read_text(encoding="utf-8"))
    base = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["output_result"]
    report = ROOT / spec["output_report"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite adaptive-fusion result")
    checkpoint = ROOT / base["checkpoint"]
    velocity_path = ROOT / base["velocity_file"]
    receivers_path = ROOT / base["receivers_file"]
    if sha256(checkpoint) != base["checkpoint_sha256"] or sha256(velocity_path) != base["velocity_sha256"]:
        raise ValueError("Frozen checkpoint or velocity hash changed")
    receivers = np.asarray(np.load(receivers_path), dtype=np.float32)
    if sha256(receivers_path) != base["receivers_sha256"]:
        raise ValueError("Receiver geometry hash changed")
    velocity = np.asarray(np.load(velocity_path, mmap_mode="r")[::2, ::2, ::2], dtype=np.float32)
    cache = ROOT / "outputs/low_rank_pick_perturbation/fmm_fields_stride2.npy"
    fields = np.load(cache, mmap_mode="r") if cache.exists() else receiver_fields(velocity, receivers, float(base["fmm_spacing_m"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain, _ = load_model_with_fields(checkpoint, fields, receivers, float(base["fmm_spacing_m"]), device)
    bounds = np.asarray(base["source_bounds_m_s"], dtype=np.float32)
    search, arrival = base["search"], base["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    for event_spec in base["events"]:
        path = ROOT / base["test_directory"] / event_spec["file"]
        event = crop_to_model_window(load_event(path, receivers), float(domain[3]))
        picks = corrected_picks(event, arrival)
        valid = np.isfinite(picks)
        coarse = sampled_waveforms(event, domain, search["coarse_receiver_stride"], search["coarse_time_stride"], device)
        fine = sampled_waveforms(event, domain, search["refine_receiver_stride"], search["refine_time_stride"], device)
        waveform = waveform_scores(model, candidates, domain, coarse, search["candidate_batch"])
        truth = np.asarray(event_spec["truth_m_s"], dtype=np.float64)
        for offset in spec["offset_samples"]:
            for seed in spec["seeds"]:
                rng = np.random.default_rng(int(seed) + int(offset) * 100003)
                perturbed = picks.copy()
                perturbed[valid] += rng.normal(0.0, float(offset) * float(base["dt_s"]), size=int(valid.sum())).astype(np.float32)
                enough = int(valid.sum()) >= int(spec["minimum_valid_picks"])
                travel = arrival_scores(model, candidates, perturbed, arrival["arrival_scale_s"], search["candidate_batch"]) if enough else np.full(len(candidates), np.inf)
                branches = {}
                for name, alpha in (("waveform_only", 0.0), ("fixed_hybrid", float(arrival["hybrid_weight"]))):
                    if alpha and not enough:
                        branches[name] = {"status": "failure", "location_error_m": None, "origin_time_error_s": None}
                        continue
                    best = run_branch(model, candidates, waveform, travel, alpha, domain, bounds, fine, perturbed, search, arrival)
                    branches[name] = error(best, truth)
                alpha, decision = adaptive_alpha(int(valid.sum()), spec)
                if decision == "fmm_fallback":
                    branches["adaptive"] = fmm_locate(fields, {"truth": truth}, perturbed, bounds[:3], float(base["fmm_spacing_m"]), search, arrival)
                    branches["adaptive"]["decision"] = decision
                else:
                    best = run_branch(model, candidates, waveform, travel, alpha, domain, bounds, fine, perturbed, search, arrival)
                    branches["adaptive"] = error(best, truth)
                    branches["adaptive"]["decision"] = decision
                fmm = fmm_locate(fields, {"truth": truth}, perturbed, bounds[:3], float(base["fmm_spacing_m"]), search, arrival) if enough else {"status": "failure", "location_error_m": None, "origin_time_error_s": None}
                branches["fmm"] = fmm
                rows.append({"event": event_spec["name"], "offset_samples_sd": int(offset), "seed": int(seed),
                             "valid_picks": int(valid.sum()), "adaptive_alpha": float(alpha),
                             "branches": branches, "truth_m_s": truth.tolist()})
        print(f"{event_spec['name']}: completed", flush=True)
    summary = {}
    for offset in spec["offset_samples"]:
        selected = [r for r in rows if r["offset_samples_sd"] == offset]
        out = {"n": len(selected), "branches": {}}
        for name in ("waveform_only", "fixed_hybrid", "adaptive", "fmm"):
            good = [r["branches"][name]["location_error_m"] for r in selected if r["branches"][name].get("location_error_m") is not None]
            out["branches"][name] = {"n_success": len(good), "failure_count": len(selected)-len(good),
                                      "mean_eloc_m": float(np.mean(good)) if good else None,
                                      "median_eloc_m": float(np.median(good)) if good else None}
        summary[str(offset)] = out
    result = {"purpose": spec["purpose"], "protocol": str(ROOT_PROTOCOL.resolve()), "base_protocol_sha256": sha256(BASE_PROTOCOL),
              "summary": summary, "events": rows, "limitations": spec["limitations"]}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# True adaptive fusion audit", "", spec["limitations"], "",
             "| Pick SD (samples) | Waveform median | Fixed hybrid median | Adaptive median | FMM median |", "|---:|---:|---:|---:|---:|"]
    for offset in spec["offset_samples"]:
        s = summary[str(offset)]["branches"]
        fmt = lambda x: "NA" if x is None else f"{x:.1f}"
        lines.append(f"| {offset} | {fmt(s['waveform_only']['median_eloc_m'])} | {fmt(s['fixed_hybrid']['median_eloc_m'])} | {fmt(s['adaptive']['median_eloc_m'])} | {fmt(s['fmm']['median_eloc_m'])} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
