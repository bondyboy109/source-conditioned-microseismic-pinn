"""Paired source-domain pretraining and target-domain transfer for low-rank operator."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.invert_fmm_receiver_fields import interpolate_fields
from examples.run_source_conditioned_low_rank_localization import load_model
from examples.run_traveltime_aligned_moe_development import (
    load_observations, relative_l2, sample_training_batch,
)
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, coarse_grid, choose_diverse_starts, load_event,
    pick_arrivals, refine, sampled_waveforms, waveform_scores,
)
from source_conditioned_low_rank import SourceConditionedLowRankOperator
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator

PROTOCOL = ROOT / "experiments/source_transfer_low_rank_protocol.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def new_model(fields: np.ndarray, receivers: np.ndarray, setup: dict,
              device: torch.device):
    m = setup["model"]
    model = SourceConditionedLowRankOperator(
        ReceiverTravelTimeInterpolator(torch.from_numpy(np.array(fields, copy=True)), 80.0),
        torch.from_numpy(receivers), tuple(m["spatial_lengths_m"]),
        time_length_s=float(m["time_length_s"]), frequency_hz=float(m["frequency_hz"]),
        rank=int(m["rank"]), hidden_dim=int(m["hidden_dim"]),
        hidden_layers=int(m["hidden_layers"]),
        causal_width_s=float(m["causal_width_s"])).to(device)
    return model


def phase(model, events, receivers, times, fields, setup, rng, updates, device):
    training = setup["training"]
    m = setup["model"]
    domain = np.asarray([*m["spatial_lengths_m"], m["time_length_s"]], dtype=np.float32)
    arrivals = interpolate_fields(fields,
                                  np.stack([event["source"][:3] for event in events]), 80.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    started = time.perf_counter()
    for step in range(1, updates + 1):
        model.train()
        points, sources, receiver_ids, target = sample_training_batch(
            events, receivers, times, arrivals, setup, rng, domain, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(points, sources, receiver_ids)[0]
        waveform = (output - target).square().mean()
        loss = waveform + float(training["coefficient_weight"]) * model.coefficient_regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                      float(training["gradient_clip_norm"]))
        optimizer.step()
        if step % 500 == 0:
            print(f"  step {step}/{updates}: loss={float(loss.detach()):.5f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads((ROOT / spec["parent_transfer_protocol"]).read_text(encoding="utf-8"))
    setup = json.loads((ROOT / spec["model_protocol"]).read_text(encoding="utf-8"))
    if digest(ROOT / spec["parent_transfer_protocol"]) != spec["parent_transfer_protocol_sha256"]:
        raise ValueError("Parent transfer protocol changed")
    if digest(ROOT / spec["model_protocol"]) != spec["model_protocol_sha256"]:
        raise ValueError("Low-rank model protocol changed")
    source_fields_path = ROOT / spec["source_fmm_fields"]
    target_fields_path = ROOT / spec["target_fmm_fields"]
    if digest(source_fields_path) != spec["source_fmm_fields_sha256"]:
        raise ValueError("Source FMM fields changed")
    if digest(target_fields_path) != spec["target_fmm_fields_sha256"]:
        raise ValueError("Target FMM fields changed")
    output_dir = ROOT / spec["output_directory"]
    result_path = ROOT / spec["result_file"]
    report_path = ROOT / spec["report_file"]
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite source-transfer results")
    manifest_path = ROOT / setup["training_manifest_checkpoint"]
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    target_paths = [Path(row["path"]) for row in
                    manifest["training_record"]["receiver_observations"]]
    if len(target_paths) != 33:
        raise ValueError("Target training manifest changed")
    source_paths = [ROOT / spec["source_observation_directory"] / f"source_{i:02d}.npz"
                    for i in parent["source_training_indices"]]
    test_paths = [ROOT / spec["test_observation_directory"] / row["file"]
                  for row in parent["test_events"]]
    if not all(path.is_file() for path in source_paths + target_paths + test_paths):
        raise FileNotFoundError("Source, target, or held-out records are missing")
    source_events, receivers, times = load_observations(source_paths)
    target_events, target_receivers, target_times = load_observations(target_paths)
    if not np.array_equal(receivers, target_receivers) or not np.array_equal(times, target_times):
        raise ValueError("Source and target acquisition differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_fields = np.load(source_fields_path, mmap_mode="r")
    target_fields = np.load(target_fields_path, mmap_mode="r")
    source_subset = [target_events[i] for i in parent["source_training_indices"]]
    checkpoints = {}
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            checkpoint = output_dir / f"{arm}_seed{seed}.pth"
            if checkpoint.exists():
                raise FileExistsError(f"Refusing to overwrite {checkpoint}")
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if device.type == "cuda": torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            model = new_model(source_fields if arm == "source_to_target" else target_fields,
                              receivers, setup, device)
            seconds = {}
            if arm != "target_only_1500":
                first_events = source_events if arm == "source_to_target" else source_subset
                first_fields = source_fields if arm == "source_to_target" else target_fields
                seconds["first_phase"] = phase(model, first_events, receivers, times,
                                                first_fields, setup, rng,
                                                int(spec["phase_updates"]), device)
                if arm == "source_to_target":
                    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    del model
                    if device.type == "cuda": torch.cuda.empty_cache()
                    model = new_model(target_fields, receivers, setup, device)
                    model.load_state_dict(weights)
            seconds["target_phase"] = phase(model, target_events, receivers, times,
                                             target_fields, setup, rng,
                                             int(spec["phase_updates"]), device)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "protocol_sha256": digest(PROTOCOL), "arm": arm, "seed": seed,
                        "seconds": seconds}, checkpoint)
            checkpoints[(arm, seed)] = checkpoint
            print(f"Saved {checkpoint.name}: {seconds}", flush=True)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()

    loc = json.loads((ROOT / "experiments/source_conditioned_low_rank_localization_protocol.json").read_text())
    bounds = np.asarray(loc["shared_source_bounds_m_s"], dtype=np.float32)
    search, arrival = loc["search"], loc["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    target_receivers = receivers
    rows = []
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            payload = torch.load(checkpoints[(arm, seed)], map_location="cpu", weights_only=False)
            model = new_model(target_fields, target_receivers, setup, device)
            model.load_state_dict(payload["state_dict"]); model.eval().requires_grad_(False)
            for path, truth_row in zip(test_paths, parent["test_events"]):
                event = load_event(path, target_receivers)
                _, picks = pick_arrivals(event, arrival)
                valid = int(np.isfinite(picks).sum())
                coarse = sampled_waveforms(event,
                                           np.asarray([*setup["model"]["spatial_lengths_m"], setup["model"]["time_length_s"]], dtype=np.float32),
                                           int(search["coarse_receiver_stride"]), int(search["coarse_time_stride"]), device)
                fine = sampled_waveforms(event,
                                         np.asarray([*setup["model"]["spatial_lengths_m"], setup["model"]["time_length_s"]], dtype=np.float32),
                                         int(search["refine_receiver_stride"]), int(search["refine_time_stride"]), device)
                domain = np.asarray([*setup["model"]["spatial_lengths_m"], setup["model"]["time_length_s"]], dtype=np.float32)
                started = time.perf_counter()
                wave = waveform_scores(model, candidates, domain, coarse, int(search["candidate_batch"]))
                fmm = arrival_scores(model, candidates, picks, float(arrival["arrival_scale_s"]), int(search["candidate_batch"]))
                score = wave + float(arrival["hybrid_weight"]) * fmm
                ids = choose_diverse_starts(candidates, score, int(search["restarts"]), float(search["minimum_start_separation_m"]))
                starts = [{"source_m_s": candidates[i].tolist(), "coarse_total_loss": float(score[i])} for i in ids]
                best, _ = refine(model, starts, domain, bounds, fine, picks, search, arrival, float(arrival["hybrid_weight"]))
                estimate = np.asarray(best["estimated_source_m_s"], dtype=np.float64)
                truth = np.asarray(truth_row["source_m_s"], dtype=np.float64)
                full = []
                with torch.inference_mode():
                    full_parts = []
                    shape = event["data"].shape
                    coords = np.concatenate((np.broadcast_to(target_receivers[:, None, :], (*shape, 3)), np.broadcast_to(event["times"][None, :, None], (*shape, 1))), axis=-1).reshape(-1, 4) / domain
                    sources = np.broadcast_to(truth / domain, (len(coords), 4)).astype(np.float32)
                    rid = np.repeat(np.arange(len(target_receivers)), len(event["times"]))
                    for start in range(0, len(coords), 8192):
                        full_parts.append(model(torch.as_tensor(coords[start:start+8192], device=device), torch.as_tensor(sources[start:start+8192], device=device), torch.as_tensor(rid[start:start+8192], device=device))[0].squeeze(1).cpu().numpy())
                    full = np.concatenate(full_parts).reshape(shape)
                rows.append({"seed": seed, "arm": arm, "event": path.name,
                             "truth_m_s": truth.tolist(), "estimate_m_s": estimate.tolist(),
                             "location_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
                             "origin_time_error_s": float(abs(estimate[3] - truth[3])),
                             "full_waveform_relative_l2": relative_l2(event["data"] / event["rms"], full),
                             "valid_picks": valid, "search_seconds": time.perf_counter() - started})
                print(f"{arm} seed={seed} {path.name}: eloc={rows[-1]['location_error_m']:.1f} m, L2={rows[-1]['full_waveform_relative_l2']:.3f}", flush=True)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
    summary = {}
    for arm in spec["arms"]:
        selected = [row for row in rows if row["arm"] == arm]
        summary[arm] = {"median_location_error_m": float(np.median([r["location_error_m"] for r in selected])),
                        "mean_location_error_m": float(np.mean([r["location_error_m"] for r in selected])),
                        "mean_origin_time_error_s": float(np.mean([r["origin_time_error_s"] for r in selected])),
                        "mean_full_waveform_relative_l2": float(np.mean([r["full_waveform_relative_l2"] for r in selected]))}
    result = {"purpose": spec["purpose"], "protocol_sha256": digest(PROTOCOL),
              "device": str(device), "summary": summary, "rows": rows,
              "interpretation": spec["interpretation"],
              "budget_clarification": "The two staged arms each use phase_updates twice; target_only_1500 uses phase_updates once. Only the staged arms match the total update budget. The frozen protocol's common-budget description is too broad."}
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Source-conditioned low-rank source-transfer result", "",
             "| Arm | Median eloc (m) | Mean eloc (m) | Mean et (s) | Mean full relative L2 |",
             "|---|---:|---:|---:|---:|"]
    for arm in spec["arms"]:
        row = summary[arm]
        lines.append(f"| {arm} | {row['median_location_error_m']:.1f} | {row['mean_location_error_m']:.1f} | {row['mean_origin_time_error_s']:.4f} | {row['mean_full_waveform_relative_l2']:.4f} |")
    lines.extend(["", spec["interpretation"], "", result["budget_clarification"]])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
