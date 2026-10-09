"""Paired local-source ranking development test, without touching frozen pilot models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.diagnose_source_transfer_trajectory import (
    PROTOCOL as TRAJECTORY_PROTOCOL, config, intermediate,
)
from examples.run_source_transfer_moe import digest, new_model
from examples.run_traveltime_aligned_moe_development import (
    load_observations, sample_training_batch,
)
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, choose_diverse_starts, coarse_grid, load_event,
    pick_arrivals, refine, sampled_waveforms, waveform_scores,
)
from examples.invert_fmm_receiver_fields import interpolate_fields

PROTOCOL = ROOT / "experiments/source_transfer_local_rank_protocol.json"


def read_config():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    trajectory, parent, setup, loc, target_paths = config()
    if (ROOT / spec["trajectory_protocol"] != TRAJECTORY_PROTOCOL
            or ROOT / spec["parent_protocol"] != ROOT / trajectory["parent_protocol"]
            or spec["seeds"] != trajectory["seeds"]
            or spec["arms"] != trajectory["arms"]
            or spec["initial_target_updates"] != 500
            or spec["initial_target_updates"] + spec["additional_target_updates"]
            != parent["phase_updates"]
            or spec["branches"] != ["restarted_control", "local_rank"]):
        raise ValueError("Protocol differs from frozen source-transfer trajectory")
    return spec, trajectory, parent, setup, loc, target_paths


def checkpoint_path(spec, arm, seed, branch):
    return ROOT / spec["checkpoint_directory"] / f"{arm}_seed{seed}_{branch}.pth"


def negative_sources(sources, domain, bounds, spec, rng, device):
    count = spec["negative_samples_per_batch"]
    theta = rng.uniform(0, 2 * np.pi, size=count)
    radius = rng.uniform(*spec["negative_horizontal_radius_m"], size=count)
    depth = rng.choice([-1.0, 1.0], size=count) * rng.uniform(
        *spec["negative_vertical_magnitude_m"], size=count)
    offsets = np.stack((radius * np.cos(theta), radius * np.sin(theta), depth), axis=1)
    wrong = sources.detach().clone()
    wrong[:, :3] += torch.as_tensor(offsets / domain[:3],
                                    device=device, dtype=torch.float32)
    low = torch.as_tensor(bounds[:3, 0] / domain[:3], device=device, dtype=torch.float32)
    high = torch.as_tensor(bounds[:3, 1] / domain[:3], device=device, dtype=torch.float32)
    wrong[:, :3] = torch.minimum(torch.maximum(wrong[:, :3], low), high)
    return wrong


def train(spec, trajectory, parent, setup, loc, paths):
    events, receivers, times = load_observations(paths)
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    arrivals = interpolate_fields(fields, np.stack([e["source"][:3] for e in events]), 80.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    bounds = np.asarray(loc["source_bounds_m_s"], dtype=np.float32)
    training = setup["training"]
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            initial_path = intermediate(trajectory, arm, seed,
                                        spec["initial_target_updates"])
            initial_sha = digest(initial_path)
            payload = torch.load(initial_path, map_location="cpu", weights_only=False)
            if (payload["trajectory_protocol_sha256"] != digest(TRAJECTORY_PROTOCOL)
                    or payload["arm"] != arm or payload["seed"] != seed
                    or payload["target_updates"] != spec["initial_target_updates"]):
                raise ValueError(f"Invalid initial checkpoint: {initial_path}")
            for branch in spec["branches"]:
                output = checkpoint_path(spec, arm, seed, branch)
                if output.exists():
                    existing = torch.load(output, map_location="cpu", weights_only=False)
                    if (existing["protocol_sha256"] != digest(PROTOCOL)
                            or existing["initial_checkpoint_sha256"] != initial_sha
                            or existing["arm"] != arm or existing["seed"] != seed
                            or existing["branch"] != branch):
                        raise ValueError(f"Existing checkpoint invalid: {output}")
                    print(f"Preserving {output.name}", flush=True)
                    continue
                model = new_model(fields, receivers, setup, device)
                model.load_state_dict(payload["state_dict"])
                optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
                batch_rng = np.random.default_rng(seed + 50000)
                negative_rng = np.random.default_rng(seed + 60000)
                last = {}
                for step in range(1, spec["additional_target_updates"] + 1):
                    model.train()
                    points, sources, receiver_ids, target = sample_training_batch(
                        events, receivers, times, arrivals, setup, batch_rng, domain, device)
                    optimizer.zero_grad(set_to_none=True)
                    predicted, gates, _ = model(points, sources, receiver_ids)
                    positive = (predicted - target).square().mean()
                    balance = (gates.mean(0) - 1.0 / model.expert_count).square().mean()
                    separation = model.prototype_separation_loss()
                    loss = (positive + training["gate_balance_weight"] * balance
                            + training["prototype_separation_weight"] * separation)
                    rank = torch.zeros((), device=device)
                    if branch == "local_rank":
                        ids = negative_rng.choice(len(points),
                                                  size=spec["negative_samples_per_batch"],
                                                  replace=False)
                        selected = torch.as_tensor(ids, device=device, dtype=torch.long)
                        wrong_sources = negative_sources(sources[selected], domain, bounds,
                                                         spec, negative_rng, device)
                        wrong = model(points[selected], wrong_sources,
                                      receiver_ids[selected])[0]
                        positive_subset = (predicted[selected] - target[selected]).square().mean()
                        negative = (wrong - target[selected]).square().mean()
                        rank = torch.relu(spec["contrastive_margin_mse"]
                                          + positive_subset - negative)
                        loss = loss + spec["contrastive_weight"] * rank
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   training["gradient_clip_norm"])
                    optimizer.step()
                    with torch.no_grad():
                        model.prototype_centers.clamp_(
                            *setup["model"]["prototype_bounds_normalized"])
                    if step % 500 == 0:
                        last = {"positive_mse": float(positive.detach()),
                                "ranking_hinge": float(rank.detach()),
                                "total_loss": float(loss.detach())}
                        print(f"{arm} seed={seed} {branch} step={step}: {last}",
                              flush=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in
                                            model.state_dict().items()},
                            "protocol_sha256": digest(PROTOCOL),
                            "initial_checkpoint_sha256": initial_sha,
                            "target_fmm_sha256": digest(ROOT / parent["target_fmm_fields"]),
                            "arm": arm, "seed": seed, "branch": branch,
                            "last_training_loss": last}, output)
                print(f"Saved {output.name}", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def evaluate(spec, trajectory, parent, setup, loc):
    output = ROOT / spec["result_json"]
    report = ROOT / spec["result_report"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite local-rank results")
    comparisons = []
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            initial_sha = digest(intermediate(trajectory, arm, seed,
                                              spec["initial_target_updates"]))
            states = {}
            for branch in spec["branches"]:
                checkpoint = checkpoint_path(spec, arm, seed, branch)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if (payload["protocol_sha256"] != digest(PROTOCOL)
                        or payload["initial_checkpoint_sha256"] != initial_sha
                        or payload["target_fmm_sha256"] != digest(
                            ROOT / parent["target_fmm_fields"])
                        or payload["arm"] != arm or payload["seed"] != seed
                        or payload["branch"] != branch):
                    raise ValueError(f"Checkpoint invalid: {checkpoint}")
                states[branch] = payload["state_dict"]
            control = states["restarted_control"]
            ranked = states["local_rank"]
            if control.keys() != ranked.keys():
                raise ValueError("Branch model parameters differ")
            max_delta = max((control[k] - ranked[k]).abs().max().item()
                            for k in control)
            comparisons.append({"seed": seed, "arm": arm,
                                "max_parameter_absolute_difference": max_delta})
    if all(row["max_parameter_absolute_difference"] == 0 for row in comparisons):
        result = {"protocol_sha256": digest(PROTOCOL),
                  "development_only": True,
                  "paired_checkpoint_comparisons": comparisons,
                  "all_branch_parameters_identical": True,
                  "localization_skipped": True,
                  "reason": "The local-ranking branch is exactly the same model as "
                            "the paired control in all six comparisons; repeating "
                            "identical localization would add no information.",
                  "restriction": spec["restriction"]}
        lines = ["# Local source-ranking development test", "",
                 spec["comparison"], "",
                 "The proposed auxiliary loss was ineffective: all six paired "
                 "control and ranking models have exactly equal parameters "
                 "(maximum absolute difference = 0). The logged ranking hinge "
                 "was zero at both recorded milestones, 500 and 1000 updates, "
                 "for every seed and training path. Identical models would give "
                 "identical localization predictions, so evaluation was skipped.", "",
                 "| Training path | Seeds | Max parameter difference |",
                 "|---|---:|---:|",
                 "| source_to_target | 4101, 4102, 4103 | 0 |",
                 "| target_to_target | 4101, 4102, 4103 | 0 |", "",
                 "The batch-averaged contrast between randomly perturbed "
                 "training sources is already satisfied. This does not address "
                 "the hard wrong-source minima observed on development events. "
                 "Changing the margin using those events would amount to tuning "
                 "on the diagnostic set. A physically supported next experiment "
                 "would add FD records from neighboring *training* sources "
                 "and check source-dependent waveform behavior before another "
                 "independent localization evaluation.", "", spec["restriction"]]
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return
    names = trajectory["localization_events"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    receivers = np.load(ROOT / "data/observations/receiver_coords.npy").astype(np.float32)
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    bounds = np.asarray(loc["source_bounds_m_s"], dtype=np.float32)
    search, arrival = loc["search"], loc["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    observations = {}
    for name in names:
        path = ROOT / name
        observation = load_event(path, receivers)
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if truth.shape != (4,) or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1]):
            raise ValueError(f"Invalid true source: {path}")
        _, picks = pick_arrivals(observation, arrival)
        if np.isfinite(picks).sum() < arrival["minimum_valid_picks"]:
            raise ValueError(f"Insufficient arrival picks: {path}")
        coarse = sampled_waveforms(observation, domain,
                                   search["coarse_receiver_stride"],
                                   search["coarse_time_stride"], device)
        fine = sampled_waveforms(observation, domain,
                                 search["refine_receiver_stride"],
                                 search["refine_time_stride"], device)
        observations[name] = (truth, picks, coarse, fine)

    rows = []
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            initial_sha = digest(intermediate(trajectory, arm, seed,
                                              spec["initial_target_updates"]))
            for branch in spec["branches"]:
                checkpoint = checkpoint_path(spec, arm, seed, branch)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if (payload["protocol_sha256"] != digest(PROTOCOL)
                        or payload["initial_checkpoint_sha256"] != initial_sha
                        or payload["target_fmm_sha256"] != digest(
                            ROOT / parent["target_fmm_fields"])
                        or payload["arm"] != arm or payload["seed"] != seed
                        or payload["branch"] != branch):
                    raise ValueError(f"Checkpoint invalid: {checkpoint}")
                model = new_model(fields, receivers, setup, device)
                model.load_state_dict(payload["state_dict"])
                model.eval().requires_grad_(False)
                for name in names:
                    truth, picks, coarse, fine = observations[name]
                    wave = waveform_scores(model, candidates, domain, coarse,
                                           search["candidate_batch"])
                    travel = arrival_scores(model, candidates, picks,
                                            arrival["arrival_scale_s"],
                                            search["candidate_batch"])
                    score = wave + arrival["hybrid_weight"] * travel
                    indices = choose_diverse_starts(
                        candidates, score, search["restarts"],
                        search["minimum_start_separation_m"])
                    starts = [{"source_m_s": candidates[i].tolist(),
                               "coarse_total_loss": float(score[i])} for i in indices]
                    best, _ = refine(model, starts, domain, bounds, fine, picks,
                                     search, arrival, arrival["hybrid_weight"])
                    estimate = np.asarray(best["estimated_source_m_s"], dtype=np.float64)
                    pair = np.stack((truth, estimate)).astype(np.float32)
                    fine_wave = waveform_scores(model, pair, domain, fine, 2)
                    fine_arrival = arrival_scores(model, pair, picks,
                                                  arrival["arrival_scale_s"], 2)
                    gap = float(fine_wave[1] - fine_wave[0]
                                + arrival["hybrid_weight"]
                                * (fine_arrival[1] - fine_arrival[0]))
                    rows.append({"seed": seed, "arm": arm, "branch": branch,
                                 "event": name, "truth_m_s": truth.tolist(),
                                 "estimate_m_s": estimate.tolist(),
                                 "location_error_m": float(np.linalg.norm(
                                     estimate[:3] - truth[:3])),
                                 "origin_time_error_s": float(abs(estimate[3] - truth[3])),
                                 "fine_waveform_at_truth": float(fine_wave[0]),
                                 "fine_waveform_at_estimate": float(fine_wave[1]),
                                 "arrival_at_truth": float(fine_arrival[0]),
                                 "hybrid_estimate_minus_truth": gap})
                    print(f"{seed} {arm} {branch} {Path(name).parent.name}/"
                          f"{Path(name).name}: eloc={rows[-1]['location_error_m']:.1f} m, "
                          f"objective_gap={gap:+.4f}", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summaries, pairs = {}, []
    for arm in spec["arms"]:
        for branch in spec["branches"]:
            group = [r for r in rows if r["arm"] == arm and r["branch"] == branch]
            summaries[f"{arm}_{branch}"] = {
                "median_location_error_m": float(np.median([
                    r["location_error_m"] for r in group])),
                "mean_location_error_m": float(np.mean([
                    r["location_error_m"] for r in group])),
                "mean_fine_waveform_at_truth": float(np.mean([
                    r["fine_waveform_at_truth"] for r in group])),
                "wrong_estimate_has_lower_hybrid_objective": sum(
                    r["hybrid_estimate_minus_truth"] < 0 for r in group),
            }
        control = {(r["seed"], r["event"]): r for r in rows
                   if r["arm"] == arm and r["branch"] == "restarted_control"}
        modified = {(r["seed"], r["event"]): r for r in rows
                    if r["arm"] == arm and r["branch"] == "local_rank"}
        if control.keys() != modified.keys() or len(control) != 12:
            raise ValueError("Missing paired outcomes")
        for key in sorted(control):
            pairs.append({"arm": arm, "seed": key[0], "event": key[1],
                          "rank_minus_control_location_error_m":
                          modified[key]["location_error_m"] - control[key]["location_error_m"],
                          "rank_minus_control_truth_waveform_mse":
                          modified[key]["fine_waveform_at_truth"]
                          - control[key]["fine_waveform_at_truth"]})
        group = [p for p in pairs if p["arm"] == arm]
        summaries[f"{arm}_paired"] = {
            "ranking_improves_location": sum(
                p["rank_minus_control_location_error_m"] < 0 for p in group),
            "mean_rank_minus_control_location_m": float(np.mean([
                p["rank_minus_control_location_error_m"] for p in group])),
            "median_rank_minus_control_location_m": float(np.median([
                p["rank_minus_control_location_error_m"] for p in group])),
        }
    result = {"protocol_sha256": digest(PROTOCOL),
              "target_fmm_sha256": digest(ROOT / parent["target_fmm_fields"]),
              "device": str(device), "development_only": True,
              "summaries": summaries, "paired_differences": pairs,
              "rows": rows, "restriction": spec["restriction"]}
    lines = ["# Local source-ranking development test", "",
             spec["comparison"], "",
             "| Training path | Branch | Median location error (m) | "
             "Mean location error (m) | Mean fine waveform MSE at truth | "
             "Wrong estimate wins hybrid objective |",
             "|---|---|---:|---:|---:|---:|"]
    for arm in spec["arms"]:
        for branch in spec["branches"]:
            row = summaries[f"{arm}_{branch}"]
            lines.append(f"| {arm} | {branch} | "
                         f"{row['median_location_error_m']:.1f} | "
                         f"{row['mean_location_error_m']:.1f} | "
                         f"{row['mean_fine_waveform_at_truth']:.4f} | "
                         f"{row['wrong_estimate_has_lower_hybrid_objective']}/12 |")
    lines.extend(["", "Paired local-rank minus restarted-control location error "
                  "(negative favors the auxiliary loss):", "",
                  "| Training path | Ranking improves | Mean difference (m) | "
                  "Median difference (m) |", "|---|---:|---:|---:|"])
    for arm in spec["arms"]:
        row = summaries[f"{arm}_paired"]
        lines.append(f"| {arm} | {row['ranking_improves_location']}/12 | "
                     f"{row['mean_rank_minus_control_location_m']:+.1f} | "
                     f"{row['median_rank_minus_control_location_m']:+.1f} |")
    lines.extend(["", spec["restriction"], "",
                  "Each set of twelve outcomes is three paired seeds times four "
                  "development source positions in one synthetic geology; neither "
                  "independent confirmation nor physical waveform validation at "
                  "the artificially perturbed negative sources is implied."])
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "evaluate"))
    args = parser.parse_args()
    spec, trajectory, parent, setup, loc, paths = read_config()
    if args.stage == "train":
        train(spec, trajectory, parent, setup, loc, paths)
    else:
        evaluate(spec, trajectory, parent, setup, loc)


if __name__ == "__main__":
    main()
