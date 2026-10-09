"""Generate and evaluate a locked source-transfer localization holdout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/source_transfer_final_blind_protocol.json"
sys.path.insert(0, str(ROOT))

from examples.confirm_fmm_quality_gate import source_registry  # noqa: E402
from examples.run_adaptive_fusion_independent import fmm_estimate  # noqa: E402
from examples.run_source_transfer_moe import (  # noqa: E402
    digest, new_model, read_setup as transfer_setup,
)
from examples.run_traveltime_aligned_moe_localization import (  # noqa: E402
    arrival_scores, choose_diverse_starts, coarse_grid, load_event,
    pick_arrivals, refine, sampled_waveforms, waveform_scores,
)


def locked_setup() -> tuple[dict, dict, dict, list[Path]]:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if digest(Path(__file__).resolve()) != spec["runner_sha256"]:
        raise ValueError("Runner changed after protocol lock")
    for relative, expected in spec["frozen_inputs"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")

    parent, model_spec, localization, target_paths = transfer_setup()
    if (digest(ROOT / "experiments/source_transfer_moe_protocol.json")
            != spec["parent_protocol_sha256"]
            or digest(ROOT / "experiments/traveltime_aligned_moe_localization_protocol.json")
            != spec["localization_protocol_sha256"]):
        raise ValueError("Parent training or localization protocol changed")

    rows, registry_hash = source_registry(
        {"fd": {"output_directory": spec["fd"]["observation_directory"]}})
    audit = spec["source_registry_audit"]
    unique = {tuple(round(value, 3) for value in xyz) for _, xyz in rows}
    if (len(rows) != audit["archive_count"]
            or len(unique) != audit["unique_xyz_count"]
            or registry_hash != audit["manifest_sha256"]):
        raise ValueError("Existing source-truth registry changed after protocol lock")

    proposed = np.asarray([event["source_m_s"][:3] for event in spec["events"]],
                          dtype=np.float64)
    existing = np.asarray([xyz for _, xyz in rows], dtype=np.float64)
    nearest = np.linalg.norm(proposed[:, None, :] - existing[None, :, :], axis=2).min(axis=1)
    if np.any(nearest < audit["minimum_new_to_existing_distance_m"]):
        raise ValueError("A locked test source is too close to an existing source truth")
    pairwise = np.linalg.norm(proposed[:, None, :] - proposed[None, :, :], axis=2)
    pairwise += np.eye(len(proposed)) * 1e9
    if pairwise.min() < audit["minimum_new_to_new_distance_m"]:
        raise ValueError("Locked test sources are not sufficiently separated")
    for model in spec["models"]:
        for seed in spec["seeds"]:
            path = ROOT / model["checkpoint_pattern"].format(seed=seed)
            expected = model["checkpoint_sha256_by_seed"][str(seed)]
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"Frozen checkpoint changed: {path}")
    return spec, model_spec, localization, target_paths


def generate(spec: dict) -> None:
    manifest_path = ROOT / spec["outputs"]["manifest"]
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite generation manifest: {manifest_path}")
    folder = ROOT / spec["fd"]["observation_directory"]
    folder.mkdir(parents=True, exist_ok=True)
    config = ROOT / spec["fd"]["config_file"]
    records = []
    for event in spec["events"]:
        path = folder / event["file"]
        if not path.exists():
            subprocess.run(
                [sys.executable, str(ROOT / spec["fd"]["generator"]),
                 "--config", str(config), "--output", str(path), "--source",
                 *(str(float(value)) for value in event["source_m_s"])],
                cwd=ROOT, check=True,
            )
        with np.load(path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.allclose(archive["source_truth"], event["source_m_s"],
                                       rtol=0, atol=1e-5)):
                raise ValueError(f"Generated record does not match protocol: {path}")
        records.append({"event": event["name"],
                        "file": str(path.relative_to(ROOT)),
                        "source_m_s": event["source_m_s"],
                        "sha256": digest(path)})
        print(f"record ready: {event['name']}", flush=True)
    manifest = {"protocol_sha256": digest(PROTOCOL),
                "runner_sha256": digest(Path(__file__).resolve()),
                "fd_config_sha256": digest(config),
                "records": records}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _score_model(model, event: dict, domain: np.ndarray, bounds: np.ndarray,
                 search: dict, arrival_spec: dict, candidates: np.ndarray,
                 picks: np.ndarray, device: torch.device) -> tuple[dict, float]:
    coarse = sampled_waveforms(event, domain,
                               int(search["coarse_receiver_stride"]),
                               int(search["coarse_time_stride"]), device)
    fine = sampled_waveforms(event, domain,
                             int(search["refine_receiver_stride"]),
                             int(search["refine_time_stride"]), device)
    waveform = waveform_scores(model, candidates, domain, coarse,
                               int(search["candidate_batch"]))
    arrival = arrival_scores(model, candidates, picks,
                             float(arrival_spec["arrival_scale_s"]),
                             int(search["candidate_batch"]))
    alpha = float(arrival_spec["hybrid_weight"])
    total = waveform + alpha * arrival
    indices = choose_diverse_starts(
        candidates, total, int(search["restarts"]),
        float(search["minimum_start_separation_m"]))
    starts = [{"source_m_s": candidates[index].tolist(),
               "coarse_total_loss": float(total[index])} for index in indices]
    best, _ = refine(model, starts, domain, bounds, fine, picks,
                     search, arrival_spec, alpha)
    return best, float(best["total_loss"])


def _bootstrap_mean_ci(values: np.ndarray, replicates: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(replicates, len(values)))
    estimates = values[index].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975))]


def summarize(rows: list[dict], spec: dict) -> dict:
    arms = [model["arm"] for model in spec["models"]]
    source_errors = {arm: {} for arm in arms}
    for arm in arms:
        for event in spec["events"]:
            group = [row["methods"].get(arm, {}).get("location_error_m") for row in rows
                     if row["event"] == event["name"]
                     and row["methods"].get(arm) is not None
                     and row["methods"][arm].get("location_error_m") is not None]
            source_errors[arm][event["name"]] = (
                float(np.median(group)) if group else None)
    fmm_errors = {}
    for event in spec["events"]:
        group = [row["methods"]["fmm"]["location_error_m"] for row in rows
                 if row["event"] == event["name"]
                 and row["methods"]["fmm"]["location_error_m"] is not None]
        fmm_errors[event["name"]] = float(np.median(group)) if group else None
    source_errors["fmm"] = fmm_errors

    summaries = {}
    for arm in arms:
        values = np.asarray([value for value in source_errors[arm].values()
                             if value is not None], dtype=np.float64)
        outcomes = [row["methods"][arm]["location_error_m"] for row in rows
                    if row["methods"].get(arm) is not None
                    and row["methods"][arm]["location_error_m"] is not None]
        summaries[arm] = {
            "source_count_with_estimate": int(len(values)),
            "source_level_median_eloc_m": float(np.median(values)) if len(values) else None,
            "source_level_mean_eloc_m": float(np.mean(values)) if len(values) else None,
            "sources_with_median_error_le_500m": int(np.sum(values <= 500.0)),
            "seed_event_outcomes": len(outcomes),
            "seed_event_median_eloc_m": float(np.median(outcomes)) if outcomes else None,
            "seed_event_successes_le_500m": int(np.sum(np.asarray(outcomes) <= 500.0)),
            "source_level_errors_m": source_errors[arm],
        }
    fmm_values = np.asarray([value for value in fmm_errors.values()
                             if value is not None], dtype=np.float64)
    summaries["fmm"] = {
        "source_count_with_estimate": int(len(fmm_values)),
        "source_level_median_eloc_m": float(np.median(fmm_values)) if len(fmm_values) else None,
        "source_level_mean_eloc_m": float(np.mean(fmm_values)) if len(fmm_values) else None,
        "sources_with_error_le_500m": int(np.sum(fmm_values <= 500.0)),
        "source_level_errors_m": fmm_errors,
    }

    def paired(first: str, second: str, salt: int) -> dict:
        keys = [event["name"] for event in spec["events"]
                if source_errors[first][event["name"]] is not None
                and source_errors[second][event["name"]] is not None]
        delta = np.asarray([source_errors[first][key] - source_errors[second][key]
                            for key in keys], dtype=np.float64)
        return {"first_minus_second_by_source_m": dict(zip(keys, delta.tolist())),
                "paired_source_count": len(delta),
                "mean_delta_m": float(delta.mean()) if len(delta) else None,
                "median_delta_m": float(np.median(delta)) if len(delta) else None,
                "mean_delta_source_bootstrap_95ci_m": (
                    _bootstrap_mean_ci(delta,
                                       int(spec["analysis"]["bootstrap_replicates"]),
                                       int(spec["analysis"]["bootstrap_seed"]) + salt)
                    if len(delta) else None)}

    comparisons = {
        "source_to_target_vs_target_to_target": paired("source_to_target",
                                                         "target_to_target", 0),
        "source_to_target_vs_fmm": paired("source_to_target", "fmm", 1),
        "target_to_target_vs_fmm": paired("target_to_target", "fmm", 2),
    }
    return {"arms": summaries, "paired_source_comparisons": comparisons}


def write_report(result: dict, spec: dict) -> None:
    summary = result["summary"]
    lines = [
        "# Source-transfer final blind localization test",
        "",
        spec["interpretation_scope"],
        "",
        "Primary summary is computed per source by taking the median over three training seeds. Bootstrap intervals resample the 12 source locations, not the 36 seed-event rows.",
        "",
        "| Method | Sources with estimate | Median source-level eloc (m) | Mean source-level eloc (m) | Sources <=500 m |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in [row["arm"] for row in spec["models"]]:
        values = summary["arms"][arm]
        median = values["source_level_median_eloc_m"]
        mean = values["source_level_mean_eloc_m"]
        lines.append(f"| {arm} | {values['source_count_with_estimate']} | "
                     f"{'NA' if median is None else f'{median:.1f}'} | "
                     f"{'NA' if mean is None else f'{mean:.1f}'} | "
                     f"{values['sources_with_median_error_le_500m']}/12 |")
    fmm = summary["arms"]["fmm"]
    fmm_median, fmm_mean = fmm["source_level_median_eloc_m"], fmm["source_level_mean_eloc_m"]
    lines.append(f"| FMM first-arrival baseline | {fmm['source_count_with_estimate']} | "
                 f"{'NA' if fmm_median is None else f'{fmm_median:.1f}'} | "
                 f"{'NA' if fmm_mean is None else f'{fmm_mean:.1f}'} | "
                 f"{fmm['sources_with_error_le_500m']}/12 |")
    lines.extend(["", "## Paired source-level differences", "",
                  "Negative error differences favor the first method.", "",
                  "| Comparison | Sources | Mean difference (m) | Median difference (m) | Source-bootstrap 95% CI for mean (m) |",
                  "|---|---:|---:|---:|---:|"])
    labels = [
        ("source_to_target_vs_target_to_target", "Source-to-target minus target-to-target"),
        ("source_to_target_vs_fmm", "Source-to-target minus FMM"),
        ("target_to_target_vs_fmm", "Target-to-target minus FMM"),
    ]
    for key, label in labels:
        row = summary["paired_source_comparisons"][key]
        ci = row["mean_delta_source_bootstrap_95ci_m"]
        ci_text = "NA" if ci is None else f"[{ci[0]:.1f}, {ci[1]:.1f}]"
        mean = "NA" if row["mean_delta_m"] is None else f"{row['mean_delta_m']:.1f}"
        median = "NA" if row["median_delta_m"] is None else f"{row['median_delta_m']:.1f}"
        lines.append(f"| {label} | {row['paired_source_count']} | {mean} | {median} | {ci_text} |")

    lines.extend(["", "## Per-source results", "",
                  "| Source | FMM eloc (m) | Source-to-target median across seeds (m) | Target-to-target median across seeds (m) | Target-only median across seeds (m) |",
                  "|---|---:|---:|---:|---:|"])
    for event in spec["events"]:
        name = event["name"]
        fmm_error = summary["arms"]["fmm"]["source_level_errors_m"][name]
        values = [summary["arms"][arm]["source_level_errors_m"][name]
                  for arm in ("source_to_target", "target_to_target", "target_only_1500")]
        format_value = lambda value: "NA" if value is None else f"{value:.1f}"
        lines.append(f"| {name} | {format_value(fmm_error)} | "
                     f"{format_value(values[0])} | {format_value(values[1])} | "
                     f"{format_value(values[2])} |")

    test = summary["paired_source_comparisons"]["source_to_target_vs_target_to_target"]
    ci = test["mean_delta_source_bootstrap_95ci_m"]
    improvement = (test["paired_source_count"] == len(spec["events"])
                   and ci is not None and ci[1] < 0
                   and summary["arms"]["source_to_target"]["sources_with_median_error_le_500m"]
                   >= summary["arms"]["target_to_target"]["sources_with_median_error_le_500m"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        ("The registered source-transfer criterion passed: source-to-target has lower paired source-level error with a bootstrap interval below zero and no fewer sources within 500 m than target-to-target."
         if improvement else
         "The registered source-transfer criterion did not pass. Do not claim that source-domain pretraining plus transfer improves localization on this test."),
        "",
        "FMM uses picked first arrivals and the known heterogeneous target velocity. The learned arms use full traces plus the same frozen FMM-arrival term. This is an operational comparison with different information inputs, not a same-input algorithm contest. All records are noise-free synthetic 5-Hz Overthrust data with one receiver geometry; they do not establish field or cross-geology performance.",
        "",
        f"Protocol SHA-256: `{result['protocol_sha256']}`. Observation count: {len(spec['events'])}; independent source count: {len(spec['events'])}; model seeds: {spec['seeds']}.",
    ])
    (ROOT / spec["outputs"]["report"]).write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")


def evaluate(spec: dict, model_spec: dict, localization: dict) -> None:
    manifest_path = ROOT / spec["outputs"]["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError("Run --generate before --evaluate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["protocol_sha256"] != digest(PROTOCOL)
            or manifest["runner_sha256"] != spec["runner_sha256"]
            or len(manifest["records"]) != len(spec["events"])):
        raise ValueError("Generated observations do not match the locked protocol")
    for record in manifest["records"]:
        path = ROOT / record["file"]
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"Observation hash mismatch: {path}")

    result_path = ROOT / spec["outputs"]["result"]
    report_path = ROOT / spec["outputs"]["report"]
    partial_path = ROOT / spec["outputs"]["partial"]
    if result_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite final blind-test results")
    completed = {}
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial["protocol_sha256"] != digest(PROTOCOL):
            raise ValueError("Partial results belong to another protocol")
        completed = {row["key"]: row for row in partial["rows"]}

    receivers = np.asarray(np.load(ROOT / "data/observations/receiver_coords.npy"),
                           dtype=np.float32)
    fields = np.load(ROOT / spec["fmm_fields"], mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = np.asarray([*model_spec["model"]["spatial_lengths_m"],
                         model_spec["model"]["time_length_s"]], dtype=np.float32)
    loc = localization
    bounds = np.asarray(loc["source_bounds_m_s"], dtype=np.float32)
    search, arrival_spec = loc["search"], loc["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    record_by_event = {row["event"]: row for row in manifest["records"]}
    fmm_search = spec["fmm_baseline"]["search"]
    fmm_arrival = spec["fmm_baseline"]["arrival"]

    for model_row in spec["models"]:
        arm = model_row["arm"]
        for seed in spec["seeds"]:
            checkpoint = ROOT / model_row["checkpoint_pattern"].format(seed=seed)
            if digest(checkpoint) != model_row["checkpoint_sha256_by_seed"][str(seed)]:
                raise ValueError(f"Frozen checkpoint hash mismatch: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (payload.get("protocol_sha256") != spec["parent_protocol_sha256"]
                    or payload.get("arm") != arm or int(payload.get("seed")) != seed):
                raise ValueError(f"Checkpoint provenance mismatch: {checkpoint}")
            model = new_model(fields, receivers, model_spec, device)
            model.load_state_dict(payload["state_dict"])
            model.eval().requires_grad_(False)

            for event_spec in spec["events"]:
                key = f"{arm}|{seed}|{event_spec['name']}"
                if key in completed:
                    continue
                record = record_by_event[event_spec["name"]]
                path = ROOT / record["file"]
                observed = load_event(path, receivers)
                raw_picks, corrected_picks = pick_arrivals(observed, arrival_spec)
                valid_picks = int(np.isfinite(corrected_picks).sum())
                started = time.perf_counter()
                learned = None
                if valid_picks >= int(arrival_spec["minimum_valid_picks"]):
                    winner, _ = _score_model(model, observed, domain, bounds,
                                              search, arrival_spec, candidates,
                                              corrected_picks, device)
                    learned = {"estimated_source_m_s": winner["estimated_source_m_s"],
                               "normalized_waveform_mse": winner["waveform_normalized_mse"],
                               "normalized_arrival_mse": winner["arrival_normalized_mse"],
                               "search_loss": winner["total_loss"],
                               "on_boundary": winner["on_boundary"]}
                fmm = fmm_estimate(fields, corrected_picks, bounds,
                                   float(spec["fmm_spacing_m"]), fmm_search,
                                   fmm_arrival)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                search_seconds = time.perf_counter() - started
                # Read the registered truth only after both estimators finish.
                with np.load(path) as archive:
                    truth = np.asarray(archive["source_truth"], dtype=np.float64)
                if learned is None:
                    learned = {"estimated_source_m_s": None}
                methods = {arm: learned, "fmm": fmm}
                for method in methods.values():
                    estimate = method.get("estimated_source_m_s") if method else None
                    if estimate is None:
                        method["location_error_m"] = None
                        method["origin_time_error_s"] = None
                        method["success_500m"] = False
                    else:
                        delta = np.asarray(estimate, dtype=np.float64) - truth
                        method["location_error_m"] = float(np.linalg.norm(delta[:3]))
                        method["origin_time_error_s"] = float(abs(delta[3]))
                        method["success_500m"] = bool(method["location_error_m"] <= 500.0)
                row = {"key": key, "event": event_spec["name"], "seed": seed,
                       "arm": arm, "observation_sha256": record["sha256"],
                       "truth_m_s_post_search": truth.tolist(),
                       "valid_picks": valid_picks,
                       "raw_picks_s": [float(value) if np.isfinite(value) else None
                                       for value in raw_picks],
                       "search_seconds": search_seconds,
                       "methods": methods}
                completed[key] = row
                partial_path.write_text(json.dumps({
                    "protocol_sha256": digest(PROTOCOL), "rows": list(completed.values())
                }, indent=2) + "\n", encoding="utf-8")
                eloc = methods[arm]["location_error_m"]
                print(f"{arm} seed={seed} {event_spec['name']}: eloc={eloc}",
                      flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows = list(completed.values())
    result = {"purpose": spec["purpose"],
              "protocol": str(PROTOCOL.resolve()),
              "protocol_sha256": digest(PROTOCOL),
              "runner_sha256": digest(Path(__file__).resolve()),
              "generation_manifest_sha256": digest(manifest_path),
              "device": str(device), "events": rows,
              "summary": summarize(rows, spec),
              "interpretation_scope": spec["interpretation_scope"]}
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, spec)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(result["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--generate", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    spec, model_spec, localization, _ = locked_setup()
    if args.generate:
        generate(spec)
    else:
        evaluate(spec, model_spec, localization)


if __name__ == "__main__":
    main()
