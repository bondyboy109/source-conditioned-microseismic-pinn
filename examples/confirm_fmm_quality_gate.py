"""Generate and evaluate a locked, source-independent sparse-pick FMM gate test."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/fmm_quality_gate_confirmation_protocol.json"
sys.path.insert(0, str(ROOT))

from examples.develop_fmm_quality_gate import quality_features  # noqa: E402
from examples.run_adaptive_fusion_independent import fmm_estimate  # noqa: E402
from examples.run_holdout_noise_picker_comparison import robust_arrival  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_registry(spec: dict) -> tuple[list[tuple[str, list[float]]], str]:
    root = ROOT / "data/observations"
    excluded = (ROOT / spec["fd"]["output_directory"]).resolve()
    rows = []
    for path in sorted(root.rglob("*.npz")):
        if excluded in path.resolve().parents:
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "source_truth" not in archive:
                    continue
                xyz = np.asarray(archive["source_truth"], dtype=np.float64).reshape(-1)[:3]
                rows.append((str(path.relative_to(root)), xyz.tolist()))
        except (OSError, ValueError, EOFError):
            continue
    payload = json.dumps(rows, separators=(",", ":"))
    return rows, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_frozen_inputs(spec: dict) -> None:
    runner = Path(__file__).resolve()
    if sha256(runner) != spec["confirmation_runner_sha256"]:
        raise ValueError("Confirmation runner changed after protocol lock")
    frozen = [
        (spec["development_gate_file"], spec["development_gate_sha256"]),
        (spec["development_results_file"], spec["development_results_sha256"]),
        (spec["development_protocol_file"], spec["development_protocol_sha256"]),
        (spec["velocity_file"], spec["velocity_sha256"]),
        (spec["receivers_file"], spec["receivers_sha256"]),
        (spec["fmm_cache"], spec["fmm_cache_sha256"]),
        (spec["fd"]["generator"], spec["fd"]["generator_sha256"]),
        (spec["fd"]["config_file"], spec["fd"]["config_sha256"]),
    ]
    for relative, expected in frozen:
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    rows, digest = source_registry(spec)
    audit = spec["source_registry_audit"]
    unique = {tuple(round(value, 3) for value in xyz) for _, xyz in rows}
    if (len(rows) != int(audit["archive_count"])
            or len(unique) != int(audit["unique_xyz_count"])
            or digest != audit["manifest_sha256"]):
        raise ValueError("Pre-existing source-truth registry changed after protocol lock")
    candidates = np.asarray([event["source_m_s"][:3] for event in spec["events"]],
                            dtype=np.float64)
    existing = np.asarray([xyz for _, xyz in rows], dtype=np.float64)
    nearest = np.linalg.norm(candidates[:, None, :] - existing[None, :, :], axis=2).min(axis=1)
    if np.any(nearest < float(audit["minimum_new_to_existing_distance_m"])):
        raise ValueError("A locked test source is too close to an archived source truth")
    if len(candidates) > 1:
        pairwise = np.linalg.norm(candidates[:, None, :] - candidates[None, :, :], axis=2)
        pairwise += np.eye(len(candidates)) * 1e9
        if pairwise.min() < float(audit["minimum_new_to_new_distance_m"]):
            raise ValueError("Locked test sources are not sufficiently separated")


def noisy_archive(clean_path: Path, output_path: Path,
                  snr_db: int, seed: int, standardized: np.ndarray) -> dict:
    with np.load(clean_path) as archive:
        signal = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        truth = np.asarray(archive["source_truth"], dtype=np.float32)
    if output_path.exists():
        with np.load(output_path) as saved:
            if (int(saved["noise_snr_db"]) != snr_db
                    or int(saved["noise_seed"]) != seed
                    or not np.allclose(saved["source_truth"], truth, atol=1e-5)):
                raise FileExistsError(f"Existing noise record does not match protocol: {output_path}")
        return {"file": str(output_path.relative_to(ROOT)),
                "sha256": sha256(output_path), "snr_db": int(snr_db),
                "seed": int(seed)}
    rms = np.sqrt(np.mean(signal.astype(np.float64) ** 2, axis=1))
    noise = standardized * (rms / (10.0 ** (snr_db / 20.0))).astype(np.float32)[:, None]
    np.savez_compressed(output_path, data=(signal + noise).astype(np.float32),
                        times=times, receiver_coords=receivers, source_truth=truth,
                        noise_snr_db=np.int32(snr_db), noise_seed=np.int64(seed))
    return {"file": str(output_path.relative_to(ROOT)), "sha256": sha256(output_path),
            "snr_db": int(snr_db), "seed": int(seed)}


def generate(spec: dict) -> None:
    verify_frozen_inputs(spec)
    manifest_path = ROOT / spec["outputs"]["manifest"]
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite generation manifest: {manifest_path}")
    folder = ROOT / spec["fd"]["output_directory"]
    folder.mkdir(parents=True, exist_ok=True)
    clean_records, noise_records = [], []
    generator = ROOT / spec["fd"]["generator"]
    config = ROOT / spec["fd"]["config_file"]
    for event_index, event in enumerate(spec["events"]):
        clean_path = folder / event["file"]
        if not clean_path.exists():
            subprocess.run([sys.executable, str(generator), "--config", str(config),
                            "--output", str(clean_path), "--source",
                            *(str(float(value)) for value in event["source_m_s"])],
                           cwd=ROOT, check=True)
        with np.load(clean_path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.allclose(archive["source_truth"], event["source_m_s"], atol=1e-5)):
                raise ValueError(f"Generated clean record failed validation: {clean_path}")
        clean_records.append({"event": event["name"], "file": str(clean_path.relative_to(ROOT)),
                             "source_m_s": event["source_m_s"], "sha256": sha256(clean_path)})
        for realization, base_seed in enumerate(spec["noise"]["realization_seeds"], start=1):
            seed = int(base_seed + event_index * 1009)
            with np.load(clean_path) as archive:
                shape = archive["data"].shape
                standardized = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
            for snr in spec["noise"]["snr_db"]:
                noisy_path = folder / f"qg_{event_index + 1:02d}_snr{int(snr):02d}_r{realization}.npz"
                noise_records.append(noisy_archive(clean_path, noisy_path, int(snr),
                                                   seed, standardized))
        print(f"generated locked event {event['name']}", flush=True)
    manifest = {"protocol_sha256": sha256(PROTOCOL),
                "generation_code_sha256": sha256(Path(__file__)),
                "fd_config_sha256": sha256(config),
                "clean_records": clean_records, "noise_records": noise_records}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(clean_records)} clean and {len(noise_records)} noisy records.",
          flush=True)


def load_conditions(spec: dict, manifest: dict) -> list[tuple[dict, Path, int | None, int | None]]:
    by_event = {row["event"]: row for row in manifest["clean_records"]}
    conditions = []
    folder = ROOT / spec["fd"]["output_directory"]
    for event in spec["events"]:
        conditions.append((event, folder / by_event[event["name"]]["file"], None, None))
        for snr in spec["noise"]["snr_db"]:
            for realization in range(1, len(spec["noise"]["realization_seeds"]) + 1):
                path = folder / f"qg_{spec['events'].index(event) + 1:02d}_snr{int(snr):02d}_r{realization}.npz"
                conditions.append((event, path, int(snr), realization))
    return conditions


def evaluate_condition(event: dict, path: Path, snr: int | None,
                       realization: int | None, spec: dict,
                       fields: np.ndarray, receivers: np.ndarray,
                       gate: dict) -> dict:
    with np.load(path) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
    picker = spec["picker"]
    raw = np.asarray([robust_arrival(
        trace, times, noise_window_s=float(picker["noise_window_s"]),
        smooth_samples=int(picker["smooth_samples"]),
        sigma_multiplier=float(picker["sigma_multiplier"]),
        consecutive=int(picker["consecutive_samples"])) for trace in data], dtype=np.float64)
    peaks = np.max(np.abs(data), axis=1)
    valid = (np.isfinite(raw)
             & (peaks >= float(picker["minimum_receiver_peak_fraction_of_record"])
                * peaks.max())
             & (raw <= times[-1] - float(picker["last_record_s_excluded_for_pick"])))
    picks = raw - float(picker["pick_bias_s"])
    picks[~valid] = np.nan
    count = int(valid.sum())
    row = {"event": event["name"], "condition": path.stem,
           "snr_db": snr, "realization": realization,
           "valid_picks": count, "truth_used_in_search": False,
           "fmm_estimate_m_s": None, "location_error_m": None,
           "origin_time_error_s": None, "success_500m": False,
           "features": None, "gate_accepted": False}
    if not 4 <= count <= 5:
        return row
    bounds = np.asarray(spec["search"]["source_bounds_m_s"], dtype=np.float64)
    search = {"coarse_grid_xyz": spec["search"]["coarse_grid_xyz"],
              "refinement_grid_xyz": spec["search"]["refinement_grid_xyz"],
              "refinement_levels": spec["search"]["refinement_levels"]}
    arrival = {"minimum_valid_picks_for_fmm_identifiability":
               spec["search"]["minimum_valid_picks"],
               "origin_time_bounds_s": spec["search"]["origin_time_bounds_s"]}
    result = fmm_estimate(fields, picks, bounds, float(spec["fmm_spacing_m"]),
                          search, arrival)
    if result.get("estimated_source_m_s") is None:
        return row
    estimate = np.asarray(result["estimated_source_m_s"], dtype=np.float64)
    features = quality_features(fields, receivers, np.flatnonzero(valid), estimate,
                                result, float(spec["fmm_spacing_m"]))
    with np.load(path) as archive:
        truth = np.asarray(archive["source_truth"], dtype=np.float64)
    delta = estimate - truth
    error = float(np.linalg.norm(delta[:3]))
    value = features.get(gate["feature"])
    accepted = bool(not gate["enabled"] or (value is not None
                    and np.isfinite(value) and float(value) <= float(gate["threshold"])))
    row.update({"fmm_estimate_m_s": estimate.tolist(),
                "arrival_rmse_s": float(result["arrival_rmse_s"]),
                "location_error_m": error,
                "origin_time_error_s": float(abs(delta[3])),
                "success_500m": bool(error <= float(gate["success_threshold_m"])),
                "features": features, "gate_accepted": accepted})
    return row


def summarize(rows: list[dict], gate: dict) -> dict:
    eligible = [row for row in rows if row["valid_picks"] in (4, 5)
                and row["fmm_estimate_m_s"] is not None]
    accepted = [row for row in eligible if row["gate_accepted"]]
    successes = [row for row in eligible if row["success_500m"]]
    retained_successes = [row for row in successes if row["gate_accepted"]]
    return {"eligible_cases": len(eligible),
            "gate_accepted": len(accepted),
            "gate_coverage": len(accepted) / len(eligible) if eligible else None,
            "always_accept_precision": (len(successes) / len(eligible) if eligible else None),
            "accepted_precision": (float(np.mean([row["success_500m"] for row in accepted]))
                                   if accepted else None),
            "all_case_success_rate": len(successes) / len(eligible) if eligible else None,
            "successful_estimates_retained_fraction": (len(retained_successes) / len(successes)
                                                       if successes else None),
            "median_location_error_always_accept_m": (float(np.median(
                [row["location_error_m"] for row in eligible])) if eligible else None),
            "median_location_error_accepted_m": (float(np.median(
                [row["location_error_m"] for row in accepted])) if accepted else None),
            "median_origin_time_error_accepted_s": (float(np.median(
                [row["origin_time_error_s"] for row in accepted])) if accepted else None),
            "gate_rule": gate["rule"]}


def write_report(result: dict, spec: dict) -> None:
    summary = result["summary"]
    lines = ["# Locked FMM quality-gate confirmation", "",
             spec["limitations"], "",
             f"Frozen rule: **{summary['gate_rule']}**", "",
             "| Metric | Always accept FMM | Quality gate |", "|---|---:|---:|"]
    rows = [
        ("Eligible four-to-five-pick cases", summary["eligible_cases"], summary["gate_accepted"]),
        ("Coverage", 1.0 if summary["eligible_cases"] else None, summary["gate_coverage"]),
        ("Success precision among accepted", summary["always_accept_precision"], summary["accepted_precision"]),
        ("Successes retained", summary["eligible_cases"] and sum(r["success_500m"] for r in result["events"] if r["valid_picks"] in (4, 5)),
         sum(r["success_500m"] and r["gate_accepted"] for r in result["events"])),
        ("Median location error (m)", summary["median_location_error_always_accept_m"], summary["median_location_error_accepted_m"]),
    ]
    for name, baseline, gated in rows:
        if isinstance(baseline, float): baseline_text = f"{baseline:.3f}"
        else: baseline_text = str(baseline)
        if isinstance(gated, float): gated_text = f"{gated:.3f}"
        else: gated_text = str(gated)
        lines.append(f"| {name} | {baseline_text} | {gated_text} |")
    lines.extend(["", "## By SNR", "", "| SNR | Eligible | Accepted | Successes / eligible | Precision among accepted |", "|---:|---:|---:|---:|---:|"])
    groups = [("clean", None)] + [(str(snr), snr) for snr in spec["noise"]["snr_db"]]
    for label, snr in groups:
        subset = [r for r in result["events"] if r["valid_picks"] in (4, 5)
                  and r["fmm_estimate_m_s"] is not None and r["snr_db"] == snr]
        accepted = [r for r in subset if r["gate_accepted"]]
        precision = (sum(r["success_500m"] for r in accepted) / len(accepted)
                     if accepted else None)
        lines.append(f"| {label} | {len(subset)} | {len(accepted)} | "
                     f"{sum(r['success_500m'] for r in subset)}/{len(subset)} | "
                     f"{'NA' if precision is None else f'{precision:.3f}'} |")
    lines.extend(["", "## Interpretation", "",
                  "The gate only abstains from selected FMM outputs; it does not modify or improve the FMM source estimates. Read any precision increase together with coverage and the fraction of successful estimates retained.",
                  "", f"Total observations evaluated: **{len(result['events'])}**."])
    (ROOT / spec["outputs"]["report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(spec: dict) -> None:
    verify_frozen_inputs(spec)
    manifest_path = ROOT / spec["outputs"]["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError("Run --generate before --evaluate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("Generated records belong to a different protocol")
    expected_clean = len(spec["events"])
    expected_noisy = (expected_clean * len(spec["noise"]["snr_db"])
                      * len(spec["noise"]["realization_seeds"]))
    if (len(manifest["clean_records"]) != expected_clean
            or len(manifest["noise_records"]) != expected_noisy):
        raise ValueError("Generation manifest does not contain the locked number of records")
    for row in manifest["clean_records"] + manifest["noise_records"]:
        path = ROOT / row["file"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Generated record hash mismatch: {path}")
    output = ROOT / spec["outputs"]["result"]
    report = ROOT / spec["outputs"]["report"]
    partial = ROOT / spec["outputs"]["partial"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite locked confirmation results")
    protocol_hash = sha256(PROTOCOL)
    completed = {}
    if partial.exists():
        saved = json.loads(partial.read_text(encoding="utf-8"))
        if saved.get("protocol_sha256") != protocol_hash:
            raise ValueError("Partial results belong to a different protocol")
        completed = {row["condition"]: row for row in saved["events"]}
    gate = json.loads((ROOT / spec["development_gate_file"]).read_text(encoding="utf-8"))
    receivers = np.asarray(np.load(ROOT / spec["receivers_file"]), dtype=np.float32)
    fields = np.load(ROOT / spec["fmm_cache"], mmap_mode="r")
    if fields.shape != tuple(spec["fmm_cache_shape"]):
        raise ValueError("FMM cache shape differs from the locked protocol")
    conditions = load_conditions(spec, manifest)
    for event, path, snr, realization in conditions:
        if path.stem in completed:
            continue
        row = evaluate_condition(event, path, snr, realization, spec,
                                 fields, receivers, gate)
        completed[row["condition"]] = row
        partial.write_text(json.dumps({"protocol_sha256": protocol_hash,
                                       "events": list(completed.values())}, indent=2) + "\n",
                           encoding="utf-8")
        print(f"{row['condition']}: picks={row['valid_picks']} "
              f"accepted={row['gate_accepted']} "
              f"eloc={row['location_error_m']}", flush=True)
    rows = list(completed.values())
    result = {"purpose": spec["purpose"], "protocol": str(PROTOCOL.resolve()),
              "protocol_sha256": protocol_hash,
              "gate_sha256": sha256(ROOT / spec["development_gate_file"]),
              "generation_manifest_sha256": sha256(manifest_path),
              "fmm_cache_sha256": sha256(ROOT / spec["fmm_cache"]),
              "source_registry_sha256": spec["source_registry_audit"]["manifest_sha256"],
              "events": rows, "summary": summarize(rows, gate),
              "limitations": spec["limitations"]}
    (ROOT / spec["outputs"]["result"]).write_text(json.dumps(result, indent=2) + "\n",
                                                   encoding="utf-8")
    write_report(result, spec)
    partial.unlink(missing_ok=True)
    print(json.dumps(result["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--generate", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if not spec["protocol_written_before_data_generation"]:
        raise ValueError("The test protocol must be locked before generating observations")
    if args.generate:
        generate(spec)
    else:
        evaluate(spec)


if __name__ == "__main__":
    main()
