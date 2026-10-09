"""Run the locked source-held-out confirmation of the arrival-residual gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/fmm_quality_gate_residual_confirmation_protocol.json"
sys.path.insert(0, str(ROOT))

import examples.confirm_fmm_quality_gate as engine  # noqa: E402


def locked_spec() -> dict:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if engine.sha256(Path(__file__).resolve()) != spec["runner_sha256"]:
        raise ValueError("Confirmation runner changed after protocol lock")

    aliases = {
        "confirmation_runner_sha256": "confirmation_engine_sha256",
        "development_gate_file": "gate_file",
        "development_gate_sha256": "gate_sha256",
        "development_results_file": "development_results",
        "development_results_sha256": "development_results_sha256",
        "development_protocol_file": "development_protocol",
        "development_protocol_sha256": "development_protocol_sha256",
    }
    adapted = dict(spec)
    for target, source in aliases.items():
        adapted[target] = spec[source]

    for prior in spec["prior_holdouts"]:
        for key in ("protocol", "results"):
            path_key = key
            hash_key = f"{key}_sha256"
            path = ROOT / prior[path_key]
            if not path.is_file() or engine.sha256(path) != prior[hash_key]:
                raise ValueError(f"Prior holdout artifact changed after protocol lock: {path}")

    engine.PROTOCOL = PROTOCOL
    engine.verify_frozen_inputs(adapted)
    if engine.sha256(ROOT / spec["gate_file"]) != spec["gate_sha256"]:
        raise ValueError("Frozen candidate gate hash mismatch")
    return adapted


def generate(spec: dict) -> None:
    manifest = ROOT / spec["outputs"]["manifest"]
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite generation manifest: {manifest}")
    engine.generate(spec)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["driver_sha256"] = engine.sha256(Path(__file__).resolve())
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def report(result: dict, spec: dict) -> None:
    rows = result["events"]
    summary = result["summary"]
    lines = [
        "# Arrival-residual FMM gate: locked source-held-out follow-up",
        "",
        spec["design"]["post_holdout_adaptation"],
        "",
        spec["limitations"],
        "",
        f"Frozen rule: **{summary['gate_rule']}**",
        "",
        "| Metric | Always accept FMM | Residual gate |",
        "|---|---:|---:|",
    ]
    baseline_successes = sum(row["success_500m"] for row in rows
                             if row["valid_picks"] in (4, 5)
                             and row["fmm_estimate_m_s"] is not None)
    accepted_successes = sum(row["success_500m"] and row["gate_accepted"]
                             for row in rows)
    values = [
        ("Eligible cases", summary["eligible_cases"], summary["gate_accepted"]),
        ("Coverage", 1.0 if summary["eligible_cases"] else None,
         summary["gate_coverage"]),
        ("Success precision", summary["always_accept_precision"],
         summary["accepted_precision"]),
        ("Successful estimates retained", baseline_successes, accepted_successes),
        ("Median location error (m)",
         summary["median_location_error_always_accept_m"],
         summary["median_location_error_accepted_m"]),
    ]
    for label, baseline, gated in values:
        left = "NA" if baseline is None else (
            f"{baseline:.3f}" if isinstance(baseline, float) else str(baseline))
        right = "NA" if gated is None else (
            f"{gated:.3f}" if isinstance(gated, float) else str(gated))
        lines.append(f"| {label} | {left} | {right} |")

    lines.extend([
        "",
        "## By SNR",
        "",
        "| SNR | Eligible | Accepted | FMM successes / eligible | Accepted precision |",
        "|---:|---:|---:|---:|---:|",
    ])
    for snr in [None, *spec["noise"]["snr_db"]]:
        subset = [row for row in rows if row["snr_db"] == snr
                  and row["valid_picks"] in (4, 5)
                  and row["fmm_estimate_m_s"] is not None]
        accepted = [row for row in subset if row["gate_accepted"]]
        precision = (sum(row["success_500m"] for row in accepted) / len(accepted)
                     if accepted else None)
        label = "clean" if snr is None else str(snr)
        precision_text = "NA" if precision is None else f"{precision:.3f}"
        lines.append(f"| {label} | {len(subset)} | {len(accepted)} | "
                     f"{sum(row['success_500m'] for row in subset)}/{len(subset)} | "
                     f"{precision_text} |")

    lines.extend([
        "",
        "## By source",
        "",
        "Noise conditions are repeated measurements nested within each source; the independent source count is 12.",
        "",
        "| Source | Eligible conditions | Accepted | FMM successes | Accepted successes |",
        "|---|---:|---:|---:|---:|",
    ])
    for event in spec["events"]:
        subset = [row for row in rows if row["event"] == event["name"]
                  and row["valid_picks"] in (4, 5)
                  and row["fmm_estimate_m_s"] is not None]
        accepted = [row for row in subset if row["gate_accepted"]]
        lines.append(f"| {event['name']} | {len(subset)} | {len(accepted)} | "
                     f"{sum(row['success_500m'] for row in subset)} | "
                     f"{sum(row['success_500m'] for row in accepted)} |")

    lines.extend([
        "",
        "The threshold was fitted on development data only. This follow-up tests a predeclared alternative feature after the first gate failed on prior holdouts, so it is independent of threshold fitting but not a pristine first-choice preregistration. The gate only abstains; it never changes the FMM estimate. Interpret precision together with coverage and retained successes.",
        "",
        f"Total observations evaluated: **{len(rows)}**; independent source locations: **{len(spec['events'])}**.",
    ])
    (ROOT / spec["outputs"]["report"]).write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")


def evaluate(spec: dict) -> None:
    manifest_path = ROOT / spec["outputs"]["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError("Run --generate before --evaluate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_hash = engine.sha256(PROTOCOL)
    if (manifest.get("protocol_sha256") != protocol_hash
            or manifest.get("generation_code_sha256") != spec["confirmation_runner_sha256"]
            or manifest.get("driver_sha256") != engine.sha256(Path(__file__).resolve())):
        raise ValueError("Generated records do not match the locked protocol and code")

    expected_noisy = (len(spec["events"]) * len(spec["noise"]["snr_db"])
                      * len(spec["noise"]["realization_seeds"]))
    if (len(manifest["clean_records"]) != len(spec["events"])
            or len(manifest["noise_records"]) != expected_noisy):
        raise ValueError("Generation manifest has an unexpected record count")
    for record in manifest["clean_records"] + manifest["noise_records"]:
        path = ROOT / record["file"]
        if not path.is_file() or engine.sha256(path) != record["sha256"]:
            raise ValueError(f"Generated observation hash mismatch: {path}")

    output = ROOT / spec["outputs"]["result"]
    report_path = ROOT / spec["outputs"]["report"]
    partial = ROOT / spec["outputs"]["partial"]
    if output.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite locked confirmation results")
    completed = {}
    if partial.exists():
        saved = json.loads(partial.read_text(encoding="utf-8"))
        if saved.get("protocol_sha256") != protocol_hash:
            raise ValueError("Partial results belong to a different protocol")
        completed = {row["condition"]: row for row in saved["events"]}

    clean_by_event = {row["event"]: row for row in manifest["clean_records"]}
    event_by_prefix = {f"qg_{index:02d}": event
                       for index, event in enumerate(spec["events"], start=1)}
    conditions = []
    for event in spec["events"]:
        clean = clean_by_event[event["name"]]
        conditions.append((event, ROOT / clean["file"], None, None))
    for record in manifest["noise_records"]:
        path = ROOT / record["file"]
        prefix = path.stem.split("_snr", 1)[0]
        event = event_by_prefix[prefix]
        realization = int(path.stem.rsplit("_r", 1)[1])
        conditions.append((event, path, int(record["snr_db"]), realization))

    gate = json.loads((ROOT / spec["gate_file"]).read_text(encoding="utf-8"))
    receivers = np.asarray(np.load(ROOT / spec["receivers_file"]), dtype=np.float32)
    fields = np.load(ROOT / spec["fmm_cache"], mmap_mode="r")
    if fields.shape != tuple(spec["fmm_cache_shape"]):
        raise ValueError("FMM cache shape differs from the locked protocol")
    for event, path, snr, realization in conditions:
        if path.stem in completed:
            continue
        row = engine.evaluate_condition(event, path, snr, realization,
                                        spec, fields, receivers, gate)
        completed[row["condition"]] = row
        partial.write_text(json.dumps({"protocol_sha256": protocol_hash,
                                       "events": list(completed.values())}, indent=2) + "\n",
                           encoding="utf-8")
        print(f"{row['condition']}: picks={row['valid_picks']} "
              f"accepted={row['gate_accepted']} eloc={row['location_error_m']}",
              flush=True)

    rows = list(completed.values())
    result = {
        "purpose": spec["purpose"],
        "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": protocol_hash,
        "runner_sha256": engine.sha256(Path(__file__).resolve()),
        "confirmation_engine_sha256": engine.sha256(ROOT / spec["confirmation_engine"]),
        "gate_sha256": engine.sha256(ROOT / spec["gate_file"]),
        "generation_manifest_sha256": engine.sha256(manifest_path),
        "fmm_cache_sha256": engine.sha256(ROOT / spec["fmm_cache"]),
        "source_registry_sha256": spec["source_registry_audit"]["manifest_sha256"],
        "events": rows,
        "summary": engine.summarize(rows, gate),
        "limitations": spec["limitations"],
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report(result, spec)
    partial.unlink(missing_ok=True)
    print(json.dumps(result["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--generate", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    spec = locked_spec()
    if args.generate:
        generate(spec)
    else:
        evaluate(spec)


if __name__ == "__main__":
    main()
