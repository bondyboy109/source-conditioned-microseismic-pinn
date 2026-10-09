"""Run the second locked, source-held-out FMM quality-gate replication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/fmm_quality_gate_replication2_protocol.json"
sys.path.insert(0, str(ROOT))

import examples.confirm_fmm_quality_gate as engine  # noqa: E402


def locked_spec() -> dict:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if engine.sha256(Path(__file__).resolve()) != spec["replication_driver_sha256"]:
        raise ValueError("Replication driver changed after protocol lock")
    for field, hash_field in (("first_confirmation_protocol",
                               "first_confirmation_protocol_sha256"),
                              ("first_confirmation_results",
                               "first_confirmation_results_sha256")):
        path = ROOT / spec[field]
        if not path.is_file() or engine.sha256(path) != spec[hash_field]:
            raise ValueError(f"First confirmation artifact changed: {path}")
    engine.PROTOCOL = PROTOCOL
    return spec


def generate(spec: dict) -> None:
    manifest_path = ROOT / spec["outputs"]["manifest"]
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    engine.generate(spec)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["replication_driver_sha256"] = spec["replication_driver_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_report(result: dict, spec: dict) -> None:
    summary = result["summary"]
    lines = ["# Second locked FMM quality-gate replication", "",
             spec["design"]["limitation"], "", spec["limitations"], "",
             f"Frozen rule: **{summary['gate_rule']}**", "",
             "| Metric | Always accept FMM | Quality gate |", "|---|---:|---:|"]
    baseline_successes = sum(row["success_500m"] for row in result["events"]
                             if row["valid_picks"] in (4, 5)
                             and row["fmm_estimate_m_s"] is not None)
    accepted_successes = sum(row["success_500m"] and row["gate_accepted"]
                             for row in result["events"])
    values = [
        ("Eligible four-to-five-pick cases", summary["eligible_cases"], summary["gate_accepted"]),
        ("Coverage", 1.0 if summary["eligible_cases"] else None, summary["gate_coverage"]),
        ("Success precision", summary["always_accept_precision"], summary["accepted_precision"]),
        ("Successful estimates retained", baseline_successes, accepted_successes),
        ("Median location error (m)", summary["median_location_error_always_accept_m"], summary["median_location_error_accepted_m"]),
    ]
    for name, baseline, gated in values:
        b = "NA" if baseline is None else f"{baseline:.3f}" if isinstance(baseline, float) else str(baseline)
        g = "NA" if gated is None else f"{gated:.3f}" if isinstance(gated, float) else str(gated)
        lines.append(f"| {name} | {b} | {g} |")
    lines.extend(["", "## By SNR", "",
                  "| SNR | Eligible | Accepted | Successes / eligible | Accepted precision |",
                  "|---:|---:|---:|---:|---:|"])
    for snr in [None, *spec["noise"]["snr_db"]]:
        subset = [row for row in result["events"]
                  if row["snr_db"] == snr and row["valid_picks"] in (4, 5)
                  and row["fmm_estimate_m_s"] is not None]
        accepted = [row for row in subset if row["gate_accepted"]]
        precision = (sum(row["success_500m"] for row in accepted) / len(accepted)
                     if accepted else None)
        label = "clean" if snr is None else str(snr)
        p = "NA" if precision is None else f"{precision:.3f}"
        lines.append(f"| {label} | {len(subset)} | {len(accepted)} | "
                     f"{sum(row['success_500m'] for row in subset)}/{len(subset)} | {p} |")
    lines.extend(["", "## By source", "",
                  "Noise conditions from one source are clustered and are not independent source replicates.",
                  "", "| Source | Eligible | Accepted | FMM successes | Accepted successes |",
                  "|---|---:|---:|---:|---:|"])
    for event in spec["events"]:
        subset = [row for row in result["events"] if row["event"] == event["name"]
                  and row["valid_picks"] in (4, 5)
                  and row["fmm_estimate_m_s"] is not None]
        accepted = [row for row in subset if row["gate_accepted"]]
        lines.append(f"| {event['name']} | {len(subset)} | {len(accepted)} | "
                     f"{sum(row['success_500m'] for row in subset)} | "
                     f"{sum(row['success_500m'] for row in accepted)} |")
    lines.extend(["", "The gate abstains on selected estimates; it cannot change the underlying FMM estimate. Interpret conditional precision together with coverage, retained successes, and the number of source locations contributing accepted estimates.",
                  "", f"Total observations evaluated: **{len(result['events'])}**."])
    (ROOT / spec["outputs"]["report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(spec: dict) -> None:
    engine.verify_frozen_inputs(spec)
    manifest_path = ROOT / spec["outputs"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["protocol_sha256"] != engine.sha256(PROTOCOL)
            or manifest["generation_code_sha256"] != spec["confirmation_runner_sha256"]
            or manifest["replication_driver_sha256"] != spec["replication_driver_sha256"]):
        raise ValueError("Generated records do not match the locked replication protocol")
    expected_noise = (len(spec["events"]) * len(spec["noise"]["snr_db"])
                      * len(spec["noise"]["realization_seeds"]))
    if len(manifest["clean_records"]) != len(spec["events"]):
        raise ValueError("Wrong number of clean records")
    if len(manifest["noise_records"]) != expected_noise:
        raise ValueError("Wrong number of noise records")
    for record in manifest["clean_records"] + manifest["noise_records"]:
        path = ROOT / record["file"]
        if not path.is_file() or engine.sha256(path) != record["sha256"]:
            raise ValueError(f"Observation hash mismatch: {path}")

    output, report, partial = (ROOT / spec["outputs"][key]
                               for key in ("result", "report", "partial"))
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite replication results")
    protocol_hash = engine.sha256(PROTOCOL)
    completed = {}
    if partial.exists():
        saved = json.loads(partial.read_text(encoding="utf-8"))
        if saved.get("protocol_sha256") != protocol_hash:
            raise ValueError("Partial result belongs to a different protocol")
        completed = {row["condition"]: row for row in saved["events"]}

    clean_by_event = {row["event"]: row for row in manifest["clean_records"]}
    conditions = []
    for event in spec["events"]:
        clean = clean_by_event[event["name"]]
        conditions.append((event, ROOT / clean["file"], None, None))
    event_by_prefix = {f"qgr2_{index:02d}": event
                       for index, event in enumerate(spec["events"], start=1)}
    for record in manifest["noise_records"]:
        path = ROOT / record["file"]
        event = event_by_prefix[path.stem.split("_snr", 1)[0]]
        realization = int(path.stem.rsplit("_r", 1)[1])
        conditions.append((event, path, int(record["snr_db"]), realization))

    gate = json.loads((ROOT / spec["development_gate_file"]).read_text(encoding="utf-8"))
    receivers = np.asarray(np.load(ROOT / spec["receivers_file"]), dtype=np.float32)
    fields = np.load(ROOT / spec["fmm_cache"], mmap_mode="r")
    if fields.shape != tuple(spec["fmm_cache_shape"]):
        raise ValueError("FMM cache shape mismatch")
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
    result = {"purpose": spec["purpose"], "protocol": str(PROTOCOL.resolve()),
              "protocol_sha256": protocol_hash,
              "replication_driver_sha256": spec["replication_driver_sha256"],
              "confirmation_engine_sha256": spec["confirmation_runner_sha256"],
              "gate_sha256": engine.sha256(ROOT / spec["development_gate_file"]),
              "generation_manifest_sha256": engine.sha256(manifest_path),
              "evaluation_runner_sha256": engine.sha256(Path(__file__).resolve()),
              "fmm_cache_sha256": engine.sha256(ROOT / spec["fmm_cache"]),
              "events": rows, "summary": engine.summarize(rows, gate),
              "limitations": spec["limitations"]}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, spec)
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
