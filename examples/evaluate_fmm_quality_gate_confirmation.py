"""Evaluate generated locked FMM quality-gate records using the frozen protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.confirm_fmm_quality_gate import (  # noqa: E402
    PROTOCOL,
    evaluate_condition,
    sha256,
    summarize,
    verify_frozen_inputs,
    write_report,
)


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    verify_frozen_inputs(spec)
    manifest_path = ROOT / spec["outputs"]["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError("Run the locked confirmation generator first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["protocol_sha256"] != sha256(PROTOCOL)
            or manifest["generation_code_sha256"] != spec["confirmation_runner_sha256"]):
        raise ValueError("Generated records do not match the locked protocol and runner")
    expected_clean = len(spec["events"])
    expected_noisy = (expected_clean * len(spec["noise"]["snr_db"])
                      * len(spec["noise"]["realization_seeds"]))
    if (len(manifest["clean_records"]) != expected_clean
            or len(manifest["noise_records"]) != expected_noisy):
        raise ValueError("Generation manifest does not contain the locked record count")
    for record in manifest["clean_records"] + manifest["noise_records"]:
        path = ROOT / record["file"]
        if not path.is_file() or sha256(path) != record["sha256"]:
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

    gate = json.loads((ROOT / spec["development_gate_file"]).read_text(encoding="utf-8"))
    receivers = np.asarray(np.load(ROOT / spec["receivers_file"]), dtype=np.float32)
    fields = np.load(ROOT / spec["fmm_cache"], mmap_mode="r")
    if fields.shape != tuple(spec["fmm_cache_shape"]):
        raise ValueError("FMM cache shape differs from the locked protocol")
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
              "generation_runner_sha256": manifest["generation_code_sha256"],
              "evaluation_runner": str(Path(__file__).resolve()),
              "evaluation_runner_sha256": sha256(Path(__file__).resolve()),
              "fmm_cache_sha256": sha256(ROOT / spec["fmm_cache"]),
              "source_registry_sha256": spec["source_registry_audit"]["manifest_sha256"],
              "events": rows, "summary": summarize(rows, gate),
              "limitations": spec["limitations"]}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, spec)
    partial.unlink(missing_ok=True)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
