"""Score the second locked FMM quality-gate replication without refitting its rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.run_fmm_quality_gate_replication2 import (
    ROOT,
    PROTOCOL,
    engine,
    locked_spec,
    write_report,
)


def main() -> None:
    spec = locked_spec()
    engine.verify_frozen_inputs(spec)
    manifest_path = ROOT / spec["outputs"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["protocol_sha256"] != engine.sha256(PROTOCOL)
            or manifest["generation_code_sha256"] != spec["confirmation_runner_sha256"]
            or manifest["replication_driver_sha256"] != spec["replication_driver_sha256"]):
        raise ValueError("Generated records do not match the locked protocol")
    expected_noise = (len(spec["events"]) * len(spec["noise"]["snr_db"])
                      * len(spec["noise"]["realization_seeds"]))
    if (len(manifest["clean_records"]) != len(spec["events"])
            or len(manifest["noise_records"]) != expected_noise):
        raise ValueError("Generated manifest has the wrong record count")
    for record in manifest["clean_records"] + manifest["noise_records"]:
        path = ROOT / record["file"]
        if not path.is_file() or engine.sha256(path) != record["sha256"]:
            raise ValueError(f"Observation hash mismatch: {path}")

    output, report, partial = (ROOT / spec["outputs"][key]
                               for key in ("result", "report", "partial"))
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite locked replication results")
    protocol_hash = engine.sha256(PROTOCOL)
    completed = {}
    if partial.exists():
        saved = json.loads(partial.read_text(encoding="utf-8"))
        if saved.get("protocol_sha256") != protocol_hash:
            raise ValueError("Partial result belongs to a different protocol")
        completed = {row["condition"]: row for row in saved["events"]}

    event_by_prefix = {f"qg_{index:02d}": event
                       for index, event in enumerate(spec["events"], start=1)}
    conditions = []
    for record in manifest["clean_records"]:
        event = next(item for item in spec["events"] if item["name"] == record["event"])
        conditions.append((event, ROOT / record["file"], None, None))
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


if __name__ == "__main__":
    main()
