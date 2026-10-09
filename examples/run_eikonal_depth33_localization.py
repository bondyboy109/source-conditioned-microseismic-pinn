"""Compare 29- and 33-source models on the fixed spent regional search."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_depth33_localization_development_protocol.json"


def check_inputs(spec: dict) -> tuple[list[dict], list[dict]]:
    for name, digest in (("new_checkpoint", "new_sha256"),
                         ("new_development_report", "new_development_sha256"),
                         ("old_checkpoint", "old_sha256"),
                         ("old_localization", "old_localization_sha256"),
                         ("comparison", "comparison_sha256"),
                         ("fmm_baseline", "fmm_baseline_sha256"),
                         ("search_protocol", "search_protocol_sha256")):
        if hash_file(ROOT / spec[name]) != spec[digest]:
            raise ValueError(f"Registered input changed: {spec[name]}")
    new_dev = json.loads((ROOT / spec["new_development_report"]).read_text())
    if new_dev["candidate_checkpoint_sha256"] != spec["new_sha256"]:
        raise ValueError("Candidate checkpoint does not match its depth audit")
    model, payload = load_conditioned_checkpoint(ROOT / spec["new_checkpoint"],
                                                  torch.device("cpu"))
    if model.onset_gate is not None or len(payload["training_record"]["receiver_observations"]) != 33:
        raise ValueError("Candidate has unexpected gate or training record count")
    old = json.loads((ROOT / spec["old_localization"]).read_text())
    comparison = json.loads((ROOT / spec["comparison"]).read_text())
    fmm = json.loads((ROOT / spec["fmm_baseline"]).read_text())
    if not len(old["rows"]) == len(comparison["events"]) == len(fmm["events"]) == 8:
        raise ValueError("Expected eight paired FMM and old-model records")
    trained = {row["sha256"] for row in payload["training_record"]["receiver_observations"]}
    for row, event, arrival in zip(old["rows"], comparison["events"], fmm["events"]):
        if (row["event"] != event["event"] or arrival["sha256"] != event["sha256"]
                or hash_file(Path(event["observation"])) != event["sha256"]
                or event["sha256"] in trained):
            raise ValueError(f"Regional development file changed or in training: {event['event']}")
    print("Verified 8 unchanged, training-disjoint regional development events", flush=True)
    return old["rows"], fmm["events"]


def invoke(*items: Path | str) -> None:
    subprocess.run([sys.executable, *(str(item) for item in items)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    spec = json.loads(PROTOCOL.read_text())
    old, fmm = check_inputs(spec)
    outputs = [ROOT / spec[key] for key in ("derived_checkpoint", "raw_localization_output",
                                             "gated_localization_output", "result")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Regional development output exists; inspect before rerunning")
    if args.preflight:
        return
    invoke(ROOT / "examples/derive_onset_gate_checkpoint.py", ROOT / spec["new_checkpoint"],
           outputs[0], "--protocol", PROTOCOL)
    invoke(ROOT / "examples/audit_eikonal_seeded_localization.py",
           "--checkpoint", ROOT / spec["new_checkpoint"], "--output", outputs[1])
    invoke(ROOT / "examples/audit_eikonal_seeded_localization.py",
           "--checkpoint", outputs[0], "--output", outputs[2])
    raw = json.loads(outputs[1].read_text())
    gated = json.loads(outputs[2].read_text())
    if raw["checkpoint_sha256"] != spec["new_sha256"] or gated["checkpoint_sha256"] != hash_file(outputs[0]):
        raise ValueError("Regional search did not use the registered candidates")
    rows = []
    for original, arrival, ungated, tapered in zip(old, fmm, raw["rows"], gated["rows"]):
        if not (original["event"] == ungated["event"] == tapered["event"]):
            raise ValueError("Regional results are not paired")
        rows.append({"event": original["event"],
                     "fmm_location_m": arrival["location_error_m"],
                     "old29_location_m": original["eikonal_location_error_m"],
                     "new33_raw_location_m": ungated["eikonal_location_error_m"],
                     "new33_gated_location_m": tapered["eikonal_location_error_m"],
                     "old29_origin_error_s": original["eikonal_origin_time_error_s"],
                     "new33_raw_origin_error_s": ungated["eikonal_origin_time_error_s"],
                     "new33_gated_origin_error_s": tapered["eikonal_origin_time_error_s"]})
    medians = {key: float(np.median([row[key] for row in rows]))
               for key in ("fmm_location_m", "old29_location_m",
                           "new33_raw_location_m", "new33_gated_location_m")}
    result = {"role": spec["role"], "protocol_sha256": hash_file(PROTOCOL),
              "new_checkpoint_sha256": spec["new_sha256"],
              "derived_checkpoint_sha256": hash_file(outputs[0]),
              "medians": medians, "rows": rows,
              "limitations": "Uses existing FMM centers, known velocity, previously inspected regional events and a selected development model; not blind or standalone PINN localization"}
    outputs[3].write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Location-error medians (m): {medians}", flush=True)
    print(f"Saved {outputs[3]}", flush=True)


if __name__ == "__main__":
    main()
