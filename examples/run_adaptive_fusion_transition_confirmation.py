"""Generate and evaluate a locked-source pick-transition confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/adaptive_fusion_transition_confirmation_protocol.json"
sys.path.insert(0, str(ROOT))

from examples.run_adaptive_fusion_independent import (  # noqa: E402
    evaluate_condition,
    fd_config,
    make_noisy_archive,
    sha256,
    summarize,
    verify_frozen_inputs,
    write_report,
)
from examples.run_source_conditioned_low_rank_cross_geology import (  # noqa: E402
    load_model_with_fields,
)
from examples.run_traveltime_aligned_moe_localization import (  # noqa: E402
    coarse_grid,
)


def load_spec() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def generate(spec: dict) -> None:
    verify_frozen_inputs(spec)
    calibration = ROOT / spec["development_calibration_result"]
    if sha256(calibration) != spec["development_calibration_result_sha256"]:
        raise ValueError("Development pick calibration changed after SNR selection")
    chosen_snr = json.loads(calibration.read_text(encoding="utf-8"))["confirmation_snr_db"]
    if chosen_snr != spec["noise"]["snr_db"]:
        raise ValueError("Confirmation SNRs do not match the locked calibration result")
    generator = ROOT / spec["fd"]["generator"]
    if sha256(generator) != spec["fd"]["generator_sha256"]:
        raise ValueError("FD observation generator changed")
    folder = ROOT / spec["fd"]["observation_directory"]
    folder.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / spec["fd"]["config_file"]
    expected_config = fd_config(spec)
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != expected_config:
            raise FileExistsError(f"Existing FD config differs: {config_path}")
    else:
        config_path.write_text(json.dumps(expected_config, indent=2) + "\n",
                               encoding="utf-8")

    clean_rows = []
    for event in spec["events"]:
        target = folder / event["file"]
        truth = np.asarray(event["source_m_s"], dtype=np.float32)
        if not target.exists():
            subprocess.run(
                [sys.executable, str(generator), "--config", str(config_path),
                 "--output", str(target), "--source",
                 *(str(float(v)) for v in truth)], cwd=ROOT, check=True)
        with np.load(target) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.allclose(archive["source_truth"], truth, atol=1e-5)):
                raise ValueError(f"Invalid or mismatched clean event: {target}")
        clean_rows.append({"event": event["name"],
                           "file": str(target.relative_to(ROOT)),
                           "sha256": sha256(target),
                           "source_m_s": truth.tolist()})
        print(f"clean transition event ready: {event['name']}", flush=True)

    noise_rows = []
    for event_index, (event, clean_row) in enumerate(zip(spec["events"], clean_rows)):
        clean_path = ROOT / clean_row["file"]
        for snr in spec["noise"]["snr_db"]:
            for realization, base_seed in enumerate(spec["noise"]["realization_seeds"], 1):
                seed = int(base_seed + event_index * 1009)
                filename = (f"{Path(event['file']).stem}_snr{int(snr):02d}_r"
                            f"{realization}.npz")
                noise_rows.append(make_noisy_archive(
                    clean_path, folder / filename, int(snr), seed))
    manifest = {
        "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "calibration_result_sha256": sha256(calibration),
        "fd_config_sha256": sha256(config_path),
        "clean_records": clean_rows,
        "noise_records": noise_rows,
    }
    manifest_path = ROOT / spec["fd"]["generation_manifest"]
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (previous.get("protocol_sha256") != manifest["protocol_sha256"]
                or previous.get("calibration_result_sha256")
                != manifest["calibration_result_sha256"]):
            raise ValueError("Existing data manifest belongs to another protocol")
        for row in previous["clean_records"] + previous["noise_records"]:
            path = ROOT / row["file"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError(f"Generated file hash mismatch: {path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
    print(f"Generated/verified {len(clean_rows)} clean and {len(noise_rows)} noisy records.",
          flush=True)


def append_pick_coverage(report: Path, rows: list[dict]) -> None:
    groups = {"clean": [], "1": [], "2": [], "3": []}
    for row in rows:
        groups["clean" if row["snr_db"] is None else str(row["snr_db"])].append(
            int(row["valid_picks"]))
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\n## Pick-count gate coverage\n\n")
        stream.write("| Condition | 0-3 picks | 4-5 picks | 6-11 picks | 12+ picks |\n")
        stream.write("|---|---:|---:|---:|---:|\n")
        for group, counts in groups.items():
            n = len(counts)
            bins = [sum(x < 4 for x in counts),
                    sum(4 <= x < 6 for x in counts),
                    sum(6 <= x < 12 for x in counts),
                    sum(x >= 12 for x in counts)]
            stream.write(f"| {group} | " + " | ".join(f"{x}/{n}" for x in bins) + " |\n")
        stream.write("\nThe 4-5-pick subset is the prespecified fallback transition band. "
                     "Adaptive estimates in this band are identical to the pure-FMM "
                     "branch by design; the relevant comparison is whether those "
                     "FMM estimates are available and accurate while fixed hybrid "
                     "abstains.\n")


def evaluate(spec: dict) -> None:
    verify_frozen_inputs(spec)
    calibration = ROOT / spec["development_calibration_result"]
    if sha256(calibration) != spec["development_calibration_result_sha256"]:
        raise ValueError("Development calibration result changed")
    manifest_path = ROOT / spec["fd"]["generation_manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError("Run --generate before --evaluate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise ValueError("Generated data do not match the confirmation protocol")
    for row in manifest["clean_records"] + manifest["noise_records"]:
        path = ROOT / row["file"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Observation hash mismatch: {path}")

    output = ROOT / spec["evaluation"]["output_result"]
    report = ROOT / spec["evaluation"]["output_report"]
    partial = ROOT / spec["evaluation"]["partial_results_file"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite transition-confirmation results")
    protocol_hash = sha256(PROTOCOL)
    completed = {}
    if partial.exists():
        saved = json.loads(partial.read_text(encoding="utf-8"))
        if saved.get("protocol_sha256") != protocol_hash:
            raise ValueError("Partial results belong to another confirmation protocol")
        completed = {row["condition"]: row for row in saved["events"]}

    fields = np.load(ROOT / spec["fmm_cache"], mmap_mode="r")
    if fields.shape != (25, 201, 201, 47) or not np.isfinite(fields).all():
        raise ValueError("FMM travel-time cache shape or content mismatch")
    receivers = np.asarray(np.load(ROOT / spec["receivers_file"]), dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain, _ = load_model_with_fields(
        ROOT / spec["checkpoint"], fields, receivers,
        float(spec["fmm_spacing_m"]), device)
    bounds = np.asarray(spec["source_bounds_m_s"], dtype=np.float32)
    candidates = coarse_grid(bounds, spec["search"]["coarse_grid_xyzt"])
    folder = ROOT / spec["fd"]["observation_directory"]
    conditions = []
    for event in spec["events"]:
        clean = folder / event["file"]
        conditions.append((event, clean, clean))
        for snr in spec["noise"]["snr_db"]:
            for realization in range(1, len(spec["noise"]["realization_seeds"]) + 1):
                filename = (f"{Path(event['file']).stem}_snr{int(snr):02d}_r"
                            f"{realization}.npz")
                conditions.append((event, folder / filename, clean))

    started = time.perf_counter()
    for event, observed, truth_path in conditions:
        key = observed.stem
        if key in completed:
            continue
        row = evaluate_condition(model, fields, observed, truth_path, spec,
                                 candidates, domain, bounds, device)
        row["event"] = event["name"]
        completed[key] = row
        partial.write_text(json.dumps({
            "protocol_sha256": protocol_hash,
            "events": list(completed.values()),
        }, indent=2) + "\n", encoding="utf-8")
        adaptive = row["branches"]["adaptive"]["location_error_m"]
        print(f"{key}: picks={row['valid_picks']} decision={row['adaptive_decision']} "
              f"eloc={'NA' if adaptive is None else f'{adaptive:.1f}m'}", flush=True)

    rows = list(completed.values())
    result = {
        "purpose": spec["purpose"],
        "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": protocol_hash,
        "calibration_result_sha256": sha256(calibration),
        "generation_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": sha256(ROOT / spec["checkpoint"]),
        "velocity_sha256": sha256(ROOT / spec["velocity_file"]),
        "receiver_sha256": sha256(ROOT / spec["receivers_file"]),
        "fmm_cache_sha256": sha256(ROOT / spec["fmm_cache"]),
        "evaluation_helper_sha256": sha256(
            ROOT / "examples/run_adaptive_fusion_independent.py"),
        "device": str(device),
        "runtime_seconds": time.perf_counter() - started,
        "events": rows,
        "summary": summarize(rows, spec),
        "limitations": spec["limitations"],
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(report, result, spec)
    append_pick_coverage(report, rows)
    partial.unlink(missing_ok=True)
    print(f"Saved {output} and {report}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--generate", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    spec = load_spec()
    if not spec["protocol_written_before_confirmation_generation"]:
        raise ValueError("Confirmation protocol must be written before generation")
    if args.generate:
        generate(spec)
    else:
        evaluate(spec)


if __name__ == "__main__":
    main()
