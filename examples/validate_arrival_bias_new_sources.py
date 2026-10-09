"""Generate registered FD records, then evaluate a frozen picker bias once."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.calibrate_arrival_bias_development import summarize
from examples.invert_fmm_receiver_fields import receiver_fields
from examples.validate_arrival_constrained import digest, load_protocol, pick_receivers, source_search

PROTOCOL = ROOT / "experiments/arrival_bias_new_source_protocol.json"


def locked_inputs():
    plan = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    decision = ROOT / plan["development_result"]
    if digest(decision) != plan["development_result_sha256"]:
        raise ValueError("Development calibration changed after registration")
    development = json.loads(decision.read_text(encoding="utf-8"))
    if (not development["calibration_adoptable"]
            or development["selected_picker"] != plan["picker"]
            or development["all_development_event_median_bias_s"] != plan["bias_s"]):
        raise ValueError("Frozen bias differs from the development decision")
    spec = load_protocol()
    if digest(ROOT / spec["fd_config"]) != plan["fd_config_sha256"]:
        raise ValueError("FD generator configuration changed")
    if digest(ROOT / "experiments/arrival_constrained_fresh_test_protocol.json") != plan["source_protocol_sha256"]:
        raise ValueError("Registered FMM search configuration changed")
    historical = [np.asarray(event["source_m_s"][:3]) for event in spec["test_events"]]
    for group in spec["development_groups"]:
        for path in sorted((ROOT / group).glob("test_*.npz")):
            with np.load(path) as observation:
                historical.append(np.asarray(observation["source_truth"][:3]))
    registered = [np.asarray(event["source_m_s"][:3]) for event in plan["test_events"]]
    for index, source in enumerate(registered):
        if any(np.array_equal(source, old) for old in historical + registered[:index]):
            raise ValueError(f"Duplicate source position for {plan['test_events'][index]['id']}")
    return plan, spec


def generate(plan, spec):
    destination = ROOT / plan["manifest"]
    if destination.exists():
        raise FileExistsError(destination)
    paths = [ROOT / plan["test_directory"] / (event["id"] + ".npz")
             for event in plan["test_events"]]
    if any(path.exists() or path.with_suffix(".json").exists() for path in paths):
        raise FileExistsError("New event path already exists; refusing to overwrite")
    records = []
    for event, path in zip(plan["test_events"], paths):
        print(f"Generating {event['id']}: {event['source_m_s']}", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "examples/generate_fd_observation.py"),
                        "--config", str(ROOT / spec["fd_config"]), "--output", str(path),
                        "--source", *(str(x) for x in event["source_m_s"])],
                       cwd=ROOT, check=True)
        with np.load(path) as observation:
            if not np.allclose(observation["source_truth"], event["source_m_s"], atol=1e-5):
                raise ValueError(f"FD source differs from registration: {path}")
        records.append({"id": event["id"], "source_m_s": event["source_m_s"],
                        "observation_sha256": digest(path),
                        "fd_report_sha256": digest(path.with_suffix(".json"))})
    destination.write_text(json.dumps({"protocol_sha256": digest(PROTOCOL),
                                       "events": records}, indent=2), encoding="utf-8")


def evaluate(plan, spec):
    output = ROOT / plan["result"]
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((ROOT / plan["manifest"]).read_text(encoding="utf-8"))
    if (manifest["protocol_sha256"] != digest(PROTOCOL)
            or len(manifest["events"]) != len(plan["test_events"])):
        raise ValueError("FD manifest does not match locked protocol")
    velocity = np.asarray(np.load(ROOT / spec["velocity"], mmap_mode="r")
                          [::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"]],
                          dtype=np.float32)
    registered_receivers = np.load(ROOT / spec["receivers"])
    setup = time.perf_counter()
    fields = receiver_fields(velocity, registered_receivers, spec["fmm_spacing_m"])
    setup_seconds = time.perf_counter() - setup
    rows = []
    for event, record in zip(plan["test_events"], manifest["events"]):
        path = ROOT / plan["test_directory"] / (event["id"] + ".npz")
        if (event["id"] != record["id"] or event["source_m_s"] != record["source_m_s"]
                or digest(path) != record["observation_sha256"]
                or digest(path.with_suffix(".json")) != record["fd_report_sha256"]):
            raise ValueError(f"Observation differs from FD manifest: {path}")
        with np.load(path) as observation:
            data = np.asarray(observation["data"], dtype=np.float32)
            times = np.asarray(observation["times"], dtype=np.float64)
            receivers = np.asarray(observation["receiver_coords"])
        if not np.array_equal(receivers, registered_receivers):
            raise ValueError(f"Receiver geometry changed: {path}")
        start = time.perf_counter()
        picks = pick_receivers(data, times, plan["picker"], spec)
        # The only difference between branches is the frozen development bias.
        raw_search = source_search(fields, picks, spec)
        corrected_picks = picks - plan["bias_s"]
        corrected_search = source_search(fields, corrected_picks, spec)
        search_seconds = time.perf_counter() - start
        with np.load(path) as observation:
            truth = np.asarray(observation["source_truth"], dtype=np.float64)
        if not np.allclose(truth, event["source_m_s"], atol=1e-5):
            raise ValueError(f"Test source changed: {path}")
        # Score truth only after the observed waveforms have been searched.
        def scored(search):
            if search is None:
                return {"status": "insufficient_valid_picks"}
            sources, rmse, _ = search
            estimate = sources[int(np.argmin(rmse))]
            delta = estimate - truth
            return {"status": "ok", "estimated_source_m_s": estimate.tolist(),
                    "component_error_m_s": delta.tolist(),
                    "location_error_m": float(np.linalg.norm(delta[:3])),
                    "origin_time_error_s": float(abs(delta[3])),
                    "origin_time_signed_error_s": float(delta[3]),
                    "arrival_rmse_s": float(np.min(rmse))}
        row = {"id": event["id"], "observation_sha256": record["observation_sha256"],
               "valid_picks": int(np.isfinite(picks).sum()),
               "search_seconds": search_seconds,
               "source_truth_m_s_post_search": truth.tolist(),
               "raw": scored(raw_search), "calibrated": scored(corrected_search)}
        rows.append(row)
        print(f"{event['id']}: raw={row['raw'].get('origin_time_error_s')}, "
              f"calibrated={row['calibrated'].get('origin_time_error_s')}", flush=True)
    raw, corrected = summarize(rows, "raw"), summarize(rows, "calibrated")
    rule = plan["acceptance_rule"]
    time_gain = (1.0 - corrected["mean_origin_time_error_s"] / raw["mean_origin_time_error_s"]
                 if raw["mean_origin_time_error_s"] else None)
    spatial_increase = (corrected["mean_location_error_m"] / raw["mean_location_error_m"] - 1.0
                        if raw["mean_location_error_m"] else None)
    accepted = (time_gain is not None and spatial_increase is not None
                and time_gain >= rule["min_mean_origin_error_reduction_fraction"]
                and spatial_increase <= rule["max_mean_location_error_increase_fraction"]
                and corrected["failure_count"] - raw["failure_count"] <= rule["max_new_failures"])
    output.write_text(json.dumps({"protocol_sha256": digest(PROTOCOL),
                                  "manifest_sha256": digest(ROOT / plan["manifest"]),
                                  "bias_s": plan["bias_s"], "fmm_setup_seconds": setup_seconds,
                                  "raw": raw, "calibrated": corrected,
                                  "relative_origin_error_reduction": time_gain,
                                  "relative_location_error_increase": spatial_increase,
                                  "acceptance_rule": rule, "accepted": bool(accepted),
                                  "events": rows, "scope": plan["scope"]}, indent=2),
                      encoding="utf-8")
    print(f"Saved {output}; accepted={accepted}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("generate", "evaluate"))
    args = parser.parse_args()
    plan, spec = locked_inputs()
    if args.stage == "generate":
        generate(plan, spec)
    else:
        evaluate(plan, spec)


if __name__ == "__main__":
    main()
