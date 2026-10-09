"""Generate registered FD sources and evaluate fixed coarse/fine FMM searches."""

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
from examples.audit_arrival_spatial_resolution import refined_search
from examples.calibrate_arrival_bias_development import summarize
from examples.invert_fmm_receiver_fields import receiver_fields
from examples.validate_arrival_constrained import digest, event_files, load_protocol, pick_receivers, source_search

PROTOCOL = ROOT / "experiments/arrival_spatial_refinement_protocol.json"


def locked_inputs():
    plan = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for key, path in (("development_diagnostic", plan["development_diagnostic"]),
                      ("refinement_implementation", plan["refinement_implementation"]),
                      ("bias_development_result", plan["bias_development_result"]),
                      ("fd_config", "data/observations/experiment_draft.json"),
                      ("fd_implementation", "examples/generate_fd_observation.py"),
                      ("fmm_protocol", "experiments/arrival_constrained_fresh_test_protocol.json")):
        if digest(ROOT / path) != plan[f"{key}_sha256"]:
            raise ValueError(f"Frozen input changed: {path}")
    development = json.loads((ROOT / plan["bias_development_result"]).read_text(encoding="utf-8"))
    diagnostic = json.loads((ROOT / plan["development_diagnostic"]).read_text(encoding="utf-8"))
    if (not development["calibration_adoptable"]
            or development["selected_picker"] != plan["picker"]
            or development["all_development_event_median_bias_s"] != plan["calibration_bias_s"]
            or diagnostic["development_sha256"] != plan["bias_development_result_sha256"]
            or diagnostic["selection"]["top_coarse_seeds"] != plan["refinement"]["top_coarse_seeds"]
            or diagnostic["selection"]["refinement_rounds"] != plan["refinement"]["rounds"]
            or plan["refinement"]["step_divisors"] != [2, 4, 8]):
        raise ValueError("Fixed method differs from development diagnostic")
    spec = load_protocol()
    historical = [np.asarray(item["source_m_s"][:3]) for item in spec["test_events"]]
    previous = json.loads((ROOT / "experiments/arrival_bias_new_source_protocol.json").read_text(encoding="utf-8"))
    historical.extend(np.asarray(item["source_m_s"][:3]) for item in previous["test_events"])
    for path in event_files(spec, "development"):
        with np.load(path) as archive:
            historical.append(np.asarray(archive["source_truth"][:3]))
    registered = [np.asarray(event["source_m_s"][:3]) for event in plan["test_events"]]
    for index, source in enumerate(registered):
        if any(np.array_equal(source, old) for old in historical + registered[:index]):
            raise ValueError(f"Duplicate source: {plan['test_events'][index]['id']}")
    return plan, spec


def generate(plan, spec):
    manifest = ROOT / plan["manifest"]
    if manifest.exists():
        raise FileExistsError(manifest)
    paths = [ROOT / plan["test_directory"] / (event["id"] + ".npz")
             for event in plan["test_events"]]
    events = []
    for event, path in zip(plan["test_events"], paths):
        if path.exists() != path.with_suffix(".json").exists():
            raise ValueError(f"Incomplete pre-existing FD output: {path}")
        if not path.exists():
            print(f"Generating {event['id']}: {event['source_m_s']}", flush=True)
            subprocess.run([sys.executable, "-u", str(ROOT / "examples/generate_fd_observation.py"),
                            "--config", str(ROOT / spec["fd_config"]), "--output", str(path),
                            "--source", *(str(value) for value in event["source_m_s"])],
                           cwd=ROOT, check=True)
        with np.load(path) as archive:
            if not np.allclose(archive["source_truth"], event["source_m_s"], atol=1e-5):
                raise ValueError(f"FD source differs from registration: {path}")
        events.append({"id": event["id"], "source_m_s": event["source_m_s"],
                       "observation_sha256": digest(path),
                       "fd_report_sha256": digest(path.with_suffix(".json"))})
    manifest.write_text(json.dumps({"protocol_sha256": digest(PROTOCOL), "events": events},
                                   indent=2), encoding="utf-8")


def score(estimate, truth, rmse):
    delta = estimate - truth
    return {"status": "ok", "estimated_source_m_s": estimate.tolist(),
            "component_error_m_s": delta.tolist(),
            "location_error_m": float(np.linalg.norm(delta[:3])),
            "origin_time_error_s": float(abs(delta[3])),
            "origin_time_signed_error_s": float(delta[3]),
            "arrival_rmse_s": float(rmse)}


def evaluate(plan, spec):
    output = ROOT / plan["result"]
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((ROOT / plan["manifest"]).read_text(encoding="utf-8"))
    if (manifest["protocol_sha256"] != digest(PROTOCOL)
            or len(manifest["events"]) != len(plan["test_events"])):
        raise ValueError("FD manifest differs from frozen protocol")
    velocity = np.asarray(np.load(ROOT / spec["velocity"], mmap_mode="r")
                          [::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"]],
                          dtype=np.float32)
    receivers = np.load(ROOT / spec["receivers"])
    setup_start = time.perf_counter()
    fields = receiver_fields(velocity, receivers, spec["fmm_spacing_m"])
    setup_seconds = time.perf_counter() - setup_start
    rows = []
    for event, registered in zip(plan["test_events"], manifest["events"]):
        path = ROOT / plan["test_directory"] / (event["id"] + ".npz")
        if (event["id"] != registered["id"] or event["source_m_s"] != registered["source_m_s"]
                or digest(path) != registered["observation_sha256"]
                or digest(path.with_suffix(".json")) != registered["fd_report_sha256"]):
            raise ValueError(f"FD observation differs from manifest: {path}")
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float64)
            if not np.array_equal(archive["receiver_coords"], receivers):
                raise ValueError(f"Receiver geometry differs: {path}")
        picks = pick_receivers(data, times, plan["picker"], spec) - plan["calibration_bias_s"]
        started = time.perf_counter()
        baseline = source_search(fields, picks, spec)
        baseline_seconds = time.perf_counter() - started
        started = time.perf_counter()
        refined = (refined_search(fields, picks, spec,
                                  seeds=plan["refinement"]["top_coarse_seeds"],
                                  rounds=plan["refinement"]["rounds"])
                   if baseline is not None else None)
        refined_seconds = time.perf_counter() - started
        # Read truth only after both independent searches have completed.
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if not np.allclose(truth, event["source_m_s"], atol=1e-5):
            raise ValueError(f"FD source differs from protocol: {path}")
        if baseline is not None:
            candidates, residuals, _ = baseline
            idx = int(np.argmin(residuals))
            coarse_result = score(candidates[idx], truth, residuals[idx])
            refined_result = score(refined[0], truth, refined[1])
        else:
            coarse_result = refined_result = {"status": "insufficient_valid_picks"}
        rows.append({"id": event["id"], "observation_sha256": registered["observation_sha256"],
                     "valid_picks": int(np.isfinite(picks).sum()),
                     "source_truth_m_s_post_search": truth.tolist(),
                     "coarse": coarse_result, "refined": refined_result,
                     "coarse_search_seconds": baseline_seconds,
                     "refined_search_seconds": refined_seconds,
                     "refined_candidate_evaluations": refined[2] if refined else None})
        print(f"{event['id']}: coarse={coarse_result.get('location_error_m')}, "
              f"refined={refined_result.get('location_error_m')}", flush=True)
    coarse, refined = summarize(rows, "coarse"), summarize(rows, "refined")
    rule = plan["acceptance_rule"]
    location_gain = (1 - refined["mean_location_error_m"] / coarse["mean_location_error_m"]
                     if coarse["mean_location_error_m"] and refined["mean_location_error_m"] is not None
                     else None)
    time_increase = (refined["mean_origin_time_error_s"] - coarse["mean_origin_time_error_s"]
                     if refined["mean_origin_time_error_s"] is not None
                     and coarse["mean_origin_time_error_s"] is not None else None)
    accepted = (location_gain is not None and time_increase is not None
                and location_gain >= rule["min_mean_location_error_reduction_fraction"]
                and time_increase <= rule["max_mean_origin_error_increase_s"]
                and refined["failure_count"] - coarse["failure_count"] <= rule["max_new_failures"])
    result = {"protocol_sha256": digest(PROTOCOL),
              "manifest_sha256": digest(ROOT / plan["manifest"]),
              "fmm_setup_seconds": setup_seconds, "coarse": coarse,
              "refined": refined, "relative_location_error_reduction": location_gain,
              "absolute_origin_error_increase_s": time_increase,
              "acceptance_rule": rule, "accepted": bool(accepted),
              "events": rows, "scope": plan["scope"]}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("coarse", "refined", "accepted")}, indent=2),
          flush=True)


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
