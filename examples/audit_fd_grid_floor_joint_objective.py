"""Post-hoc truth and nearest-grid FD objective audit on inspected events."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.audit_fd_joint_arrival_development import score_all_weights
from examples.run_fd_waveform_global_refined_confirmation import global_grid, observed_record
from examples.run_traveltime_aligned_moe_localization import pick_arrivals
from fd_waveform import simulate_candidates
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator

PRIOR = ROOT / "experiments/fd_joint_arrival_global_audit_results.json"
PROTOCOL = ROOT / "experiments/fd_joint_arrival_global_audit_protocol.json"
RESULT = ROOT / "experiments/fd_grid_floor_joint_objective_audit.json"
REPORT = RESULT.with_suffix(".md")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("Audit result exists; refusing to overwrite")
    previous = json.loads(PRIOR.read_text(encoding="utf-8"))
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if previous["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Previous protocol changed")
    base_path = ROOT / spec["base_protocol"]
    config_path = ROOT / json.loads(base_path.read_text(encoding="utf-8"))["fd_config"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    picker_path = ROOT / spec["picker_protocol"]
    arrival_spec = json.loads(picker_path.read_text(encoding="utf-8"))["arrival_constraint"]
    field_path = ROOT / spec["fmm_fields"]
    for key, path in ((spec["base_protocol"], base_path),
                      (spec["picker_protocol"], picker_path),
                      (spec["fmm_fields"], field_path)):
        if digest(path) != previous["input_sha256"][key]:
            raise ValueError(f"Input changed: {key}")
    grid = global_grid(np.asarray(config["source_location_bounds_m"]).T,
                       base["spatial_grid_counts"])
    if len(grid) != previous["candidate_count"]:
        raise ValueError("Candidate grid changed")
    receivers = np.load(config["receivers_file"])
    fields = np.load(field_path, mmap_mode="r")
    spacing = json.loads((ROOT / "experiments/traveltime_aligned_moe_development_protocol.json").read_text(
        encoding="utf-8"))["fmm"]["grid_spacing_m"]
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)), float(spacing))
    prior_base = json.loads((ROOT / spec["base_result"]).read_text(encoding="utf-8"))
    if prior_base["protocol_sha256"] != digest(base_path):
        raise ValueError("Earlier FD baseline protocol changed")
    base_events = {item["id"]: item for item in prior_base["events"]}
    records, positions = [], []
    for row in previous["events"]:
        obs_path = ROOT / base["observation_directory"] / f"{row['id']}.npz"
        if digest(obs_path) != base_events[row["id"]]["observation_sha256"]:
            raise ValueError(f"Observation changed: {row['id']}")
        observed, truth = observed_record(obs_path, tuple(base["waveform_window_s"]))
        with np.load(obs_path) as archive:
            _, picks = pick_arrivals({"data": archive["data"],
                                      "times": archive["times"]}, arrival_spec)
            if not np.array_equal(archive["receiver_coords"], receivers):
                raise ValueError("Receiver geometry changed")
        if not np.allclose(truth, row["truth_source_m_s"], atol=1e-5):
            raise ValueError("Truth changed")
        nearest = int(np.argmin(np.linalg.norm(grid - truth[:3], axis=1)))
        records.append((row, obs_path, observed, truth, picks, nearest))
        positions.extend((truth[:3], grid[nearest]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    templates, runtime = simulate_candidates(
        config_path, np.asarray(positions), base["template_duration_s"],
        origin_time_s=base["reference_origin_time_s"], batch_size=2, device=device)
    rows = []
    with torch.inference_mode():
        xyz = torch.as_tensor(np.repeat(np.asarray(positions), len(receivers), axis=0),
                              dtype=torch.float32)
        ids = torch.arange(len(receivers)).repeat(len(positions))
        travels = interpolator(xyz, ids).reshape(len(positions), -1).numpy()
    for i, (old, obs_path, observed, truth, picks, nearest) in enumerate(records):
        evaluated = {}
        for label, offset in (("truth", 0), ("nearest_grid", 1)):
            score = score_all_weights(
                templates[2 * i + offset], observed, runtime["time_step_s"],
                base["reference_origin_time_s"],
                tuple(base["origin_time_bounds_s"]), travels[2 * i + offset],
                picks, spec["arrival_scale_s"], [0.0, 0.1])
            evaluated[label] = {str(weight): value for weight, value in score.items()}
        if (nearest == old["scores"]["0.0"]["candidate_index"]
                and not np.isclose(
                    evaluated["nearest_grid"]["0.0"]["waveform_normalized_mse"],
                    old["scores"]["0.0"]["waveform_normalized_mse"], atol=1e-5)):
            raise ValueError(f"Nearest-grid replay mismatch: {old['id']}")
        rows.append({"id": old["id"], "observation_sha256": digest(obs_path),
                     "truth_source_m_s": truth.tolist(),
                     "nearest_grid_index": nearest,
                     "nearest_grid_xyz_m": grid[nearest].tolist(),
                     "nearest_grid_error_m": float(np.linalg.norm(grid[nearest] - truth[:3])),
                     "scores": evaluated,
                     "prior_winners": {weight: old["scores"][weight]
                                       for weight in ("0.0", "0.1")}})
        print(f"{old['id']}: truth joint={evaluated['truth']['0.1']['total_loss']:.4f}, "
              f"nearest joint={evaluated['nearest_grid']['0.1']['total_loss']:.4f}, "
              f"selected joint={old['scores']['0.1']['total_loss']:.4f}", flush=True)
    result = {"role": "post-hoc truth-versus-grid objective audit",
              "prior_sha256": digest(PRIOR), "protocol_sha256": digest(PROTOCOL),
              "fd_config_sha256": digest(config_path),
              "fd_solver_sha256": digest(ROOT / "fd_waveform.py"),
              "field_sha256": digest(field_path),
              "picker_sha256": digest(picker_path),
              "device": str(device), "template_runtime_seconds": runtime["runtime_seconds"],
              "events": rows,
              "limitation": "Truth enters these comparisons explicitly; all three events were previously inspected. This isolates grid and objective effects and is not an independent localization result."}
    lines = ["# FD truth-versus-nearest-grid objective audit", "", result["limitation"], "",
             "The same frozen corrected picks, FMM fields, FD solver, 0.1 weight, and "
             "origin-time bounds are used at each position. The truth position is "
             "included solely for this diagnostic.", "",
             "| Event | Grid floor (m) | Truth waveform MSE | Truth arrival loss | Truth joint | Nearest waveform MSE | Nearest joint | Selected joint |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        truth = row["scores"]["truth"]["0.1"]
        near = row["scores"]["nearest_grid"]["0.1"]
        selected = row["prior_winners"]["0.1"]
        lines.append(f"| {row['id']} | {row['nearest_grid_error_m']:.1f} | "
                     f"{truth['waveform_normalized_mse']:.5f} | "
                     f"{truth['arrival_normalized_mse']:.4f} | "
                     f"{truth['total_loss']:.4f} | "
                     f"{near['waveform_normalized_mse']:.4f} | "
                     f"{near['total_loss']:.4f} | {selected['total_loss']:.4f} |")
    lines.extend(["", "The JSON also records the optimized origin time, gain, and "
                  "separate losses for both fixed positions and the previous global winners. "
                  "The nearest-grid candidate was part of the original 75-node search; "
                  "the truth position was not.", ""])
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {RESULT} and {REPORT}", flush=True)


if __name__ == "__main__":
    main()
