"""Audit receiver picks and FMM residuals behind the post-hoc FD failure."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.run_traveltime_aligned_moe_localization import pick_arrivals
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator

SOURCE = ROOT / "experiments/fd_joint_arrival_global_audit_results.json"
SOURCE_PROTOCOL = ROOT / "experiments/fd_joint_arrival_global_audit_protocol.json"
RESULT = ROOT / "experiments/fd_arrival_pick_failure_audit.json"
REPORT = RESULT.with_suffix(".md")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(predicted: np.ndarray, corrected: np.ndarray, valid: np.ndarray) -> dict:
    residual = (predicted - corrected)[valid]
    return {
        "median_signed_residual_s": float(np.median(residual)),
        "mean_signed_residual_s": float(np.mean(residual)),
        "rmse_s": float(np.sqrt(np.mean(residual ** 2))),
        "median_abs_residual_s": float(np.median(np.abs(residual))),
        "max_abs_residual_s": float(np.max(np.abs(residual))),
        "count_over_0p1s": int((np.abs(residual) > 0.1).sum()),
    }


def main() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("Arrival-pick audit already exists")
    prior = json.loads(SOURCE.read_text(encoding="utf-8"))
    spec = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    if prior["protocol_sha256"] != digest(SOURCE_PROTOCOL):
        raise ValueError("Joint arrival audit protocol changed")
    base = json.loads((ROOT / spec["base_protocol"]).read_text(encoding="utf-8"))
    picker = json.loads((ROOT / spec["picker_protocol"]).read_text(encoding="utf-8"))
    config = json.loads((ROOT / base["fd_config"]).read_text(encoding="utf-8"))
    field_path = ROOT / spec["fmm_fields"]
    if (digest(field_path) != prior["input_sha256"][spec["fmm_fields"]]
            or digest(ROOT / spec["picker_protocol"])
            != prior["input_sha256"][spec["picker_protocol"]]):
        raise ValueError("FMM or picker protocol changed")
    fields = np.load(field_path, mmap_mode="r")
    spacing = float(json.loads((ROOT / "experiments/traveltime_aligned_moe_development_protocol.json").read_text(
        encoding="utf-8"))["fmm"]["grid_spacing_m"])
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)), spacing)
    expected_receivers = np.load(config["receivers_file"])
    if fields.shape[0] != len(expected_receivers):
        raise ValueError("Field count differs from receivers")
    rows = []
    for previous in prior["events"]:
        name = previous["id"]
        obs_path = ROOT / base["observation_directory"] / f"{name}.npz"
        with np.load(obs_path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        if (not np.array_equal(receivers, expected_receivers)
                or not np.allclose(truth, previous["truth_source_m_s"], atol=1e-5)):
            raise ValueError(f"Event geometry or truth changed: {name}")
        raw, corrected = pick_arrivals({"data": data, "times": times},
                                       picker["arrival_constraint"])
        valid = np.isfinite(corrected)
        if int(valid.sum()) != previous["valid_picks"]:
            raise ValueError(f"Picker valid count changed: {name}")
        selections = {"true": truth}
        for label, weight in (("waveform_only", "0.0"),
                              ("hybrid_0p1", "0.1")):
            selections[label] = np.asarray(
                previous["scores"][weight]["estimated_source_m_s"], dtype=np.float64)
        travels, residual_summaries = {}, {}
        with torch.inference_mode():
            for label, source in selections.items():
                xyz = torch.as_tensor(np.broadcast_to(source[:3],
                                                      (len(receivers), 3)).copy(),
                                      dtype=torch.float32)
                ids = torch.arange(len(receivers))
                travel = interpolator(xyz, ids).squeeze(1).numpy().astype(np.float64)
                travels[label] = travel
                residual_summaries[label] = summarize(source[3] + travel,
                                                       corrected, valid)
        inferred_origin = corrected[valid] - travels["true"][valid]
        peaks = np.max(np.abs(data), axis=1)
        receivers_table = []
        for i in range(len(receivers)):
            receivers_table.append({
                "receiver_index": i, "receiver_xyz_m": receivers[i].tolist(),
                "valid_pick": bool(valid[i]),
                "peak_fraction_of_event_max": float(peaks[i] / peaks.max()),
                "raw_pick_s": float(raw[i]) if np.isfinite(raw[i]) else None,
                "corrected_pick_s": (float(corrected[i]) if valid[i] else None),
                "true_source_fmm_arrival_s": float(truth[3] + travels["true"][i]),
                "true_source_residual_s": (
                    float(truth[3] + travels["true"][i] - corrected[i])
                    if valid[i] else None),
                "waveform_only_residual_s": (
                    float(selections["waveform_only"][3]
                          + travels["waveform_only"][i] - corrected[i])
                    if valid[i] else None),
                "hybrid_0p1_residual_s": (
                    float(selections["hybrid_0p1"][3]
                          + travels["hybrid_0p1"][i] - corrected[i])
                    if valid[i] else None),
            })
        rows.append({
            "id": name, "observation_sha256": digest(obs_path),
            "truth_source_m_s": truth.tolist(), "valid_picks": int(valid.sum()),
            "true_fmm_arrivals_before_end_of_record": int((
                truth[3] + travels["true"] <= times[-1]).sum()),
            "true_fmm_arrivals_before_picker_cutoff": int((
                truth[3] + travels["true"]
                <= times[-1] - picker["arrival_constraint"][
                    "last_record_s_excluded_for_pick"]).sum()),
            "median_origin_from_picks_at_true_xyz_s": float(np.median(inferred_origin)),
            "mean_origin_from_picks_at_true_xyz_s": float(np.mean(inferred_origin)),
            "origin_truth_minus_pick_median_s": float(
                truth[3] - np.median(inferred_origin)),
            "residuals": residual_summaries,
            "receivers": receivers_table,
        })
        print(f"{name}: valid={valid.sum()}, true residual "
              f"RMSE={residual_summaries['true']['rmse_s']:.3f} s, "
              f"median inferred origin={np.median(inferred_origin):.3f} s",
              flush=True)
    result = {"role": "post-hoc per-receiver picker/FMM failure audit",
              "source_result_sha256": digest(SOURCE),
              "source_protocol_sha256": digest(SOURCE_PROTOCOL),
              "field_sha256": digest(field_path),
              "picker_protocol_sha256": digest(ROOT / spec["picker_protocol"]),
              "picker_bias_s": picker["arrival_constraint"]["pick_bias_s"],
              "fmm_spacing_m": spacing, "events": rows,
              "limitation": "Truth is used explicitly in this failure audit. The three events were inspected before; these diagnostics cannot select a weight or validate a new locator."}
    lines = ["# Receiver-pick and FMM failure audit", "",
             result["limitation"], "",
             f"A fixed {result['picker_bias_s']:.4f} s development bias is removed "
             "from raw picks (a negative bias increases corrected pick times). "
             "Residual is predicted onset minus corrected pick; RMSE uses only "
             "receivers passing the original picker filter.", "",
             "| Event | Valid/25 | True arrivals <= picker cutoff | Median pick-inferred origin (s) | Truth origin (s) | True RMSE (s) | Waveform winner RMSE (s) | Hybrid winner RMSE (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        residual = row["residuals"]
        lines.append(f"| {row['id']} | {row['valid_picks']}/25 | "
                     f"{row['true_fmm_arrivals_before_picker_cutoff']} | "
                     f"{row['median_origin_from_picks_at_true_xyz_s']:.3f} | "
                     f"{row['truth_source_m_s'][3]:.3f} | "
                     f"{residual['true']['rmse_s']:.3f} | "
                     f"{residual['waveform_only']['rmse_s']:.3f} | "
                     f"{residual['hybrid_0p1']['rmse_s']:.3f} |")
    lines.extend(["", "The JSON records every receiver's coordinates, peak "
                  "fraction, raw/corrected pick, and signed residual at the "
                  "true and both selected sources. Counts for predicted arrivals "
                  "before the picker cutoff describe geometric visibility; "
                  "they do not independently validate the picker or FMM.", ""])
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {RESULT} and {REPORT}", flush=True)


if __name__ == "__main__":
    main()
