"""Develop a source-held-out quality gate for sparse-pick FMM estimates."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/fmm_quality_gate_development_protocol.json"
sys.path.insert(0, str(ROOT))

from examples.invert_fmm_receiver_fields import interpolate_fields  # noqa: E402
from examples.run_adaptive_fusion_independent import (  # noqa: E402
    fmm_estimate,
    sha256,
)
from examples.run_holdout_noise_picker_comparison import robust_arrival  # noqa: E402


FEATURES = ["arrival_rmse_s", "max_azimuth_gap_deg",
            "log10_directional_condition_number", "log10_directional_gdop"]


def quality_features(fields: np.ndarray, receivers: np.ndarray,
                     valid_ids: np.ndarray, estimate: np.ndarray,
                     fmm_result: dict, spacing: float) -> dict:
    xyz = np.asarray(estimate[:3], dtype=np.float64)
    selected_receivers = receivers[valid_ids].astype(np.float64)
    angles = np.mod(np.arctan2(selected_receivers[:, 1] - xyz[1],
                               selected_receivers[:, 0] - xyz[0]), 2 * np.pi)
    gaps = np.diff(np.r_[np.sort(angles), np.sort(angles)[0] + 2 * np.pi])
    feature = {"arrival_rmse_s": float(fmm_result["arrival_rmse_s"]),
               "max_azimuth_gap_deg": float(np.degrees(gaps.max())),
               "azimuth_sectors_8": int(np.unique(
                   np.floor(angles / (2 * np.pi / 8)).astype(int)).size)}

    receiver_fields = fields[valid_ids]
    axes = np.eye(3, dtype=np.float64) * float(spacing)
    gradients = []
    for axis in axes:
        plus = interpolate_fields(receiver_fields, (xyz + axis)[None, :], spacing)[0]
        minus = interpolate_fields(receiver_fields, (xyz - axis)[None, :], spacing)[0]
        gradients.append((plus - minus) / (2.0 * float(spacing)))
    gradient_matrix = np.stack(gradients, axis=1)
    norms = np.linalg.norm(gradient_matrix, axis=1)
    usable = np.isfinite(norms) & (norms > 1e-12)
    direction = gradient_matrix[usable] / norms[usable, None]
    if len(direction) < 4:
        condition, gdop = float("inf"), float("inf")
    else:
        centered = direction - direction.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        if len(singular) < 3 or singular[-1] <= 1e-9:
            condition, gdop = float("inf"), float("inf")
        else:
            condition = float(singular[0] / singular[-1])
            gdop = float(np.sqrt(np.sum(1.0 / np.square(singular))))
    feature["directional_condition_number"] = condition
    feature["directional_gdop"] = gdop
    feature["log10_directional_condition_number"] = (
        float(np.log10(condition)) if np.isfinite(condition) else None)
    feature["log10_directional_gdop"] = (
        float(np.log10(gdop)) if np.isfinite(gdop) else None)
    return feature


def threshold_for(rows: list[dict], feature: str,
                  min_coverage: float = 0.5) -> dict | None:
    finite = [row for row in rows if row["features"].get(feature) is not None
              and np.isfinite(row["features"][feature])]
    if not finite:
        return None
    candidates = sorted({float(row["features"][feature]) for row in finite})
    best = None
    for threshold in candidates:
        accepted = [row for row in finite
                    if float(row["features"][feature]) <= threshold]
        coverage = len(accepted) / len(rows)
        if coverage + 1e-12 < min_coverage:
            continue
        precision = float(np.mean([row["success_500m"] for row in accepted]))
        candidate = {"feature": feature, "threshold": threshold,
                     "n_accepted": len(accepted), "coverage": coverage,
                     "precision": precision}
        if (best is None or candidate["precision"] > best["precision"]
                or (candidate["precision"] == best["precision"]
                    and candidate["coverage"] > best["coverage"])
                or (candidate["precision"] == best["precision"]
                    and candidate["coverage"] == best["coverage"]
                    and candidate["threshold"] < best["threshold"])):
            best = candidate
    return best


def grouped_cross_validation(rows: list[dict], feature: str) -> dict:
    accepted_rows, fold_details = [], []
    sources = sorted({row["event"] for row in rows})
    for source in sources:
        train = [row for row in rows if row["event"] != source]
        test = [row for row in rows if row["event"] == source]
        fitted = threshold_for(train, feature, min_coverage=0.5)
        if fitted is None:
            fold_details.append({"held_out_source": source,
                                 "threshold": None, "n_test": len(test),
                                 "n_accepted": 0})
            continue
        accepted = [row for row in test
                    if row["features"].get(feature) is not None
                    and np.isfinite(row["features"][feature])
                    and float(row["features"][feature]) <= fitted["threshold"]]
        accepted_rows.extend(accepted)
        fold_details.append({"held_out_source": source,
                             "threshold": fitted["threshold"],
                             "n_test": len(test), "n_accepted": len(accepted),
                             "accepted_successes": sum(r["success_500m"] for r in accepted)})
    n_total = len(rows)
    n_accepted = len(accepted_rows)
    return {"n_total": n_total, "n_accepted": n_accepted,
            "coverage": n_accepted / n_total if n_total else 0.0,
            "precision": float(np.mean([row["success_500m"]
                                        for row in accepted_rows]))
            if accepted_rows else 0.0,
            "details": fold_details}


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    outputs = [ROOT / spec["outputs"][name] for name in ("result", "report", "selected_gate")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite FMM quality-gate development outputs")
    event_protocol = ROOT / spec["source_event_protocol"]
    if sha256(event_protocol) != spec["source_event_protocol_sha256"]:
        raise ValueError("Development event registry changed")
    source_spec = json.loads(event_protocol.read_text(encoding="utf-8"))
    folder = ROOT / source_spec["observation_directory"]
    receiver_path = ROOT / spec["receivers_file"]
    receivers = np.asarray(np.load(receiver_path), dtype=np.float32)
    fields = np.load(ROOT / spec["fmm_cache"], mmap_mode="r")
    if fields.shape != (25, 201, 201, 47):
        raise ValueError("FMM cache has unexpected shape")
    rows = []
    picker = spec["picker"]
    bounds = np.asarray(spec["source_bounds_m_s"], dtype=np.float32)
    search = spec["search"]
    spacing = float(spec["fmm_spacing_m"])
    for event_index, event in enumerate(source_spec["events"]):
        path = folder / event["file"]
        if sha256(path) != event["sha256"]:
            raise ValueError(f"Development event changed: {path}")
        with np.load(path) as archive:
            clean = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers_from_file = np.asarray(archive["receiver_coords"], dtype=np.float32)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        keep = times <= float(spec["model_window_s"]) + 1e-7
        clean, times = clean[:, keep], times[keep]
        if not np.array_equal(receivers_from_file, receivers):
            raise ValueError(f"Receiver geometry mismatch in {path}")
        rms = np.sqrt(np.mean(clean.astype(np.float64) ** 2, axis=1))
        for realization, seed_base in enumerate(spec["realization_seeds"], start=1):
            seed = int(seed_base + event_index * 1009)
            standardized = np.random.default_rng(seed).standard_normal(
                clean.shape).astype(np.float32)
            for snr in spec["snr_db"]:
                noisy = clean + standardized * (
                    rms / 10 ** (float(snr) / 20)
                ).astype(np.float32)[:, None]
                raw_picks = np.asarray([robust_arrival(
                    trace, times,
                    noise_window_s=float(picker["noise_window_s"]),
                    smooth_samples=int(picker["smooth_samples"]),
                    sigma_multiplier=float(picker["sigma_multiplier"]),
                    consecutive=int(picker["consecutive_samples"]))
                    for trace in noisy], dtype=np.float32)
                peaks = np.max(np.abs(noisy), axis=1)
                valid = (np.isfinite(raw_picks)
                         & (peaks >= float(picker["minimum_receiver_peak_fraction_of_record"])
                            * peaks.max())
                         & (raw_picks <= times[-1]
                            - float(picker["last_record_s_excluded_for_pick"])))
                picks = raw_picks - float(picker["pick_bias_s"])
                picks[~valid] = np.nan
                count = int(valid.sum())
                if not 4 <= count <= 5:
                    continue
                result = fmm_estimate(fields, picks, bounds, spacing,
                                      search, {"minimum_valid_picks_for_fmm_identifiability": 4,
                                               "origin_time_bounds_s": search["origin_time_bounds_s"]})
                if result["estimated_source_m_s"] is None:
                    continue
                estimate = np.asarray(result["estimated_source_m_s"], dtype=np.float64)
                features = quality_features(fields, receivers,
                                            np.flatnonzero(valid), estimate,
                                            result, spacing)
                # Truth enters only here, after FMM search and feature calculation.
                error = estimate - truth
                eloc = float(np.linalg.norm(error[:3]))
                rows.append({"event": event["name"], "snr_db": int(snr),
                             "realization": realization, "seed": seed,
                             "valid_picks": count,
                             "estimated_source_m_s": estimate.tolist(),
                             "location_error_m": eloc,
                             "success_500m": bool(eloc <= 500.0),
                             "origin_time_error_s": float(abs(error[3])),
                             "features": features})
        print(f"quality features complete: {event['name']}", flush=True)

    if len(rows) < 10:
        raise RuntimeError(f"Too few four-to-five-pick development cases: {len(rows)}")
    baseline = float(np.mean([row["success_500m"] for row in rows]))
    folds = {feature: grouped_cross_validation(rows, feature) for feature in FEATURES}
    rankings = sorted(FEATURES,
                      key=lambda feature: (-folds[feature]["precision"],
                                           -folds[feature]["coverage"], feature))
    best_feature = rankings[0]
    best_fold = folds[best_feature]
    gate = threshold_for(rows, best_feature, min_coverage=0.5)
    enabled = bool(gate is not None
                   and best_fold["coverage"] >= 0.4
                   and best_fold["precision"] >= baseline + 0.10)
    selected_gate = ({"enabled": True, **gate,
                      "feature": best_feature,
                      "loeo_coverage": best_fold["coverage"],
                      "loeo_precision": best_fold["precision"],
                      "development_baseline_precision": baseline,
                      "success_threshold_m": 500.0,
                      "rule": f"accept FMM estimate only if {best_feature} <= {gate['threshold']:.12g}"}
                     if enabled else {
                         "enabled": False, "feature": None, "threshold": None,
                         "loeo_coverage": best_fold["coverage"],
                         "loeo_precision": best_fold["precision"],
                         "development_baseline_precision": baseline,
                         "success_threshold_m": 500.0,
                         "rule": "No quality gate met the prespecified grouped-validation criteria; always accept an identifiable FMM estimate."})
    result = {
        "purpose": spec["purpose"],
        "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "source_event_protocol_sha256": sha256(event_protocol),
        "checkpoint_sha256": sha256(ROOT / spec["checkpoint"]),
        "velocity_sha256": sha256(ROOT / spec["velocity_file"]),
        "receiver_sha256": sha256(receiver_path),
        "fmm_cache_sha256": sha256(ROOT / spec["fmm_cache"]),
        "events": rows,
        "n_sparse_pick_cases": len(rows),
        "development_baseline_precision": baseline,
        "feature_thresholds_fit_all_development": {
            feature: threshold_for(rows, feature, min_coverage=0.5)
            for feature in FEATURES},
        "grouped_validation_by_source": folds,
        "feature_ranking": rankings,
        "selected_gate": selected_gate,
        "limitations": spec["limitations"],
    }
    result_path, report_path, gate_path = outputs
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    gate_path.write_text(json.dumps(selected_gate, indent=2) + "\n",
                         encoding="utf-8")
    lines = ["# FMM sparse-pick quality-gate development", "",
             spec["limitations"], "",
             f"Four-to-five-pick development cases: **{len(rows)}**.",
             f"Always-accept success precision (<=500 m): **{baseline:.3f}**.", "",
             "| Feature (lower is better) | LOEO coverage | LOEO precision |", "|---|---:|---:|"]
    for feature in rankings:
        lines.append(f"| {feature} | {folds[feature]['coverage']:.3f} | "
                     f"{folds[feature]['precision']:.3f} |")
    lines.extend(["", f"Selected gate: **{selected_gate['rule']}**",
                  f"Gate enabled: **{selected_gate['enabled']}**.", "",
                  "Threshold selection used leave-one-source-out validation. The locked-source test must evaluate the selected rule without changing it."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(selected_gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
