"""Development audit for reliability-adaptive FMM/waveform fusion.

The rule is fixed before evaluation and uses only pick count and the best FMM
arrival residual. Source truth is read only for reporting errors. This is a
post-hoc audit on already inspected synthetic records, not a blind test.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL = ROOT / "experiments/reliability_adaptive_fusion_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adaptive_weight(valid_picks: int, arrival_rmse_s: float, spec: dict) -> tuple[float, str]:
    """Map observable FMM reliability to a fixed weight or FMM fallback."""
    if valid_picks < int(spec["minimum_valid_picks"]):
        return 1.0, "fmm_fallback"
    count_score = np.clip(
        (valid_picks - spec["minimum_valid_picks"])
        / (spec["full_pick_count"] - spec["minimum_valid_picks"]), 0.0, 1.0)
    residual_score = np.clip(
        (spec["arrival_rmse_good_s"] - arrival_rmse_s)
        / (spec["arrival_rmse_good_s"] - spec["arrival_rmse_bad_s"]), 0.0, 1.0)
    reliability = float(count_score * residual_score)
    if reliability < float(spec["fallback_reliability_threshold"]):
        return 1.0, "fmm_fallback"
    # Less reliable arrivals receive less FMM loss; the pure FMM result is the
    # fallback only when the arrival evidence is below the fixed reliability gate.
    weight = float(spec["min_hybrid_weight"]
                   + (spec["max_hybrid_weight"] - spec["min_hybrid_weight"])
                   * reliability)
    return weight, "adaptive_hybrid"


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path = ROOT / spec["output_result"]
    report_path = ROOT / spec["output_report"]
    if result_path.exists():
        raise FileExistsError("Refusing to overwrite adaptive-fusion audit")
    source_result = ROOT / spec["source_result"]
    fmm_result = ROOT / spec["fmm_result"]
    if sha256(source_result) != spec["source_result_sha256"]:
        raise ValueError("Source result changed")
    if sha256(fmm_result) != spec["fmm_result_sha256"]:
        raise ValueError("FMM result changed")
    source = json.loads(source_result.read_text(encoding="utf-8"))
    fmm = json.loads(fmm_result.read_text(encoding="utf-8"))
    fmm_by_event = {row["event"]: row for row in fmm["events"]}
    rows = []
    for row in source["events"]:
        event = row["event"]
        if event not in fmm_by_event:
            raise ValueError(f"Missing FMM row for {event}")
        baseline = fmm_by_event[event]
        adaptive, decision = adaptive_weight(
            int(baseline["fmm"]["valid_fd_picks"]), float(spec["missing_arrival_rmse_s"]), spec)
        hybrid_error = float(baseline["hybrid"]["location_error_m"])
        fmm_error = float(baseline["fmm"]["location_error_m"])
        waveform_error = float(baseline["pinn_only"]["location_error_m"])
        # The adaptive audit uses the already computed FMM-seeded waveform
        # candidate as the hybrid estimate. It does not rerun or retune search.
        selected_error = fmm_error if decision == "fmm_fallback" else hybrid_error
        rows.append({
            "event": event,
            "valid_picks": int(baseline["fmm"]["valid_fd_picks"]),
            "arrival_rmse_s": None,
            "adaptive_reliability_weight": adaptive,
            "decision": decision,
            "waveform_only_location_error_m": waveform_error,
            "fixed_hybrid_location_error_m": hybrid_error,
            "pure_fmm_location_error_m": fmm_error,
            "adaptive_location_error_m": selected_error,
            "adaptive_improves_over_fixed_hybrid": bool(selected_error < hybrid_error),
            "adaptive_improves_over_waveform_only": bool(selected_error < waveform_error),
        })
    def values(key: str) -> np.ndarray:
        return np.asarray([r[key] for r in rows], dtype=np.float64)
    summary = {}
    for name, key in (("waveform_only", "waveform_only_location_error_m"),
                      ("fixed_hybrid", "fixed_hybrid_location_error_m"),
                      ("pure_fmm", "pure_fmm_location_error_m"),
                      ("adaptive", "adaptive_location_error_m")):
        v = values(key)
        summary[name] = {"n": len(v), "mean_eloc_m": float(v.mean()),
                         "median_eloc_m": float(np.median(v)),
                         "events_below_500_m": int(np.sum(v < 500.0))}
    result = {
        "purpose": spec["purpose"], "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL), "source_result_sha256": sha256(source_result),
        "fmm_result_sha256": sha256(fmm_result), "summary": summary, "events": rows,
        "limitations": spec["limitations"],
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Reliability-adaptive fusion development audit", "", spec["limitations"], "",
             "| Method | Mean eloc (m) | Median eloc (m) | <500 m |", "|---|---:|---:|---:|"]
    for name in ("waveform_only", "fixed_hybrid", "adaptive", "pure_fmm"):
        item = summary[name]
        lines.append(f"| {name} | {item['mean_eloc_m']:.1f} | {item['median_eloc_m']:.1f} | {item['events_below_500_m']}/{item['n']} |")
    lines.extend(["", "| Event | Picks | Arrival RMSE (s) | Adaptive weight | Decision | Waveform (m) | Fixed hybrid (m) | Adaptive (m) | FMM (m) |", "|---|---:|---:|---:|---|---:|---:|---:|---:|"])
    for r in rows:
        arrival_text = "NA" if r["arrival_rmse_s"] is None else f"{r['arrival_rmse_s']:.4f}"
        lines.append(f"| {r['event']} | {r['valid_picks']} | {arrival_text} | {r['adaptive_reliability_weight']:.3f} | {r['decision']} | {r['waveform_only_location_error_m']:.1f} | {r['fixed_hybrid_location_error_m']:.1f} | {r['adaptive_location_error_m']:.1f} | {r['pure_fmm_location_error_m']:.1f} |")
    lines.extend(["", "The rule was applied without source truth and without changing the frozen forward model or search."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
