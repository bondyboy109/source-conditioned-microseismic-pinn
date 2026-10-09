"""Summarize source-depth errors and boundary/pick diagnostics from saved runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "experiments" / name).read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def main() -> None:
    loc = load("source_conditioned_low_rank_localization_results.json")
    cross = load("source_conditioned_low_rank_cross_geology_results.json")
    noise = load("source_conditioned_low_rank_noise_results.json")
    rows = []
    for record in loc["events"]:
        truth = record["truth_m_s_post_search"]
        wave = record["waveform_only"]
        hybrid = record.get("waveform_plus_fmm")
        estimate = (hybrid or wave)["estimated_source_m_s"]
        rows.append({"dataset": "Overthrust confirmation", "event": record["event"],
                     "picks": record["valid_arrival_picks"], "truth_depth_m": truth[2],
                     "estimated_depth_m": estimate[2],
                     "depth_error_m": estimate[2] - truth[2],
                     "location_error_m": (hybrid or wave)["location_error_m"],
                     "origin_error_s": (hybrid or wave)["origin_time_error_s"],
                     "boundary": bool((hybrid or wave).get("selected_on_boundary", False)),
                     "branch": "hybrid" if hybrid else "waveform_only"})
    for record in cross["events"]:
        truth = record["truth_m_s"]
        for branch in ("waveform_plus_fmm", "fmm"):
            result = record["branches"][branch]
            estimate = result["estimated_source_m_s"]
            rows.append({"dataset": "Layered post-hoc audit", "event": record["event"],
                         "picks": record["valid_arrival_picks"], "truth_depth_m": truth[2],
                         "estimated_depth_m": estimate[2],
                         "depth_error_m": estimate[2] - truth[2],
                         "location_error_m": result["location_error_m"],
                         "origin_error_s": result["origin_time_error_s"],
                         "boundary": bool(abs(estimate[2] - 1500.0) < 1.0
                                          or abs(estimate[2] - 100.0) < 1.0),
                         "branch": branch})
    groups = {}
    for label, selected in (("overthrust_hybrid", [r for r in rows if r["dataset"].startswith("Overthrust") and r["branch"] == "hybrid"]),
                            ("layered_hybrid", [r for r in rows if r["dataset"].startswith("Layered") and r["branch"] == "waveform_plus_fmm"]),
                            ("layered_fmm", [r for r in rows if r["dataset"].startswith("Layered") and r["branch"] == "fmm"])):
        dz = [abs(r["depth_error_m"]) for r in selected]
        eloc = [r["location_error_m"] for r in selected]
        picks = [r["picks"] for r in selected]
        groups[label] = {"n": len(selected), "median_abs_depth_error_m": statistics.median(dz),
                         "median_location_error_m": statistics.median(eloc),
                         "median_picks": statistics.median(picks),
                         "depth_boundary_count": sum(r["boundary"] for r in selected),
                         "depth_error_p90_m": percentile(dz, .9)}
    noise_groups = {}
    for snr in (20.0, 10.0, 5.0):
        selected = [r for r in noise["events"] if r["snr_db"] == snr]
        noise_groups[str(int(snr))] = {
            "attempts": len(selected),
            "hybrid_available": sum("waveform_plus_fmm" in r["branches"] for r in selected),
            "median_picks": statistics.median(r["valid_arrival_picks"] for r in selected),
            "pick_count_range": [min(r["valid_arrival_picks"] for r in selected),
                                 max(r["valid_arrival_picks"] for r in selected)],
        }
    result = {"purpose": "post-hoc source-depth failure diagnosis",
              "groups": groups, "noise_pick_counts": noise_groups,
              "records": rows,
              "interpretation": "Saved experiments do not contain per-event per-receiver true arrival residuals or a receiver geometry rank/conditioning analysis. These summary associations cannot establish causation. Depth-boundary estimates and large depth errors motivate a development-only source-depth profile/observability audit; they do not justify changing the locked confirmation objective."}
    (ROOT / "experiments/source_depth_failure_diagnosis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Source-depth failure diagnosis", "",
             "This is a post-hoc summary of saved localization and noise records. It does not alter the locked method.", "",
             "| Group | n | Median picks | Median abs depth error (m) | P90 abs depth error (m) | Median eloc (m) | Depth-boundary estimates |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"overthrust_hybrid": "Overthrust hybrid", "layered_hybrid": "Layered hybrid", "layered_fmm": "Layered FMM"}
    for key, values in groups.items():
        lines.append(f"| {labels[key]} | {values['n']} | {values['median_picks']:.1f} | {values['median_abs_depth_error_m']:.1f} | {values['depth_error_p90_m']:.1f} | {values['median_location_error_m']:.1f} | {values['depth_boundary_count']}/{values['n']} |")
    lines.extend(["", "## Noise and pick counts", "", "| SNR (dB) | Attempts | Hybrid available | Median picks | Pick-count range |", "|---:|---:|---:|---:|---:|"])
    for snr, values in noise_groups.items():
        lines.append(f"| {snr} | {values['attempts']} | {values['hybrid_available']}/{values['attempts']} | {values['median_picks']:.1f} | {values['pick_count_range'][0]}–{values['pick_count_range'][1]} |")
    lines.extend(["", "## Event-level records", "", "| Dataset | Event | Branch | Picks | True depth (m) | Estimated depth (m) | Δz (m) | eloc (m) | Boundary |", "|---|---|---:|---:|---:|---:|---:|---:|---|"])
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['event']} | {r['branch']} | {r['picks']} | {r['truth_depth_m']:.1f} | {r['estimated_depth_m']:.1f} | {r['depth_error_m']:.1f} | {r['location_error_m']:.1f} | {'yes' if r['boundary'] else 'no'} |")
    lines.extend(["", result["interpretation"]])
    (ROOT / "experiments/source_depth_failure_diagnosis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"groups": groups, "noise_pick_counts": noise_groups}, indent=2))


if __name__ == "__main__":
    main()
