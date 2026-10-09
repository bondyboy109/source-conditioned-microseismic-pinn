"""Development-only audit of confidence-weighted first-arrival residuals.

This script does not alter the locked confirmation protocol. It measures how
many picks remain after confidence weighting and compares hard and soft arrival
residuals on records already assigned to development/noise diagnostics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_holdout_noise_picker_comparison import robust_arrival

PROTOCOL = ROOT / "experiments/soft_arrival_weight_development_protocol.json"


def confidence(trace: np.ndarray, times: np.ndarray, pick: float) -> float:
    if not np.isfinite(pick):
        return 0.0
    dt = float(np.median(np.diff(times)))
    n_noise = max(5, min(len(trace) // 4, int(round(.10 / dt))))
    baseline = trace[:n_noise]
    scale = max(1.4826 * float(np.median(np.abs(baseline - np.median(baseline)))), 1e-12)
    index = int(np.argmin(np.abs(times - pick)))
    window = trace[max(0, index - 2):min(len(trace), index + 3)]
    snr = float(np.max(np.abs(window - np.median(baseline))) / scale)
    # Bounded confidence used only for diagnostic weighting, not a calibrated probability.
    return float(np.clip((snr - 3.0) / 12.0, 0.0, 1.0))


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = []
    for entry in spec["records"]:
        path = ROOT / entry["file"]
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
        picks = np.asarray([robust_arrival(trace, times) for trace in data])
        weights = np.asarray([confidence(trace, times, pick)
                              for trace, pick in zip(data, picks)])
        valid = np.isfinite(picks)
        for floor in spec["confidence_floors"]:
            selected = valid & (weights >= float(floor))
            rows.append({"file": str(path.relative_to(ROOT)),
                         "snr_db": entry["snr_db"], "seed": entry["seed"],
                         "confidence_floor": float(floor),
                         "raw_valid_picks": int(valid.sum()),
                         "weighted_valid_picks": int(selected.sum()),
                         "weight_sum": float(weights[selected].sum()),
                         "median_confidence": float(np.median(weights[valid])) if valid.any() else 0.0})
    summary = {}
    for snr in spec["snr_db"]:
        summary[str(snr)] = {}
        for floor in spec["confidence_floors"]:
            selected = [r for r in rows if r["snr_db"] == snr and r["confidence_floor"] == floor]
            values = [r["weighted_valid_picks"] for r in selected]
            summary[str(snr)][str(floor)] = {
                "records": len(values), "mean_weighted_picks": float(np.mean(values)),
                "minimum_weighted_picks": int(min(values)),
                "records_with_at_least_six": int(sum(v >= 6 for v in values)),
            }
    result = {"purpose": spec["purpose"], "protocol": str(PROTOCOL.resolve()),
              "summary": summary, "rows": rows, "interpretation": spec["interpretation"]}
    (ROOT / spec["output_result"]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Soft arrival-weight development audit", "",
             "Confidence is a bounded SNR-like diagnostic and is not an empirical probability.", "",
             "| SNR | Floor | Mean weighted picks | Minimum | Records with >=6 |", "|---:|---:|---:|---:|---:|"]
    for snr in spec["snr_db"]:
        for floor in spec["confidence_floors"]:
            row = summary[str(snr)][str(floor)]
            lines.append(f"| {snr:g} | {floor:g} | {row['mean_weighted_picks']:.2f} | {row['minimum_weighted_picks']} | {row['records_with_at_least_six']}/{row['records']} |")
    lines.extend(["", result["interpretation"]])
    (ROOT / spec["output_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
