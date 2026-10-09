"""Calibrate SNR levels for a four-to-five-pick transition test.

This development-only script uses archived development events and only counts
valid picks. It does not run localization or use source truth for selection.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/adaptive_fusion_transition_development_protocol.json"
sys.path.insert(0, str(ROOT))

from examples.run_holdout_noise_picker_comparison import robust_arrival  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_record(path: Path, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
    keep = times <= window_s + 1e-7
    data, times = data[:, keep], times[keep]
    if data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError(f"Invalid cropped development event: {path}")
    return data, times


def valid_pick_count(data: np.ndarray, times: np.ndarray,
                     picker: dict) -> int:
    raw = np.asarray([robust_arrival(
        trace, times,
        noise_window_s=float(picker["noise_window_s"]),
        smooth_samples=int(picker["smooth_samples"]),
        sigma_multiplier=float(picker["sigma_multiplier"]),
        consecutive=int(picker["consecutive_samples"]))
        for trace in data], dtype=np.float32)
    peaks = np.max(np.abs(data), axis=1)
    valid = (np.isfinite(raw)
             & (peaks >= float(picker["minimum_receiver_peak_fraction_of_record"])
                * peaks.max())
             & (raw <= times[-1]
                - float(picker["last_record_s_excluded_for_pick"])))
    return int(valid.sum())


def select_snr(rows: list[dict], snr_values: list[int]) -> tuple[int, list[int]]:
    summaries = []
    for snr in snr_values:
        counts = [row["valid_picks"] for row in rows if row["snr_db"] == snr]
        if not counts:
            raise ValueError(f"No calibration counts for SNR={snr} dB")
        summaries.append((int(snr), float(np.mean(
            np.logical_and(np.asarray(counts) >= 4, np.asarray(counts) <= 5))),
            float(abs(np.median(counts) - 4.5))))
    chosen = min(summaries, key=lambda row: (-row[1], row[2], row[0]))[0]
    neighbors = sorted(snr_values, key=lambda value: (abs(value - chosen), value))[:3]
    return chosen, sorted(neighbors)


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["output_result"]
    report = ROOT / spec["output_report"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite transition-calibration results")
    if not spec["protocol_written_before_calibration"]:
        raise ValueError("Calibration protocol was not registered before running")

    folder = ROOT / spec["observation_directory"]
    events = []
    for event in spec["events"]:
        path = folder / event["file"]
        if sha256(path) != event["sha256"]:
            raise ValueError(f"Development event hash changed: {path}")
        data, times = crop_record(path, float(spec["model_window_s"]))
        events.append((event, data, times))

    noise_rows = []
    picker = spec["picker"]
    for event_index, (event, clean, times) in enumerate(events):
        signal_rms = np.sqrt(np.mean(clean.astype(np.float64) ** 2, axis=1))
        for realization, base_seed in enumerate(spec["realization_seeds"], start=1):
            seed = int(base_seed + event_index * 1009)
            standardized = np.random.default_rng(seed).standard_normal(
                clean.shape).astype(np.float32)
            for snr in spec["snr_db"]:
                noise = standardized * (
                    signal_rms / 10.0 ** (float(snr) / 20.0)
                ).astype(np.float32)[:, None]
                noisy = clean + noise
                count = valid_pick_count(noisy, times, picker)
                noise_rows.append({"event": event["name"], "snr_db": int(snr),
                                   "realization": realization, "seed": seed,
                                   "valid_picks": count})
        print(f"picker-noise sweep complete: {event['name']}", flush=True)

    selected_snr, confirmation_snr = select_snr(
        noise_rows, [int(x) for x in spec["snr_db"]])
    summary = {}
    for snr in spec["snr_db"]:
        counts = np.asarray([row["valid_picks"] for row in noise_rows
                             if row["snr_db"] == int(snr)], dtype=np.int64)
        summary[str(snr)] = {
            "n": int(len(counts)),
            "pick_count_histogram": {
                str(count): int(np.sum(counts == count))
                for count in range(int(counts.min()), int(counts.max()) + 1)
            },
            "fraction_with_4_or_5_picks": float(np.mean(
                (counts >= 4) & (counts <= 5))),
            "median_valid_picks": float(np.median(counts)),
            "mean_valid_picks": float(np.mean(counts)),
        }
    result = {
        "purpose": spec["purpose"],
        "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "selected_snr_db": selected_snr,
        "confirmation_snr_db": confirmation_snr,
        "selection_rule": spec["selection_rule"],
        "summary_by_snr": summary,
        "events": noise_rows,
        "limitations": spec["limitations"],
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Adaptive-fusion transition SNR calibration", "",
             spec["limitations"], "",
             "| SNR (dB) | Trials | Median valid picks | Trials with 4-5 picks |",
             "|---:|---:|---:|---:|"]
    for snr in spec["snr_db"]:
        row = summary[str(snr)]
        lines.append(f"| {snr} | {row['n']} | {row['median_valid_picks']:.1f} | "
                     f"{row['fraction_with_4_or_5_picks']:.1%} |")
    lines.extend(["", f"Selected center SNR: **{selected_snr} dB**.",
                  f"Locked confirmation SNRs: **{confirmation_snr} dB**.", "",
                  "These values were selected from pick counts only; no localization error was inspected."])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Selected {confirmation_snr} dB for the independent transition confirmation.",
          flush=True)


if __name__ == "__main__":
    main()
