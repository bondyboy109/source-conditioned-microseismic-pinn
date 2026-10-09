"""Compare threshold and robust MAD onset picks under synthetic noise."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import skfmm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from audit_fmm_traveltime import first_arrival, trilinear

VELOCITY = ROOT / "data/velocity/vp_model.npy"
OBSERVATIONS = ROOT / "data/observations/holdout_20260927"
OUT = ROOT / "experiments/holdout_20260927_noise_picker_results.json"
SPACING = 160.0
EVENTS = ["event_01.npz", "event_03.npz", "event_06.npz", "event_08.npz"]
SNR_VALUES = [20.0, 10.0, 5.0]
REALIZATION_SEEDS = [20260927, 20260928]
BOUNDS = np.asarray([[2000., 14000.], [2000., 14000.], [100., 1500.]])


def robust_arrival(trace, times, noise_window_s=0.10, smooth_samples=5,
                   sigma_multiplier=6.0, consecutive=3):
    """Pick the first sustained envelope above a robust pre-event noise level."""
    n_noise = max(5, min(len(trace) // 4, int(round(noise_window_s / np.median(np.diff(times))))))
    baseline = trace[:n_noise]
    center = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - center)))
    noise_scale = max(1.4826 * mad, 1e-12)
    envelope = np.convolve(np.abs(trace - center),
                           np.ones(smooth_samples) / smooth_samples, mode="same")
    active = envelope >= sigma_multiplier * noise_scale
    active[:n_noise] = False
    run = np.convolve(active.astype(np.int8), np.ones(consecutive, dtype=np.int8), "valid")
    indices = np.flatnonzero(run >= consecutive)
    return float(times[indices[0]]) if len(indices) else np.nan


def candidates():
    axes = [np.linspace(low, high, count, dtype=np.float64)
            for (low, high), count in zip(BOUNDS, (6, 6, 4))]
    return np.asarray(np.meshgrid(*axes, indexing="ij"), dtype=np.float64).reshape(3, -1).T


def travel_fields(velocity, source_candidates, receivers):
    output = []
    for xyz in source_candidates:
        idx = np.clip(np.rint(xyz / SPACING).astype(int), 0, np.asarray(velocity.shape) - 1)
        level = np.ones(velocity.shape, dtype=np.float64)
        level[tuple(idx)] = -1.0
        field = np.asarray(skfmm.travel_time(level, speed=velocity, dx=SPACING), dtype=np.float64)
        output.append(trilinear(field, receivers, SPACING))
    return np.asarray(output)


def locate(picks, candidates_xyz, travel):
    valid = np.isfinite(picks)
    if int(valid.sum()) < 4:
        return None, int(valid.sum())
    best = None
    for xyz, times in zip(candidates_xyz, travel):
        origin = float(np.clip(np.median(picks[valid] - times[valid]), 0.05, 0.5))
        residual = origin + times[valid] - picks[valid]
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, np.r_[xyz, origin])
    return best, int(valid.sum())


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    velocity = np.asarray(np.load(VELOCITY), dtype=np.float64)[::4, ::4, ::4]
    source_candidates = candidates()
    records = []
    started = time.perf_counter()
    for event_index, event_name in enumerate(EVENTS):
        with np.load(OBSERVATIONS / event_name) as archive:
            data = np.asarray(archive["data"], dtype=np.float64)
            times = np.asarray(archive["times"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        travel = travel_fields(velocity, source_candidates, receivers)
        signal_rms = float(np.sqrt(np.mean(data ** 2)))
        for snr_db in SNR_VALUES:
            noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
            for seed in REALIZATION_SEEDS:
                rng = np.random.default_rng(seed + int(snr_db * 100) + event_index * 1000)
                noisy = data + rng.normal(0.0, noise_rms, size=data.shape)
                threshold_picks = np.asarray([first_arrival(trace, times) for trace in noisy])
                robust_picks = np.asarray([robust_arrival(trace, times) for trace in noisy])
                for picker_name, picks in (("threshold", threshold_picks), ("robust_mad", robust_picks)):
                    best, valid_count = locate(picks, source_candidates, travel)
                    row = {"event": event_name, "snr_db": snr_db, "seed": seed,
                           "picker": picker_name, "valid_picks": valid_count}
                    if best is None:
                        row["status"] = "failed"
                    else:
                        row["status"] = "ok"
                        estimate = best[1]
                        row.update({"estimated_source_m_s": estimate.tolist(),
                                    "location_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
                                    "origin_time_error_s": float(abs(estimate[3] - truth[3])),
                                    "arrival_rmse_s": float(best[0])})
                    records.append(row)
    summary = {}
    for picker in ("threshold", "robust_mad"):
        summary[picker] = {}
        for snr_db in SNR_VALUES:
            subset = [row for row in records if row["picker"] == picker and row["snr_db"] == snr_db]
            valid = [row for row in subset if row["status"] == "ok"]
            summary[picker][str(int(snr_db))] = {
                "records": len(valid), "failure_count": len(subset) - len(valid),
                "mean_location_error_m": float(np.mean([row["location_error_m"] for row in valid])) if valid else None,
                "median_location_error_m": float(np.median([row["location_error_m"] for row in valid])) if valid else None,
                "mean_origin_time_error_s": float(np.mean([row["origin_time_error_s"] for row in valid])) if valid else None,
                "mean_arrival_rmse_s": float(np.mean([row["arrival_rmse_s"] for row in valid])) if valid else None,
            }
    result = {"role": "noise sensitivity comparison of arrival pickers",
              "events": EVENTS, "snr_db": SNR_VALUES, "realization_seeds": REALIZATION_SEEDS,
              "candidate_count": len(source_candidates), "fmm_spacing_m": SPACING,
              "summary": summary, "records": records,
              "runtime_seconds": time.perf_counter() - started,
              "limits": "Four independent events and two realizations per SNR. Both pickers use synthetic noisy FD records; no neural localization or UQ calibration claim."}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {OUT}; runtime={result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
