"""Noise sensitivity of the FMM arrival baseline on independent holdouts."""

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
OUT = ROOT / "experiments/holdout_20260927_noise_fmm_results.json"
SPACING = 160.0
SOURCE_BOUNDS = np.asarray([[2000., 14000.], [2000., 14000.], [100., 1500.]])
EVENTS = ["event_01.npz", "event_03.npz", "event_06.npz", "event_08.npz"]
SNR_VALUES = [20.0, 10.0, 5.0]
REALIZATION_SEEDS = [20260927, 20260928]


def receiver_times(field, receivers):
    return trilinear(field, receivers, SPACING)


def make_candidates():
    axes = [np.linspace(low, high, count, dtype=np.float64)
            for (low, high), count in zip(SOURCE_BOUNDS, (6, 6, 4))]
    return np.asarray(np.meshgrid(*axes, indexing="ij"), dtype=np.float64).reshape(3, -1).T


def build_fields(velocity, candidates, receivers):
    fields = []
    for source_xyz in candidates:
        index = np.clip(np.rint(source_xyz / SPACING).astype(int), 0,
                        np.asarray(velocity.shape) - 1)
        level = np.ones(velocity.shape, dtype=np.float64)
        level[tuple(index)] = -1.0
        field = np.asarray(skfmm.travel_time(level, speed=velocity, dx=SPACING), dtype=np.float64)
        fields.append(receiver_times(field, receivers))
    return np.asarray(fields, dtype=np.float64)


def locate_from_picks(picks, valid, candidates, travel_times):
    if int(valid.sum()) < 4:
        return None
    best = None
    for candidate, travel in zip(candidates, travel_times):
        origin = float(np.median(picks[valid] - travel[valid]))
        origin = float(np.clip(origin, 0.05, 0.50))
        residual = origin + travel[valid] - picks[valid]
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, candidate.copy(), origin, int(valid.sum()))
    return best


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    velocity = np.asarray(np.load(VELOCITY), dtype=np.float64)[::4, ::4, ::4]
    candidates = make_candidates()
    records = []
    clean_records = []
    started = time.perf_counter()
    for event_name in EVENTS:
        path = OBSERVATIONS / event_name
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float64)
            times = np.asarray(archive["times"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        travel_times = build_fields(velocity, candidates, receivers)
        signal_rms = float(np.sqrt(np.mean(data ** 2)))
        clean_picks = np.asarray([first_arrival(trace, times) for trace in data])
        clean_valid = np.isfinite(clean_picks)
        clean = locate_from_picks(clean_picks, clean_valid, candidates, travel_times)
        if clean is not None:
            _, clean_xyz, clean_t, clean_count = clean
            clean_estimate = np.r_[clean_xyz, clean_t]
            clean_records.append({"event": event_name, "valid_picks": clean_count,
                                  "estimated_source_m_s": clean_estimate.tolist(),
                                  "location_error_m": float(np.linalg.norm(clean_estimate[:3] - truth[:3])),
                                  "origin_time_error_s": float(abs(clean_estimate[3] - truth[3])),
                                  "arrival_rmse_s": float(clean[0])})
        for snr_db in SNR_VALUES:
            noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
            for seed in REALIZATION_SEEDS:
                event_index = EVENTS.index(event_name)
                rng = np.random.default_rng(seed + int(snr_db * 100) + event_index * 1000)
                noisy = data + rng.normal(0.0, noise_rms, size=data.shape)
                picks = np.asarray([first_arrival(trace, times) for trace in noisy])
                valid = np.isfinite(picks)
                best = locate_from_picks(picks, valid, candidates, travel_times)
                if best is None:
                    records.append({"event": event_name, "snr_db": snr_db,
                                    "seed": seed, "status": "failed",
                                    "valid_picks": int(valid.sum())})
                    continue
                _, estimate_xyz, estimate_t, valid_count = best
                estimate = np.r_[estimate_xyz, estimate_t]
                records.append({"event": event_name, "snr_db": snr_db, "seed": seed,
                                "status": "ok", "valid_picks": valid_count,
                                "source_truth_m_s": truth.tolist(),
                                "estimated_source_m_s": estimate.tolist(),
                                "location_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
                                "origin_time_error_s": float(abs(estimate[3] - truth[3])),
                                "arrival_rmse_s": float(best[0])})
    valid_records = [row for row in records if row["status"] == "ok"]
    summary = {}
    for snr_db in SNR_VALUES:
        subset = [row for row in valid_records if row["snr_db"] == snr_db]
        summary[str(int(snr_db))] = {
            "records": len(subset), "failure_count": 8 - len(subset),
            "mean_location_error_m": float(np.mean([row["location_error_m"] for row in subset])) if subset else None,
            "median_location_error_m": float(np.median([row["location_error_m"] for row in subset])) if subset else None,
            "mean_origin_time_error_s": float(np.mean([row["origin_time_error_s"] for row in subset])) if subset else None,
            "mean_arrival_rmse_s": float(np.mean([row["arrival_rmse_s"] for row in subset])) if subset else None,
        }
    result = {"role": "independent holdout FMM noise sensitivity",
              "events": EVENTS, "snr_db": SNR_VALUES,
              "realization_seeds": REALIZATION_SEEDS,
              "candidate_count": len(candidates), "fmm_spacing_m": SPACING,
              "clean_coarse_results": clean_records,
              "summary": summary, "records": records,
              "runtime_seconds": time.perf_counter() - started,
              "limits": "Four independent events, two realizations per SNR, coarse FMM grid without local refinement. Noise is synthetic Gaussian noise; no neural localization or UQ calibration claim."}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {OUT}; runtime={result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
