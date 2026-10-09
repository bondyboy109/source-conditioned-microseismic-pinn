"""Noise sensitivity test for the frozen source-conditioned low-rank operator."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_source_conditioned_low_rank_localization import load_model
from examples.run_traveltime_aligned_moe_localization import (
    arrival_scores, coarse_grid, refine, sampled_waveforms, waveform_scores,
    choose_diverse_starts,
)
from examples.run_holdout_noise_picker_comparison import robust_arrival

PROTOCOL = ROOT / "experiments/source_conditioned_low_rank_noise_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_observation(path: Path) -> dict:
    with np.load(path) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        truth = np.asarray(archive["source_truth"], dtype=np.float64)
    return {"path": path, "data": data, "times": times,
            "receivers": receivers, "truth": truth,
            "sha256": sha256(path)}


def noisy_event(clean: dict, snr_db: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(clean["data"].astype(np.float64) ** 2)))
    sigma = rms / (10.0 ** (snr_db / 20.0))
    data = clean["data"].astype(np.float64) + rng.normal(0.0, sigma,
                                                          clean["data"].shape)
    data = data.astype(np.float32)
    noisy_rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    return {"path": clean["path"], "data": data, "times": clean["times"],
            "receivers": clean["receivers"], "rms": noisy_rms,
            "sha256": clean["sha256"]}


def corrected_picks(event: dict, arrival: dict) -> tuple[np.ndarray, int]:
    data, times = event["data"], event["times"]
    raw = np.asarray([robust_arrival(trace, times) for trace in data],
                     dtype=np.float32)
    peaks = np.max(np.abs(data), axis=1)
    valid = (np.isfinite(raw)
             & (peaks >= arrival["minimum_receiver_peak_fraction_of_record"]
                * peaks.max())
             & (raw <= times[-1] - arrival["last_record_s_excluded_for_pick"]))
    corrected = raw - float(arrival["pick_bias_s"])
    corrected[~valid] = np.nan
    return corrected, int(np.isfinite(corrected).sum())


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["output_result"]
    report = ROOT / spec["output_report"]
    if output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite noise results")
    localization = json.loads((ROOT / spec["localization_protocol"]).read_text(encoding="utf-8"))
    checkpoint = ROOT / spec["model_checkpoint"]
    paths = sorted((ROOT / spec["confirmation_directory"]).glob("*.npz"))
    if len(paths) != int(spec["event_count"]):
        raise ValueError("Confirmation event set changed")
    clean = [load_observation(path) for path in paths]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain = load_model(localization, clean[0]["receivers"], device)
    bounds = np.asarray(localization["shared_source_bounds_m_s"], dtype=np.float32)
    search = localization["search"]
    arrival = localization["arrival_constraint"]
    candidates = coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    started = time.perf_counter()
    for snr_db in spec["snr_db"]:
        for seed in spec["seeds"]:
            for event_index, clean_event in enumerate(clean):
                event_seed = int(seed) + event_index * 10007 + int(snr_db * 100)
                event = noisy_event(clean_event, float(snr_db), event_seed)
                corrected, valid_count = corrected_picks(event, arrival)
                arrival_available = valid_count >= int(arrival["minimum_valid_picks"])
                coarse = sampled_waveforms(event, domain,
                                           int(search["coarse_receiver_stride"]),
                                           int(search["coarse_time_stride"]), device)
                fine = sampled_waveforms(event, domain,
                                         int(search["refine_receiver_stride"]),
                                         int(search["refine_time_stride"]), device)
                wave = waveform_scores(model, candidates, domain, coarse,
                                       int(search["candidate_batch"]))
                travel = (arrival_scores(model, candidates, corrected,
                                         float(arrival["arrival_scale_s"]),
                                         int(search["candidate_batch"]))
                          if arrival_available else np.full(len(candidates), np.inf))
                branches = {}
                branch_specs = [("waveform_only", 0.0)]
                if arrival_available:
                    branch_specs.append(("waveform_plus_fmm",
                                         float(arrival["hybrid_weight"])))
                for name, alpha in branch_specs:
                    score = wave if alpha == 0.0 else wave + alpha * travel
                    selected = choose_diverse_starts(
                        candidates, score, int(search["restarts"]),
                        float(search["minimum_start_separation_m"]))
                    starts = [{"source_m_s": candidates[i].tolist(),
                               "coarse_total_loss": float(score[i])}
                              for i in selected]
                    winner, _ = refine(model, starts, domain, bounds, fine,
                                       corrected, search, arrival, alpha)
                    estimate = np.asarray(winner["estimated_source_m_s"], dtype=np.float64)
                    error = estimate - clean_event["truth"]
                    branches[name] = {
                        "estimated_source_m_s": estimate.tolist(),
                        "location_error_m": float(np.linalg.norm(error[:3])),
                        "origin_time_error_s": float(abs(error[3])),
                        "waveform_normalized_mse": float(winner["waveform_normalized_mse"]),
                    }
                rows.append({"event": clean_event["path"].name,
                             "snr_db": float(snr_db), "seed": int(seed),
                             "noise_seed": event_seed,
                             "valid_arrival_picks": valid_count,
                             "truth_source_m_s": clean_event["truth"].tolist(),
                             "branches": branches})
                print(clean_event["path"].name, f"SNR={snr_db:g}",
                      f"seed={seed}", "valid=", valid_count, flush=True)
    summaries = {}
    for snr_db in spec["snr_db"]:
        selected = [row for row in rows if row["snr_db"] == float(snr_db)]
        summaries[str(snr_db)] = {}
        for name in ("waveform_only", "waveform_plus_fmm"):
            values = [row["branches"][name]["location_error_m"]
                      for row in selected if name in row["branches"]]
            times = [row["branches"][name]["origin_time_error_s"]
                     for row in selected if name in row["branches"]]
            summaries[str(snr_db)][name] = {
                "n": len(values), "mean_eloc_m": float(np.mean(values)) if values else None,
                "median_eloc_m": float(np.median(values)) if values else None,
                "std_eloc_m": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "mean_et_s": float(np.mean(times)) if times else None,
                "hybrid_available_fraction": float(len(values) / len(selected)),
            }
    result = {"purpose": spec["purpose"], "protocol": str(PROTOCOL.resolve()),
              "protocol_sha256": sha256(PROTOCOL),
              "checkpoint_sha256": sha256(checkpoint), "device": str(device),
              "runtime_seconds": time.perf_counter() - started,
              "summary": summaries, "events": rows, "limitations": spec["limitations"]}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Source-conditioned low-rank noise sensitivity", "",
             "The operator, arrival picker, search grid, refinement settings, and hybrid weight were frozen from the clean confirmation protocol.", "",
             "| SNR (dB) | Objective | n | Mean eloc (m) | Median eloc (m) | Std eloc (m) | Mean et (s) |", "|---:|---|---:|---:|---:|---:|---:|"]
    for snr_db in spec["snr_db"]:
        for name, label in (("waveform_only", "Waveform only"),
                            ("waveform_plus_fmm", "Waveform + FMM arrival")):
            row = summaries[str(snr_db)][name]
            lines.append(f"| {snr_db:g} | {label} | {row['n']} | {row['mean_eloc_m'] if row['mean_eloc_m'] is not None else 'NA'} | {row['median_eloc_m'] if row['median_eloc_m'] is not None else 'NA'} | {row['std_eloc_m'] if row['std_eloc_m'] is not None else 'NA'} | {row['mean_et_s'] if row['mean_et_s'] is not None else 'NA'} |")
    lines.extend(["", spec["limitations"]])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
