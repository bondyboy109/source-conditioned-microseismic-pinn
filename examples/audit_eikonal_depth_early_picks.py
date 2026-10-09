"""Audit early waveform energy and false threshold picks on spent depth data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_fmm_traveltime import first_arrival
from examples.run_eikonal_depth_sweep import PROTOCOL, DATA, OUTPUT, preflight, arrival_metrics
from examples.run_eikonal_directional_grid import checked_record, predict
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

DESTINATION = ROOT / "experiments/eikonal_depth_sweep_early_pick_audit.json"


def early_picks(fd: np.ndarray, model: np.ndarray, times: np.ndarray, origin: float) -> dict:
    picks = np.asarray([first_arrival(trace, times, threshold=.05) for trace in model])
    fd_picks = np.asarray([first_arrival(trace, times, threshold=.05) for trace in fd])
    fd_peaks = np.max(np.abs(fd), axis=1)
    selected = (fd_peaks >= .01 * fd_peaks.max()) & np.isfinite(fd_picks) & (fd_picks <= 1.85)
    if selected.sum() < 6:
        raise ValueError("FD does not provide enough valid receiver picks")
    before = times < origin - 0.10
    if not before.any():
        raise ValueError("No samples in the preregistered early window")
    fd_rms = float(np.sqrt(np.mean(fd.astype(np.float64) ** 2)))
    return {"early_window_end_s": float(origin - .10),
            "fd_early_rms_over_full_fd_rms": float(np.sqrt(np.mean(fd[:, before].astype(np.float64)**2)) / fd_rms),
            "model_early_rms_over_full_fd_rms": float(np.sqrt(np.mean(model[:, before].astype(np.float64)**2)) / fd_rms),
            "valid_fd_stations": int(selected.sum()),
            "model_picks_at_start_count": int(np.sum(selected & np.isfinite(picks) & (picks <= times[0]))),
            "model_picks_before_origin_minus_0_10_s": int(np.sum(selected & np.isfinite(picks)
                                                              & (picks < origin - .10))),
            "fd_picks_before_origin_minus_0_10_s": int(np.sum(selected & (fd_picks < origin - .10))),
            "original_threshold_arrival_metrics": arrival_metrics(fd, model, times)}


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"Diagnostic already exists: {DESTINATION}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    original = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if original["protocol_sha256"] != hash_file(PROTOCOL):
        raise ValueError("Original depth protocol changed")
    # Reuse the preflight manifest audit; it excludes the depth records from old-data overlap.
    events, _ = preflight(protocol)
    if len(events) != len(original["events"]):
        raise ValueError("Depth sweep event count changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name in ("eikonal", "straight"):
        model, payload = load_conditioned_checkpoint(ROOT / protocol[f"{name}_checkpoint"], device)
        model.eval().requires_grad_(False)
        scale = np.asarray([payload["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        models[name] = (model, scale)
    rows = []
    anchors = {r["name"]: ROOT / r["observation"] for r in protocol["anchors"]}
    for event, old in zip(events, original["events"]):
        path = DATA / event["file"]
        if event["file"] != old["file"] or hash_file(path) != old["sha256"]:
            raise ValueError(f"Depth record does not match original results: {path}")
        fd, times, receivers, truth = checked_record(path, event["source_m_s"], anchors[event["anchor"]])
        row = {"file": event["file"], "sha256": old["sha256"], "models": {}}
        for name, (model, scale) in models.items():
            prediction = predict(model, scale, fd, times, receivers, truth)
            values = early_picks(fd, prediction, times, float(truth[3]))
            before = old["models"][name]["arrival"]
            if (values["original_threshold_arrival_metrics"]["paired_pick_count"]
                    != before["paired_pick_count"]
                    or not np.isclose(values["original_threshold_arrival_metrics"]["rmse_s"],
                                      before["rmse_s"], rtol=1e-5, atol=1e-5)):
                raise ValueError(f"Original threshold metrics changed: {path}, {name}")
            row["models"][name] = values
        rows.append(row)
        print(f"{event['file']}: eikonal early picks="
              f"{row['models']['eikonal']['model_picks_before_origin_minus_0_10_s']}/"
              f"{row['models']['eikonal']['valid_fd_stations']}", flush=True)
    result = {"purpose": "post-hoc audit of early threshold artifacts on already-spent depth development sources",
              "protocol_sha256": original["protocol_sha256"],
              "original_result_sha256": hash_file(OUTPUT),
              "early_window": "t < source origin time minus 0.10 s; normalized by whole FD record RMS",
              "interpretation": "early threshold crossings are not physical first-arrival predictions; 5%-of-own-peak picks on noncausal neural background may be misleading; no refitting or new test",
              "events": rows}
    DESTINATION.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
