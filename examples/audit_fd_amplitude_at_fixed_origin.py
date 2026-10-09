"""Profile nonnegative source amplitude at the already selected origin times.

This is a post-hoc diagnostic using the saved FD depth profiles. The original
unit-amplitude waveform rankings and origin times are left unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments/fd_shallow_depth_profile_audit.json"
RESULT = ROOT / "experiments/fd_amplitude_fixed_origin_audit.json"
REPORT = RESULT.with_suffix(".md")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profile_amplitude(original_mse: float, energy_ratio: float) -> dict:
    """Return min_{a>=0} ||a p - y||^2 / ||y||^2 from E and Q.

    E = ||p-y||^2/||y||^2, Q = ||p||^2/||y||^2, and
    C = <p,y>/||y||^2 = (Q+1-E)/2. Time shift stays fixed.
    """
    e, q = float(original_mse), float(energy_ratio)
    if not (math.isfinite(e) and math.isfinite(q) and e >= 0 and q >= 0):
        raise ValueError("Nonfinite or negative waveform statistics")
    cross = (q + 1 - e) / 2
    if q == 0:
        if not math.isclose(e, 1, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Zero prediction must yield normalized MSE of one")
        gain, adjusted = 0.0, 1.0
    else:
        gain = max(0.0, cross / q)
        adjusted = max(0.0, 1 - 2 * gain * cross + gain * gain * q)
    if adjusted > e + 1e-8 * max(1.0, e):
        raise ValueError("Profiling amplitude increased the original loss")
    return {"normalized_inner_product": cross,
            "best_nonnegative_gain_at_fixed_origin": gain,
            "gain_adjusted_normalized_mse_at_fixed_origin": adjusted,
            "gain_adjusted_relative_l2_at_fixed_origin": math.sqrt(adjusted)}


def main() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("Amplitude audit already exists; refusing to overwrite")
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    frozen = (("experiments/fd_waveform_global_refined_confirmation_protocol.json",
               source["protocol_sha256"]),
              ("experiments/fd_global_grid_ranking_audit.json",
               source["grid_ranking_audit_sha256"]),
              ("data/observations/experiment_draft.json", source["config_sha256"]),
              ("fd_waveform.py", source["solver_sha256"]))
    for relative, expected in frozen:
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen depth-profile input changed: {relative}")
    rows = []
    for event in source["events"]:
        scores = []
        for row in event["scores"]:
            scores.append({"depth_m": row["depth_m"],
                           "origin_time_s_fixed": row["origin_time_s"],
                           "unit_amplitude_normalized_mse": row["normalized_mse"],
                           "unit_amplitude_energy_ratio": row[
                               "shifted_prediction_to_observation_energy"],
                           **profile_amplitude(
                               row["normalized_mse"],
                               row["shifted_prediction_to_observation_energy"])})
        original = min(scores, key=lambda s: s["unit_amplitude_normalized_mse"])
        adjusted = min(scores, key=lambda s: s[
            "gain_adjusted_normalized_mse_at_fixed_origin"])
        rows.append({"id": event["id"], "observation_sha256": event["observation_sha256"],
                     "fixed_nearest_grid_xy_m": event["fixed_nearest_grid_xy_m"],
                     "source_truth_m_s_post_hoc": event["truth_source_m_s_post_hoc"],
                     "scores": scores,
                     "original_profile_minimum_depth_m": original["depth_m"],
                     "gain_adjusted_profile_minimum_depth_m_at_fixed_origins": adjusted[
                         "depth_m"]})
        print(f"{event['id']}: unit-gain minimum z={original['depth_m']:.0f} m; "
              f"gain-adjusted fixed-origin minimum z={adjusted['depth_m']:.0f} m "
              f"(MSE={adjusted['gain_adjusted_normalized_mse_at_fixed_origin']:.4f})")

    result = {"role": "post-hoc nonnegative amplitude profiling at frozen origin times; "
                      "not a blind inversion or a reoptimized time-and-gain search",
              "input_sha256": digest(INPUT), "source_fd_protocol_sha256": source["protocol_sha256"],
              "events": rows}
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# FD amplitude attribution at fixed origin time", "",
             "The source is the saved two-event depth profile; no FD waveforms "
             "were regenerated. At each depth, the time shift remains the "
             "unit-amplitude optimum from that profile. Only a nonnegative "
             "scalar gain is fit afterward. The original localization baseline "
             "and its objective remain unchanged.", "",
             "| Event | Depth (m) | Fixed origin (s) | Unit MSE | Energy ratio | "
             "Best gain | Gain-adjusted MSE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for event in rows:
        for score in event["scores"]:
            lines.append(f"| {event['id']} | {score['depth_m']:.0f} | "
                         f"{score['origin_time_s_fixed']:.3f} | "
                         f"{score['unit_amplitude_normalized_mse']:.4f} | "
                         f"{score['unit_amplitude_energy_ratio']:.4f} | "
                         f"{score['best_nonnegative_gain_at_fixed_origin']:.4f} | "
                         f"{score['gain_adjusted_normalized_mse_at_fixed_origin']:.4f} |")
    lines.extend(["", "A large decrease after fitting gain indicates an "
                  "amplitude contribution at the frozen shift, but the "
                  "remaining loss includes timing and waveform-shape mismatch. "
                  "For event 01 at 100 m, the best nonnegative gain is zero "
                  "and the adjusted loss of one equals a zero-waveform "
                  "prediction; this is not a successful waveform fit. For "
                  "event 05 at 100 m, gain 0.1294 reduces the loss from "
                  "35.3482 to 0.2244, showing a substantial amplitude "
                  "contribution but leaving nonzero residual mismatch. "
                  "The adjusted minimum over depths is not the result of a "
                  "joint time-and-amplitude search; doing that requires "
                  "rescoring every time shift. These inspected events and "
                  "truth-selected horizontal nodes cannot support a new "
                  "blind localization claim.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {RESULT} and {REPORT}")


if __name__ == "__main__":
    main()
