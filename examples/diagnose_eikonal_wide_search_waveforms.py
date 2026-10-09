"""Post-hoc receiver-waveform audit on training and spent held-out FD events."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_fmm_traveltime import first_arrival
from examples.diagnose_loo_waveform_alignment import predict, trace_alignment
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_wide_search_independent_protocol.json"
RESULTS = ROOT / "experiments/eikonal_wide_search_independent_results.json"
OUTPUT = ROOT / "experiments/eikonal_wide_search_waveform_diagnostic.json"
PLOTS = ROOT / "outputs/eikonal_wide_search_waveform_diagnostic"


def audit_record(model, path: Path, device, scale, group: str, known_hash: str) -> tuple[dict, tuple]:
    with np.load(path) as archive:
        times = np.asarray(archive["times"], dtype=np.float64)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
        truth = np.asarray(archive["source_truth"], dtype=np.float64)
    if hash_file(path) != known_hash:
        raise ValueError(f"Record changed since training/evaluation: {path}")
    observed, predicted, dt = predict(model, path, device, scale)
    if (observed.shape != (25, 1001) or predicted.shape != observed.shape
            or not np.all(np.isfinite(predicted)) or not np.isclose(dt, .002, atol=1e-6)):
        raise ValueError(f"Unexpected waveform dimensions / samples: {path}")
    peak_fd = np.max(np.abs(observed), axis=1)
    peak_pinn = np.max(np.abs(predicted), axis=1)
    picks_fd = np.asarray([first_arrival(trace, times, threshold=.05) for trace in observed])
    picks_pinn = np.asarray([first_arrival(trace, times, threshold=.05) for trace in predicted])
    enough_fd = peak_fd >= .01 * peak_fd.max()
    enough_pinn = peak_pinn >= .01 * peak_pinn.max()
    in_window_fd = picks_fd <= times[-1] - .15
    in_window_pinn = picks_pinn <= times[-1] - .15
    valid_fd = np.isfinite(picks_fd) & enough_fd & in_window_fd
    paired = valid_fd & np.isfinite(picks_pinn) & enough_pinn & in_window_pinn
    informative = np.flatnonzero(valid_fd)

    # Shift and gain use known observations. They diagnose the error source;
    # they are never applied in either localization search or its reported score.
    alignment = [trace_alignment(observed[i], predicted[i], int(round(.2 / dt)))
                 for i in informative if np.linalg.norm(predicted[i]) > 0]
    denominator = float(np.sum(observed ** 2))
    gain = float(np.sum(observed * predicted) / np.sum(predicted ** 2))
    error = float(np.linalg.norm(predicted - observed) / np.sqrt(denominator))
    gain_error = float(np.linalg.norm(gain * predicted - observed) / np.sqrt(denominator))
    amplitude = np.divide(peak_pinn[informative], peak_fd[informative],
                          out=np.zeros(len(informative)), where=peak_fd[informative] > 0)
    detail = []
    for i in range(25):
        fd_energy = float(np.sum(observed[i] ** 2))
        detail.append({"receiver_index": i, "coords_m": receivers[i].tolist(),
                       "fd_peak": float(peak_fd[i]), "pinn_peak": float(peak_pinn[i]),
                       "fd_pick_s": float(picks_fd[i]) if np.isfinite(picks_fd[i]) else None,
                       "pinn_pick_s": float(picks_pinn[i]) if np.isfinite(picks_pinn[i]) else None,
                       "fd_valid_by_fmm_rule": bool(valid_fd[i]),
                       "pinn_valid_by_same_rule": bool(np.isfinite(picks_pinn[i])
                                                       and enough_pinn[i] and in_window_pinn[i]),
                       "relative_l2": (float(np.linalg.norm(predicted[i] - observed[i]) /
                                             np.sqrt(fd_energy)) if fd_energy > 1e-24 else None)})
    row = {"group": group, "event": path.name, "observation_sha256": known_hash,
           "true_source_used_for_forward_diagnostic_only_m_s": truth.tolist(),
           "raw_relative_l2": error, "global_gain_diagnostic_only": gain,
           "l2_after_global_gain_diagnostic_only": gain_error,
           "fd_valid_picks": int(valid_fd.sum()),
           "fd_pick_rejections": {
               "no_threshold_crossing": int(np.count_nonzero(~np.isfinite(picks_fd))),
               "too_weak": int(np.count_nonzero(np.isfinite(picks_fd) & ~enough_fd)),
               "after_1_85_s": int(np.count_nonzero(np.isfinite(picks_fd) & enough_fd
                                                    & ~in_window_fd)),
           },
           "paired_picks": int(paired.sum()),
           "median_absolute_pick_difference_s": (float(np.median(np.abs(
               picks_pinn[paired] - picks_fd[paired]))) if paired.any() else None),
           "median_peak_amplitude_ratio_informative": (
               float(np.median(amplitude)) if len(amplitude) else None),
           "median_unshifted_abs_cosine_informative": (
               float(np.median([abs(a["unshifted_cosine"]) for a in alignment]))
               if alignment else None),
           "median_best_shifted_abs_cosine_informative": (
               float(np.median([abs(a["best_signed_cosine"]) for a in alignment]))
               if alignment else None),
           "median_best_absolute_shift_s_informative": (
               float(np.median([abs(a["best_shift_samples"] * dt) for a in alignment]))
               if alignment else None),
           "receivers": detail}
    return row, (times, receivers, observed, predicted, valid_fd)


def plot_traces(name: str, data: tuple) -> None:
    times, receivers, observed, predicted, valid = data
    indices = np.argsort(np.max(np.abs(observed), axis=1))[-4:][::-1]
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    for row, index in enumerate(indices):
        reference = observed[index] * 1e6
        model = predicted[index] * 1e6
        left, right = axes[row]
        left.plot(times, reference, color="#202124", linewidth=1.1, label="FD")
        left.plot(times, model, color="#007d75", linewidth=1.1, label="PINN")
        right.plot(times, model - reference, color="#a43b36", linewidth=1.0)
        for ax in (left, right):
            ax.axhline(0, linewidth=.6, color="0.65")
            ax.grid(alpha=.2)
            ax.set_xlim(0, 2)
            ax.axvline(1.85, color="0.5", linewidth=.8, linestyle=":")
        left.set_ylabel(f"R{index:02d} ({int(receivers[index, 0])}, "
                        f"{int(receivers[index, 1])}) m\n1e-6 units")
        left.set_title(f"Receiver {index:02d}: {'valid' if valid[index] else 'rejected'} FD pick",
                       fontsize=10)
        right.set_title("PINN - FD", fontsize=10)
    axes[0, 0].legend(loc="upper right", frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"Frozen 5 Hz forward model at the true source: {name}")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / f"{name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Existing diagnostic; refusing to overwrite: {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    if (hash_file(ROOT / protocol["checkpoint"]) != protocol["checkpoint_sha256"]
            or hash_file(ROOT / protocol["velocity"]) != protocol["velocity_sha256"]
            or hash_file(PROTOCOL) != results["protocol_sha256"]
            or len(results["events"]) != 8):
        raise ValueError("Frozen inputs or prior independent results have changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(ROOT / protocol["checkpoint"], device)
    model.eval().requires_grad_(False)
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    rows = []
    training = metadata["training_record"]["receiver_observations"]
    if len(training) != 29:
        raise ValueError("Expected frozen 29-event training manifest")
    for entry in training:
        path = Path(entry["path"])
        row, traces = audit_record(model, path, device, scale, "training", entry["sha256"])
        rows.append(row)
        if path.name == "train_04_0to2s.npz":
            plot_traces("training_04", traces)
        print(f"training {path.name}: L2={row['raw_relative_l2']:.3f}, "
              f"FMM picks={row['fd_valid_picks']}", flush=True)
    for entry, previous in zip(protocol["events"], results["events"]):
        path = ROOT / protocol["data_directory"] / entry["file"]
        if previous["event"] != entry["file"] or previous["observation"] != str(path.resolve()):
            raise ValueError("Independent event pairing changed")
        row, traces = audit_record(model, path, device, scale, "spent_independent",
                                   previous["observation_sha256"])
        if row["fd_valid_picks"] != previous["fmm"]["valid_fd_picks"]:
            raise ValueError("FMM pick counts differ from independent run")
        if not np.isclose(row["raw_relative_l2"],
                          previous["pinn"]["truth_source_raw_waveform_relative_l2"], atol=1e-5):
            raise ValueError("PINN forward prediction differs from independent run")
        rows.append(row)
        if path.name in ("test_04.npz", "test_05.npz"):
            plot_traces(path.stem, traces)
        print(f"spent {path.name}: L2={row['raw_relative_l2']:.3f}, "
              f"FMM picks={row['fd_valid_picks']}, paired={row['paired_picks']}", flush=True)
    report = {"role": "post-hoc forward diagnostic on all training and already-spent test events",
              "protocol_sha256": hash_file(PROTOCOL), "checkpoint_sha256": protocol["checkpoint_sha256"],
              "result_sha256": hash_file(RESULTS),
              "methods": "true-source forward; receiver first arrivals use FMM's fixed 5%-of-trace, "
                         "3-sample, 1%-of-record and <=1.85 s gates; peak ratio and "
                         "per-receiver <=0.2 s shift/gain are diagnostic only",
              "events": rows}
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {OUTPUT} and {PLOTS}", flush=True)


if __name__ == "__main__":
    main()
