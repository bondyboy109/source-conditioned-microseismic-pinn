"""Compare raw frozen-PINN receiver records with FD references.

The selected events are previously inspected development cases.  All primary
metrics use the raw PINN output at the true source, without time shifts or
amplitude fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_forward_error_decomposition import predict


DEFAULT_CHECKPOINT = ROOT / "outputs/depth_gap_25_hybrid_1000/conditioned_5Hz.pth"
DEFAULT_EVENTS = (
    ROOT / "data/observations/cross_region_fmm/region_06.npz",
    ROOT / "data/observations/cross_region_fmm/region_03.npz",
    ROOT / "data/observations/cross_region_fmm/region_08.npz",
)


def relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(prediction - reference) / denominator)


def correlation(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = reference.ravel() - float(np.mean(reference))
    pred = prediction.ravel() - float(np.mean(prediction))
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(pred))
    return float(np.dot(ref, pred) / denominator) if denominator else float("nan")


def active_window(reference: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return a contiguous time window containing meaningful FD energy."""
    energy_by_time = np.sum(reference**2, axis=0)
    active = np.flatnonzero(energy_by_time >= float(energy_by_time.max()) * 1e-4)
    if not len(active):
        raise ValueError("FD record has no nonzero active window")
    mask = np.zeros_like(times, dtype=bool)
    mask[active[0] : active[-1] + 1] = True
    return mask, float(times[active[0]]), float(times[active[-1]])


def event_metrics(reference: np.ndarray, prediction: np.ndarray,
                  times: np.ndarray, receivers: np.ndarray) -> dict:
    mask, start, stop = active_window(reference, times)
    receiver_energy = np.sum(reference**2, axis=1)
    receiver_index = int(np.argmax(receiver_energy))
    observed_energy = float(np.sum(reference**2))
    predicted_energy = float(np.sum(prediction**2))
    peak = float(np.max(np.abs(reference)))
    return {
        "full_record_relative_l2": relative_l2(reference, prediction),
        "active_window_s": [start, stop],
        "active_window_definition": (
            "contiguous first-to-last samples where summed FD receiver energy "
            "is at least 1e-4 of its temporal maximum"
        ),
        "active_window_relative_l2": relative_l2(
            reference[:, mask], prediction[:, mask]),
        "prediction_to_fd_peak_amplitude_ratio": float(
            np.max(np.abs(prediction)) / max(peak, 1e-30)),
        "zero_lag_correlation": correlation(reference, prediction),
        "prediction_to_fd_integrated_energy_ratio": (
            predicted_energy / observed_energy),
        "representative_receiver_selection": "maximum FD integrated energy",
        "representative_receiver_index_zero_based": receiver_index,
        "representative_receiver_xyz_m": receivers[receiver_index].tolist(),
        "representative_trace_relative_l2": relative_l2(
            reference[receiver_index], prediction[receiver_index]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--events", type=Path, nargs="+", default=list(DEFAULT_EVENTS))
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    if len(args.events) != 3:
        parser.error("This fixed diagnostic figure requires exactly three events")

    rows = []
    archive = {}
    for event_path in args.events:
        fd, pinn, times, source = predict(args.checkpoint, event_path, args.chunk_size)
        with np.load(event_path) as event:
            receivers = np.asarray(event["receiver_coords"], dtype=np.float64)
        if not np.isfinite(fd).all() or not np.isfinite(pinn).all():
            raise ValueError(f"Nonfinite values in {event_path}")
        metrics = event_metrics(fd, pinn, times, receivers)
        rows.append({
            "event": event_path.stem,
            "observation": str(event_path.resolve()),
            "observation_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
            "source_truth_xyzt": source.tolist(),
            "receiver_count": int(fd.shape[0]),
            "sample_count": int(fd.shape[1]),
            **metrics,
            "fd": fd,
            "pinn": pinn,
            "times": times,
            "receivers": receivers,
        })
        key = event_path.stem
        archive[f"{key}_fd"] = fd.astype(np.float32)
        archive[f"{key}_pinn"] = pinn.astype(np.float32)
        archive[f"{key}_times_s"] = times.astype(np.float32)
        archive[f"{key}_receivers_m"] = receivers.astype(np.float32)
        archive[f"{key}_source_truth_xyzt"] = source.astype(np.float32)

    fig, axes = plt.subplots(3, 4, figsize=(14.2, 8.3), constrained_layout=True)
    image_titles = ("FD reference", "Frozen PINN", "PINN - FD")
    for row_index, row in enumerate(rows):
        fd, pinn, residual = row["fd"], row["pinn"], row["pinn"] - row["fd"]
        time_extent = [float(row["times"][0]), float(row["times"][-1]),
                       fd.shape[0] - 0.5, -0.5]
        common_limit = max(float(np.max(np.abs(fd))), float(np.max(np.abs(pinn))), 1e-30)
        residual_limit = max(float(np.max(np.abs(residual))), 1e-30)
        for column, (values, limit) in enumerate(
                ((fd, common_limit), (pinn, common_limit), (residual, residual_limit))):
            axis = axes[row_index, column]
            image = axis.imshow(values, aspect="auto", interpolation="nearest",
                                extent=time_extent, cmap="seismic", vmin=-limit, vmax=limit,
                                rasterized=True)
            axis.set_xlim(float(row["times"][0]), float(row["times"][-1]))
            if row_index == 0:
                axis.set_title(image_titles[column])
            if column == 0:
                source = row["source_truth_xyzt"]
                axis.set_ylabel(
                    f"{row['event']}\nreceiver index\n"
                    f"source=({source[0]/1000:.2f}, {source[1]/1000:.2f}, "
                    f"{source[2]/1000:.2f}) km")
            if row_index == 2:
                axis.set_xlabel("Time (s)")
            colorbar = fig.colorbar(image, ax=axis, fraction=0.047, pad=0.02)
            colorbar.ax.tick_params(labelsize=7)

        trace_axis = axes[row_index, 3]
        receiver_index = row["representative_receiver_index_zero_based"]
        scale = max(float(np.max(np.abs(fd[receiver_index]))), 1e-30)
        trace_axis.plot(row["times"], fd[receiver_index] / scale,
                        color="black", linewidth=1.15, label="FD")
        trace_axis.plot(row["times"], pinn[receiver_index] / scale,
                        color="#d95f02", linewidth=1.0, label="PINN")
        trace_axis.plot(row["times"], residual[receiver_index] / scale,
                        color="#1b9e77", linewidth=0.9, linestyle="--", label="Residual")
        trace_axis.axhline(0.0, color="0.65", linewidth=0.6)
        trace_axis.grid(alpha=0.2, linewidth=0.5)
        trace_axis.set_xlim(float(row["times"][0]), float(row["times"][-1]))
        trace_axis.set_ylabel("Amplitude / FD peak")
        trace_axis.text(
            0.02, 0.96,
            f"R{receiver_index}; trace $L_2$={row['representative_trace_relative_l2']:.3f}",
            transform=trace_axis.transAxes, va="top", fontsize=8)
        if row_index == 0:
            trace_axis.set_title("Maximum-energy receiver")
            trace_axis.legend(loc="upper right", frameon=False, fontsize=8)
        if row_index == 2:
            trace_axis.set_xlabel("Time (s)")

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.traces.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(args.traces, **archive)

    serializable_rows = []
    for row in rows:
        serializable_rows.append({key: value for key, value in row.items()
                                  if key not in {"fd", "pinn", "times", "receivers"}})
    report = {
        "purpose": "representative development-event PINN-FD receiver-record comparison",
        "status": "previously inspected development diagnostic; not independent confirmation",
        "frequency_hz": 5.0,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "primary_metrics_use_raw_predictions": True,
        "amplitude_gain_fitted": False,
        "time_shift_fitted": False,
        "events": serializable_rows,
    }
    args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in serializable_rows:
        print(
            f"{row['event']}: full L2={row['full_record_relative_l2']:.4f}, "
            f"active L2={row['active_window_relative_l2']:.4f}, "
            f"peak={row['prediction_to_fd_peak_amplitude_ratio']:.3f}, "
            f"corr={row['zero_lag_correlation']:.3f}, "
            f"energy={row['prediction_to_fd_integrated_energy_ratio']:.3f}",
            flush=True,
        )
    print(f"Saved {args.figure}, {args.metrics}, and {args.traces}")


if __name__ == "__main__":
    main()
