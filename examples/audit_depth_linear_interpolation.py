"""Compare a local linear FD depth interpolation baseline with held-out offsets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import hash_file

PROTOCOL = ROOT / "experiments/depth_linear_interpolation_protocol.json"


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        return (np.asarray(archive["data"], dtype=np.float64),
                np.asarray(archive["times"], dtype=np.float64),
                np.asarray(archive["receiver_coords"], dtype=np.float64),
                np.asarray(archive["source_truth"], dtype=np.float64))


def metrics(fd: np.ndarray, prediction: np.ndarray,
            anchor_fd: np.ndarray) -> dict:
    energy = float(np.sum(fd ** 2))
    if energy <= 0:
        raise ValueError("Empty FD record")
    active = np.abs(fd) >= .05 * np.max(np.abs(fd))
    fd_change = fd - anchor_fd
    pred_change = prediction - anchor_fd
    fd_change_norm = np.linalg.norm(fd_change)
    pred_change_norm = np.linalg.norm(pred_change)
    if fd_change_norm == 0:
        response_l2 = 0.0 if pred_change_norm == 0 else float("inf")
        response_cosine = 1.0 if pred_change_norm == 0 else 0.0
    else:
        response_l2 = float(np.linalg.norm(pred_change - fd_change) / fd_change_norm)
        response_cosine = float(np.sum(pred_change * fd_change)
                                / (pred_change_norm * fd_change_norm))
    return {
        "full_l2": float(np.sqrt(np.sum((prediction - fd) ** 2) / energy)),
        "active_l2": float(np.sqrt(np.sum((prediction - fd)[active] ** 2)
                                 / np.sum(fd[active] ** 2))),
        "response_l2": response_l2,
        "response_cosine": response_cosine,
        "zero_lag_correlation": float(np.sum(prediction * fd)
                                       / (np.linalg.norm(prediction) * np.linalg.norm(fd))),
    }


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["result_file"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    depth_protocol = json.loads((ROOT / spec["depth_protocol"]).read_text())
    depth_results = json.loads((ROOT / spec["depth_results"]).read_text())
    coverage = json.loads((ROOT / spec["coverage33_results"]).read_text())
    if (hash_file(ROOT / spec["depth_results"]) != spec["depth_results_sha256"]
            or hash_file(ROOT / spec["coverage33_results"]) != spec["coverage33_results_sha256"]
            or depth_results["protocol_sha256"] != hash_file(ROOT / spec["depth_protocol"])
            or len(depth_results["events"]) != 12):
        raise ValueError("Archived depth inputs changed")
    events = {row["file"]: row for row in depth_results["events"]}
    roots = ROOT / "data/observations/eikonal_depth_sweep"
    anchors = {row["name"]: row for row in depth_protocol["anchors"]}
    rows = []
    for anchor_name, anchor in anchors.items():
        anchor_path = ROOT / anchor["observation"]
        anchor_fd, times, receivers, anchor_source = load(anchor_path)
        if not np.allclose(anchor_source, anchor["source_m_s"], rtol=0, atol=1e-5):
            raise ValueError(f"Anchor source mismatch: {anchor_path}")
        source_records = {0: (anchor_fd, anchor_path)}
        for offset in (-160, 160):
            item = events[f"{anchor_name}_z{'m' if offset < 0 else 'p'}{abs(offset)}.npz"]
            path = roots / item["file"]
            fd, candidate_times, candidate_receivers, source = load(path)
            if (hash_file(path) != item["sha256"]
                    or not np.array_equal(candidate_times, times)
                    or not np.array_equal(candidate_receivers, receivers)
                    or not np.isclose(source[2], anchor_source[2] + offset, atol=1e-5)):
                raise ValueError(f"Training-depth record mismatch: {path}")
            source_records[offset] = (fd, path)
        for target_offset in (-80, 80):
            item = events[f"{anchor_name}_z{'m' if target_offset < 0 else 'p'}{abs(target_offset)}.npz"]
            target_path = roots / item["file"]
            target, target_times, target_receivers, target_source = load(target_path)
            if (hash_file(target_path) != item["sha256"]
                    or not np.array_equal(target_times, times)
                    or not np.array_equal(target_receivers, receivers)
                    or not np.isclose(target_source[2], anchor_source[2] + target_offset, atol=1e-5)):
                raise ValueError(f"Target-depth record mismatch: {target_path}")
            low = source_records[-160 if target_offset < 0 else 0][0]
            high = source_records[0 if target_offset < 0 else 160][0]
            prediction = .5 * low + .5 * high
            nearest = low if target_offset < 0 else high
            row = {"event": item["file"], "sha256": item["sha256"],
                   "anchor": anchor_name, "target_offset_m": target_offset,
                   "interpolation": {"low_offset_m": -160 if target_offset < 0 else 0,
                                      "high_offset_m": 0 if target_offset < 0 else 160,
                                      "weights": [0.5, 0.5]},
                   "interpolated": metrics(target, prediction, anchor_fd),
                   "nearest_training_record": metrics(target, nearest, anchor_fd)}
            rows.append(row)
            print(f"{item['file']}: interpolation L2={row['interpolated']['full_l2']:.3f}; "
                  f"nearest={row['nearest_training_record']['full_l2']:.3f}", flush=True)
    summary = {}
    for label in ("interpolated", "nearest_training_record"):
        summary[label] = {metric: float(np.mean([row[label][metric] for row in rows]))
                          for metric in ("full_l2", "active_l2", "response_l2",
                                         "response_cosine", "zero_lag_correlation")}
    coverage_groups = coverage["groups"]
    summary["registered_neural_reference"] = {
        "baseline29_mean_full_l2": coverage_groups["offset_80"]["baseline"]["mean_full_l2"],
        "augmented33_mean_full_l2": coverage_groups["offset_80"]["augmented"]["mean_full_l2"],
        "baseline29_mean_active_l2": coverage_groups["offset_80"]["baseline"]["mean_active_l2"],
        "augmented33_mean_active_l2": coverage_groups["offset_80"]["augmented"]["mean_active_l2"],
    }
    report = {"role": spec["role"], "protocol_sha256": hash_file(PROTOCOL),
              "training_offsets_m": spec["training_offsets_m"],
              "interpolation_offsets_m": spec["interpolation_offsets_m"],
              "summary": summary, "events": rows, "limits": spec["limits"]}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
