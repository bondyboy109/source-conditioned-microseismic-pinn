"""Compare regenerated training event 04 with its frozen training record."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import hash_file

PROTOCOL = ROOT / "experiments/eikonal_wide_search_independent_protocol.json"
CHECKPOINT = ROOT / "outputs/eikonal_coverage_29_hybrid_1000/conditioned_5Hz.pth"
TRAIN = ROOT / "data/observations/window_0to2s/train_04_0to2s.npz"
PARENT = ROOT / "data/observations/train_04.npz"
REGENERATED = ROOT / "outputs/train_04_regeneration_control/regenerated_train_04.npz"
RESULT = ROOT / "experiments/train04_fd_regeneration_control.json"


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(f"Refusing to overwrite previous audit: {RESULT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (hash_file(CHECKPOINT) != protocol["checkpoint_sha256"]
            or hash_file(ROOT / protocol["velocity"]) != protocol["velocity_sha256"]
            or hash_file(ROOT / "examples/generate_fd_observation.py")
            != protocol["fd_generator_sha256"]):
        raise ValueError("Frozen checkpoint, velocity or FD generator changed")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    training = checkpoint["training_record"]["receiver_observations"]
    entry = next(row for row in training if Path(row["path"]).resolve() == TRAIN.resolve())
    if hash_file(TRAIN) != entry["sha256"] or len(training) != 29:
        raise ValueError("Training data changed since the frozen 29-event checkpoint")
    old_fd = json.loads(PARENT.with_suffix(".json").read_text(encoding="utf-8"))
    cropped_fd = json.loads(TRAIN.with_suffix(".json").read_text(encoding="utf-8"))
    new_fd = json.loads(REGENERATED.with_suffix(".json").read_text(encoding="utf-8"))
    comparable = ("method", "frequency_hz", "boundary", "source", "receiver_sampling", "cfl")
    if (any(old_fd[key] != new_fd[key] or cropped_fd[key] != new_fd[key]
            for key in comparable) or old_fd["duration_s"] != 9
            or cropped_fd["duration_s"] != 2 or new_fd["duration_s"] != 2):
        raise ValueError("FD generator provenance differs between records")
    with np.load(TRAIN) as original, np.load(PARENT) as parent, np.load(REGENERATED) as new:
        if (original["data"].shape != (25, 1001)
                or new["data"].shape != (25, 1001)
                or parent["data"].shape != (25, 4501)):
            raise ValueError("Unexpected FD waveform shape")
        metadata_equal = {key: bool(np.array_equal(original[key], new[key]))
                          for key in ("source_truth", "times", "receiver_coords")}
        if not all(metadata_equal.values()):
            raise ValueError("Regenerated source, time axis or receivers differ")
        parent_prefix_equal = all(np.array_equal(original[key], parent[key])
                                  for key in ("source_truth", "receiver_coords"))
        parent_prefix_equal &= bool(np.array_equal(original["times"], parent["times"][:1001]))
        parent_prefix_equal &= bool(np.array_equal(original["data"], parent["data"][:, :1001]))
        if not parent_prefix_equal:
            raise ValueError("Frozen training record differs from its nine-second FD parent")
        observed = original["data"].astype(np.float64)
        regenerated = new["data"].astype(np.float64)
        delta = regenerated - observed
        report = {
            "role": "post-hoc reproducibility control on an in-training source; not an independent test",
            "checkpoint_sha256": protocol["checkpoint_sha256"],
            "velocity_sha256": protocol["velocity_sha256"],
            "fd_generator_sha256": protocol["fd_generator_sha256"],
            "training_record_sha256": hash_file(TRAIN),
            "parent_record_sha256": hash_file(PARENT),
            "regenerated_record_sha256": hash_file(REGENERATED),
            "training_manifest_source_m_s": entry["source_truth"],
            "provenance_fields_equal": list(comparable),
            "regenerated_source_time_receivers_exact": metadata_equal,
            "training_is_exact_first_two_seconds_of_nine_second_parent": parent_prefix_equal,
            "sample_count": int(observed.size),
            "bitwise_equal_waveform_arrays": bool(np.array_equal(observed, regenerated)),
            "nonidentical_waveform_samples": int(np.count_nonzero(delta)),
            "relative_waveform_l2": float(np.linalg.norm(delta) / np.linalg.norm(observed)),
            "maximum_absolute_sample_difference": float(np.max(np.abs(delta))),
            "reference_waveform_peak_absolute": float(np.max(np.abs(observed))),
            "limitations": "Exact replay at one training source establishes this generator and crop "
                           "path, not equivalence for every historical training event or "
                           "accuracy at any unseen source. Do not retune on spent tests."
        }
    with RESULT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Exact waveform array equality: {report['bitwise_equal_waveform_arrays']}; "
          f"{report['nonidentical_waveform_samples']}/{report['sample_count']} different; "
          f"relative L2={report['relative_waveform_l2']:.6g}", flush=True)
    print(f"Saved {RESULT}", flush=True)


if __name__ == "__main__":
    main()
