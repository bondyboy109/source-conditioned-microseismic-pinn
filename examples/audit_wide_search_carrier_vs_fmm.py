"""Compare the frozen eikonal carrier against reciprocal FMM on spent events."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.invert_fmm_receiver_fields import interpolate_fields, receiver_fields
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_wide_search_independent_protocol.json"
WAVEFORMS = ROOT / "experiments/eikonal_wide_search_waveform_diagnostic.json"
OUTPUT = ROOT / "experiments/eikonal_wide_search_carrier_vs_fmm.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Existing diagnostic must be preserved: {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    waveform = json.loads(WAVEFORMS.read_text(encoding="utf-8"))
    if (waveform["protocol_sha256"] != hash_file(PROTOCOL)
            or hash_file(ROOT / protocol["checkpoint"]) != protocol["checkpoint_sha256"]
            or hash_file(ROOT / protocol["velocity"]) != protocol["velocity_sha256"]
            or len(waveform["events"]) != 37):
        raise ValueError("Frozen inputs or prior waveform audit changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(ROOT / protocol["checkpoint"], device)
    model.eval().requires_grad_(False)
    if not hasattr(model, "eikonal_model"):
        raise ValueError("Checkpoint has no learned eikonal carrier")
    training = metadata["training_record"]["receiver_observations"]
    if (len(training) != 29 or [r["observation_sha256"] for r in waveform["events"][:29]]
            != [r["sha256"] for r in training]):
        raise ValueError("Training manifest disagrees with waveform audit")
    spent = waveform["events"][29:]
    if ([r["event"] for r in spent] != [r["file"] for r in protocol["events"]]
            or [r["group"] for r in spent] != ["spent_independent"] * 8):
        raise ValueError("Spent test event pairing changed")
    paths = ([Path(r["path"]) for r in training]
             + [ROOT / protocol["data_directory"] / r["file"]
                for r in protocol["events"]])
    sources = np.asarray([row["true_source_used_for_forward_diagnostic_only_m_s"]
                          for row in waveform["events"]], dtype=np.float64)
    known_train = np.asarray([row["source_truth"] for row in training], dtype=np.float64)
    if (not np.allclose(sources[:29], known_train, rtol=0, atol=1e-5)
            or not np.allclose(sources[29:], [r["source_m_s"] for r in protocol["events"]],
                               rtol=0, atol=1e-5)):
        raise ValueError("Recorded sources differ from registration")
    for path, record in zip(paths, waveform["events"]):
        if hash_file(path) != record["observation_sha256"]:
            raise ValueError(f"FD observation changed: {path}")
    with np.load(paths[0]) as archive:
        receiver_coords = np.asarray(archive["receiver_coords"], dtype=np.float64)
    for path in paths[1:]:
        with np.load(path) as archive:
            if not np.array_equal(archive["receiver_coords"], receiver_coords):
                raise ValueError(f"Receiver geometry changed: {path}")

    settings = json.loads((ROOT / protocol["fmm_settings"]).read_text(encoding="utf-8"))
    fmm = settings["fmm_baseline"]
    spacing_m = 40. * fmm["velocity_grid_stride"]
    original = np.load(ROOT / protocol["velocity"], mmap_mode="r")
    if original.shape != (401, 401, 94) or spacing_m != 80:
        raise ValueError("FMM reference grid differs from the fixed baseline")
    coarse = np.ascontiguousarray(original[::2, ::2, ::2], dtype=np.float32)
    started = time.perf_counter()
    fields = receiver_fields(coarse, receiver_coords, spacing_m)
    setup_s = time.perf_counter() - started
    # Reciprocal fields have their sources snapped to the nearest 80 m receiver cell.
    fmm_times = interpolate_fields(fields, sources[:, :3], spacing_m)
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m")], dtype=np.float32)
    repeat_receiver = np.tile(receiver_coords.astype(np.float32) / scale, (len(paths), 1))
    repeat_source = np.repeat(sources[:, :3].astype(np.float32) / scale,
                              len(receiver_coords), axis=0)
    with torch.inference_mode():
        neural_times = model.eikonal_model(
            torch.as_tensor(repeat_receiver, device=device),
            torch.as_tensor(repeat_source, device=device)).reshape(
                len(paths), len(receiver_coords)).cpu().numpy()
    rows = []
    for i, record in enumerate(waveform["events"]):
        observed = np.asarray(record["receivers"], dtype=object)
        picks = np.asarray([row["fd_valid_by_fmm_rule"] for row in observed], dtype=bool)
        if picks.sum() != record["fd_valid_picks"]:
            raise ValueError(f"Recorded valid station mask differs: {record['event']}")
        delta = neural_times[i].astype(np.float64) - fmm_times[i]
        nearest = float(np.min(np.linalg.norm(known_train[:, :3] - sources[i, :3], axis=1)))
        rows.append({
            "group": record["group"], "event": record["event"],
            "observation_sha256": record["observation_sha256"],
            "nearest_training_source_distance_m": nearest,
            "raw_waveform_relative_l2_at_truth": record["raw_relative_l2"],
            "valid_fd_receiver_count": int(picks.sum()),
            "all_25_eikonal_minus_fmm_mean_s": float(np.mean(delta)),
            "all_25_eikonal_vs_fmm_rmse_s": float(np.sqrt(np.mean(delta**2))),
            "all_25_demeaned_rmse_s": float(np.std(delta)),
            "valid_fd_eikonal_minus_fmm_mean_s": float(np.mean(delta[picks])),
            "valid_fd_eikonal_vs_fmm_rmse_s": float(np.sqrt(np.mean(delta[picks]**2))),
            "valid_fd_demeaned_rmse_s": float(np.std(delta[picks])),
            "fmm_traveltime_s": fmm_times[i].tolist(),
            "eikonal_traveltime_s": neural_times[i].tolist(),
        })
        print(f"{record['group']} {record['event']}: carrier-FMM "
              f"RMSE={rows[-1]['valid_fd_eikonal_vs_fmm_rmse_s']:.3f} s, "
              f"wave L2={record['raw_relative_l2']:.3f}", flush=True)
    report = {"role": "post-hoc carrier-vs-FMM diagnostic on 29 training + eight spent test events",
              "protocol_sha256": hash_file(PROTOCOL),
              "prior_waveform_diagnostic_sha256": hash_file(WAVEFORMS),
              "checkpoint_sha256": protocol["checkpoint_sha256"],
              "velocity_sha256": protocol["velocity_sha256"],
              "fmm_grid_spacing_m": spacing_m,
              "fmm_receiver_source_snap_m": (np.rint(receiver_coords / spacing_m) * spacing_m
                                              - receiver_coords).tolist(),
              "fmm_setup_seconds": setup_s,
              "fmm_receiver_field_cache_mib": fields.nbytes / 1048576,
              "interpretation_note": "80 m reciprocal FMM is approximate and includes receiver-cell "
                                     "snapping; carrier accuracy against FMM does not validate "
                                     "waveform amplitude, phase, multipathing or FD arrival picks",
              "events": rows}
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
