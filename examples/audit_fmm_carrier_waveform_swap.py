"""Post-hoc frozen-waveform-head response to substituted FMM travel times."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_wide_search_independent_protocol.json"
CARRIER = ROOT / "experiments/eikonal_wide_search_carrier_vs_fmm.json"
OUTPUT = ROOT / "experiments/eikonal_wide_search_fmm_carrier_swap.json"


def frozen_head(model, points: torch.Tensor, source: torch.Tensor,
                travel_s: torch.Tensor) -> torch.Tensor:
    """Mirror the locked carrier architecture; baseline checked against model.forward."""
    relative_xyz = points[:, :3] - source[:, :3]
    delayed_time = points[:, 3:4] - source[:, 3:4] - travel_s / model.time_length_s
    features = torch.cat((relative_xyz, delayed_time, source[:, :3],
                          travel_s / model.time_length_s), dim=-1)
    amplitude = model.net(features)
    phase = 2.0 * math.pi * model.frequency_hz * model.time_length_s * delayed_time
    return model.output_scale * (amplitude[:, :1] * torch.cos(phase)
                                 + amplitude[:, 1:] * torch.sin(phase))


def evaluate(model, path: Path, source: np.ndarray,
             fmm_s: np.ndarray, scale: np.ndarray) -> dict:
    with np.load(path) as archive:
        fd = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receiver = np.asarray(archive["receiver_coords"], dtype=np.float32)
        if not np.allclose(archive["source_truth"], source, rtol=0, atol=1e-5):
            raise ValueError(f"Source differs from registered FD event: {path}")
    if fd.shape != (25, 1001) or fmm_s.shape != (25,):
        raise ValueError("Unexpected receiver geometry or FMM carrier")
    device = next(model.parameters()).device
    source_t = torch.as_tensor(source.astype(np.float32) / scale, device=device)
    receiver_t = torch.as_tensor(receiver / scale[:3], device=device)
    times_t = torch.as_tensor(times / scale[3], device=device)
    travel_t = torch.as_tensor(fmm_s.astype(np.float32), device=device)
    baseline = np.empty_like(fd)
    swapped = np.empty_like(fd)
    with torch.inference_mode():
        for station in range(len(receiver)):
            points = torch.cat((receiver_t[station].expand(len(times), 3),
                                times_t[:, None]), dim=1)
            src = source_t.expand(len(times), 4)
            original_travel = model._travel_time(points[:, :3] - src[:, :3], src)
            original = frozen_head(model, points, src, original_travel).squeeze(-1)
            direct = model(points, src).squeeze(-1)
            if not torch.allclose(original, direct, atol=1e-12, rtol=1e-5):
                raise ValueError("Diagnostic head no longer matches frozen model.forward")
            baseline[station] = original.cpu().numpy()
            swapped[station] = frozen_head(model, points, src,
                                           travel_t[station].expand(len(times), 1)
                                           ).squeeze(-1).cpu().numpy()
    observed = fd.astype(np.float64)
    base = baseline.astype(np.float64)
    substituted = swapped.astype(np.float64)
    fd_norm = np.linalg.norm(observed)
    active = np.abs(observed) >= .05 * np.max(np.abs(observed))
    return {"baseline_relative_l2": float(np.linalg.norm(base - observed) / fd_norm),
            "fmm_swap_relative_l2": float(np.linalg.norm(substituted - observed) / fd_norm),
            "baseline_active_relative_l2": float(np.linalg.norm((base - observed)[active]) /
                                                 np.linalg.norm(observed[active])),
            "fmm_swap_active_relative_l2": float(np.linalg.norm(
                (substituted - observed)[active]) / np.linalg.norm(observed[active])),
            "baseline_vs_swap_relative_l2": float(np.linalg.norm(substituted - base) / fd_norm)}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Existing post-hoc diagnostic: {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    previous = json.loads(CARRIER.read_text(encoding="utf-8"))
    if (previous["protocol_sha256"] != hash_file(PROTOCOL)
            or previous["checkpoint_sha256"] != protocol["checkpoint_sha256"]
            or previous["velocity_sha256"] != protocol["velocity_sha256"]
            or len(previous["events"]) != 37):
        raise ValueError("Prior carrier-vs-FMM data or frozen inputs changed")
    model, metadata = load_conditioned_checkpoint(
        ROOT / protocol["checkpoint"],
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().requires_grad_(False)
    if (metadata["architecture"]["kind"] != "eikonal_travel_time_carrier"
            or metadata["architecture"].get("onset_gate") is not None
            or metadata["architecture"].get("hard_initial", False)):
        raise ValueError("Diagnostic mirrors only the locked ungated eikonal architecture")
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    training = metadata["training_record"]["receiver_observations"]
    paths = [Path(row["path"]) for row in training]
    paths += [ROOT / protocol["data_directory"] / row["file"]
              for row in protocol["events"]]
    rows = []
    for path, entry in zip(paths, previous["events"]):
        if hash_file(path) != entry["observation_sha256"] or path.name != entry["event"]:
            raise ValueError(f"Frozen observation pairing changed: {path}")
        with np.load(path) as archive:
            source = np.asarray(archive["source_truth"], dtype=np.float64)
        result = evaluate(model, path, source,
                          np.asarray(entry["fmm_traveltime_s"], dtype=np.float64), scale)
        if not np.isclose(result["baseline_relative_l2"],
                          entry["raw_waveform_relative_l2_at_truth"], atol=2e-5, rtol=1e-5):
            raise ValueError(f"Locked original forward prediction changed: {path}")
        rows.append({"group": entry["group"], "event": entry["event"],
                     "observation_sha256": entry["observation_sha256"], **result})
        print(f"{entry['group']} {entry['event']}: raw L2 "
              f"{result['baseline_relative_l2']:.3f} -> "
              f"{result['fmm_swap_relative_l2']:.3f}", flush=True)
    report = {"role": "post-hoc true-source FMM-carrier intervention on frozen waveform head",
              "protocol_sha256": hash_file(PROTOCOL),
              "previous_carrier_audit_sha256": hash_file(CARRIER),
              "checkpoint_sha256": protocol["checkpoint_sha256"],
              "fmm_spacing_m": previous["fmm_grid_spacing_m"],
              "interpretation_note": "Only the frozen head's travel_s phase/feature changes. "
                                     "No retraining, inversion or source gradient test. "
                                     "FMM has 80 m grid and snapped receiver sources; "
                                     "swapping an untrained carrier is not a trained alternative.",
              "events": rows}
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
