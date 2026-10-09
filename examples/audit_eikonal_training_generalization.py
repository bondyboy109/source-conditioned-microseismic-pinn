"""Compare a frozen eikonal-carrier model on training and spent FD sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/eikonal_independent_protocol.json"
OUTPUT = ROOT / "experiments/eikonal_training_generalization_audit.json"


def observation_groups(protocol: dict, training: list[dict]) -> dict[str, list[tuple[Path, str]]]:
    groups = {"training_29": [(Path(item["path"]), item["sha256"]) for item in training]}
    development = json.loads((ROOT / "experiments/cross_region_coverage_development_protocol.json").read_text())
    groups["near_source_development_4"] = [
        (ROOT / "data/observations/cross_region_coverage" / item["file"], "")
        for item in development["development_sources"]]
    groups["same_location_time_shift_3"] = [
        (ROOT / "data/observations/development_time_shift" /
         f"train01_t{stamp}_0to2s.npz", "") for stamp in ("016", "020", "028")]
    groups["spatial_shift_x_3"] = [
        (ROOT / "data/observations/development_spatial_radius" /
         f"train01_dx{offset:04d}_0to2s.npz", "") for offset in (40, 80, 160)]
    groups["orthogonal_source_development_10"] = [
        (ROOT / "data/observations/orthogonal_confirmation" / f"{anchor}_{direction}.npz", "")
        for anchor in ("a01", "a09")
        for direction in ("xp400", "ym120", "yp120", "zm080", "zp080")]
    earlier = json.loads((ROOT / "experiments/fmm_seeded_independent_results.json").read_text())
    groups["previously_inspected_development_8"] = [
        (ROOT / "data/observations/fmm_seeded_independent" / row["event"], row["sha256"])
        for row in earlier["events"]]
    spent = json.loads((ROOT / "experiments/eikonal_independent_results.json").read_text())
    if spent["eikonal_sha256"] != protocol["eikonal_sha256"]:
        raise ValueError("Original independent test used a different checkpoint")
    groups["spent_independent_8"] = [
        (ROOT / "data/observations/eikonal_independent" / row["event"], row["sha256"])
        for row in spent["events"]]
    expected = {"training_29": 29, "near_source_development_4": 4,
                "same_location_time_shift_3": 3, "spatial_shift_x_3": 3,
                "orthogonal_source_development_10": 10,
                "previously_inspected_development_8": 8, "spent_independent_8": 8}
    if any(len(groups[name]) != count for name, count in expected.items()):
        raise ValueError("A historical group lacks registered events")
    all_paths = [path.resolve() for rows in groups.values() for path, _ in rows]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("Some observation files are counted in multiple groups")
    return groups


def evaluate(model, scale: np.ndarray, path: Path, digest: str) -> dict:
    if not path.is_file() or (digest and hash_file(path) != digest):
        raise ValueError(f"Missing or changed FD observation: {path}")
    with np.load(path) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        source = np.asarray(archive["source_truth"], dtype=np.float32)
    if data.shape != (25, 1001) or len(times) != 1001 or not np.all(np.isfinite(data)):
        raise ValueError(f"Unexpected record geometry: {path}")
    if not np.allclose(np.diff(times), 0.002, atol=1e-5, rtol=0):
        raise ValueError(f"Unexpected time sampling: {path}")
    device = next(model.parameters()).device
    prediction = np.empty_like(data)
    normalized_source = torch.as_tensor(source / scale, device=device)
    # Keep inference batches small enough for both GPU and CPU use.
    with torch.inference_mode():
        for station, position in enumerate(receivers):
            xyzt = np.column_stack((np.broadcast_to(position, (len(times), 3)), times))
            prediction[station] = model(torch.as_tensor(xyzt / scale, device=device),
                                        normalized_source.expand(len(times), 4)).squeeze(-1).cpu().numpy()
    observed64 = data.astype(np.float64)
    predicted64 = prediction.astype(np.float64)
    energy = float(np.sum(observed64 ** 2))
    if energy <= 0 or not np.isfinite(predicted64).all():
        raise ValueError(f"Empty or nonfinite forward record: {path}")
    residual = predicted64 - observed64
    active = np.abs(observed64) >= 0.05 * np.max(np.abs(observed64))
    active_energy = float(np.sum(observed64[active] ** 2))
    return {"observation": str(path.resolve()), "sha256": hash_file(path),
            "source_m_s_post_hoc": source.tolist(),
            "full_record_relative_l2": float(np.sqrt(np.sum(residual ** 2) / energy)),
            "active_sample_relative_l2": float(np.sqrt(np.sum(residual[active] ** 2)
                                                        / active_energy)),
            "active_sample_count": int(np.sum(active)),
            "active_observed_energy_fraction": active_energy / energy,
            "predicted_to_observed_peak_ratio": float(np.max(np.abs(predicted64))
                                                       / np.max(np.abs(observed64))),
            "zero_lag_correlation": float(np.sum(predicted64 * observed64) /
                                          np.sqrt(np.sum(predicted64 ** 2) * energy))}


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    path = ROOT / protocol["eikonal_checkpoint"]
    if hash_file(path) != protocol["eikonal_sha256"]:
        raise ValueError("The frozen eikonal checkpoint changed")
    model, payload = load_conditioned_checkpoint(
        path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().requires_grad_(False)
    training = payload["training_record"]["receiver_observations"]
    groups = observation_groups(protocol, training)
    training_hashes = {item["sha256"] for item in training}
    scale = np.asarray([payload["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    training_sources = np.asarray([item["source_truth"] for item in training], dtype=np.float64)
    results = {}
    for name, entries in groups.items():
        rows = []
        for observation, digest in entries:
            row = evaluate(model, scale, observation, digest)
            if (row["sha256"] in training_hashes) != (name == "training_29"):
                raise ValueError(f"Wrong training group or content overlap: {observation}")
            source_xyz = np.asarray(row["source_m_s_post_hoc"][:3], dtype=np.float64)
            nearest = int(np.argmin(np.linalg.norm(training_sources[:, :3] - source_xyz,
                                                   axis=1)))
            row["nearest_training_source_distance_m"] = float(np.linalg.norm(
                training_sources[nearest, :3] - source_xyz))
            row["nearest_spatial_training_origin_time_difference_s"] = float(abs(
                training_sources[nearest, 3] - row["source_m_s_post_hoc"][3]))
            rows.append(row)
            print(f"{name}: {observation.name} L2={row['full_record_relative_l2']:.3f}, "
                  f"active={row['active_sample_relative_l2']:.3f}", flush=True)
        results[name] = {
            "count": len(rows),
            "mean_full_l2": float(np.mean([row["full_record_relative_l2"] for row in rows])),
            "median_full_l2": float(np.median([row["full_record_relative_l2"] for row in rows])),
            "mean_active_l2": float(np.mean([row["active_sample_relative_l2"] for row in rows])),
            "mean_peak_ratio": float(np.mean([row["predicted_to_observed_peak_ratio"] for row in rows])),
            "mean_zero_lag_correlation": float(np.mean([row["zero_lag_correlation"] for row in rows])),
            "mean_nearest_training_source_distance_m": float(np.mean([
                row["nearest_training_source_distance_m"] for row in rows])),
            "events": rows,
        }
    output = {"purpose": "post-hoc training-versus-spent-development comparison; no new independent validation",
              "checkpoint_sha256": protocol["eikonal_sha256"],
              "definitions": "Relative L2 = norm(predicted - FD) / norm(FD); active subset uses FD samples at least 5% of that event's global peak; source truth used only to measure forward prediction",
              "groups": results}
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
