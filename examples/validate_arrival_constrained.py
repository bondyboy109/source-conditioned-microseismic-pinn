"""Select a picker on development records; evaluate a frozen rule on new FD events."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_fmm_traveltime import first_arrival
from examples.invert_fmm_receiver_fields import interpolate_fields, receiver_fields, spatial_grid
from examples.run_holdout_noise_picker_comparison import robust_arrival
from mixture_of_experts import PrototypeGatedResidualMixtureOfExperts
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/arrival_constrained_fresh_test_protocol.json"
DOMAIN = np.array([16000.0, 16000.0, 3720.0, 9.0], dtype=np.float32)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_protocol() -> dict:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for key, hash_key in (("model_checkpoint", "model_sha256"),
                          ("velocity", "velocity_sha256"),
                          ("receivers", "receivers_sha256")):
        if digest(ROOT / spec[key]) != spec[hash_key]:
            raise ValueError(f"Registered input changed: {key}")
    return spec


def event_files(spec: dict, stage: str) -> list[Path]:
    if stage == "development":
        return [path for group in spec["development_groups"]
                for path in sorted((ROOT / group).glob("test_*.npz"))]
    return [ROOT / spec["test_directory"] / (event["id"] + ".npz")
            for event in spec["test_events"]]


def pick_receivers(data: np.ndarray, times: np.ndarray, picker: str,
                   spec: dict) -> np.ndarray:
    if picker == "threshold_5pct":
        picks = np.asarray([first_arrival(trace, times) for trace in data])
    elif picker == "robust_mad_6sigma":
        picks = np.asarray([robust_arrival(trace, times) for trace in data])
    else:
        raise ValueError(f"Unknown picker: {picker}")
    peaks = np.max(np.abs(data), axis=1)
    valid = (np.isfinite(picks) & (peaks >= spec["min_receiver_peak_fraction"] * peaks.max())
             & (picks <= times[-1] - spec["end_margin_s"]))
    picks[~valid] = np.nan
    return picks


def source_search(fields: np.ndarray, picks: np.ndarray, spec: dict):
    """Source search uses picks only; truth is read solely for post-search scoring."""
    valid = np.isfinite(picks)
    if valid.sum() < spec["min_valid_picks"]:
        return None
    bounds = np.asarray(spec["source_bounds_m"], dtype=np.float64)
    spacing = float(spec["fmm_spacing_m"])
    coarse = spatial_grid(bounds, tuple(spec["coarse_grid"]))
    travel = interpolate_fields(fields[valid], coarse, spacing)
    origin = np.clip(np.median(picks[valid][None, :] - travel, axis=1),
                     *spec["origin_bounds_s"])
    rmse = np.sqrt(np.mean((travel + origin[:, None] - picks[valid]) ** 2, axis=1))
    center = coarse[int(np.argmin(rmse))]
    offsets = np.asarray(list(itertools.product((-1.0, 0.0, 1.0), repeat=3)))
    local = np.clip(center[None, :] + offsets * np.asarray(spec["local_step_m"]),
                    bounds[:, 0], bounds[:, 1])
    local = np.unique(np.concatenate((local, center[None, :])), axis=0)
    local_travel = interpolate_fields(fields[valid], local, spacing)
    local_origin = np.clip(np.median(picks[valid][None, :] - local_travel, axis=1),
                           *spec["origin_bounds_s"])
    arrival_rmse = np.sqrt(np.mean((local_travel + local_origin[:, None] - picks[valid]) ** 2, axis=1))
    return np.column_stack((local, local_origin)), arrival_rmse, int(np.argmin(rmse))


def load_model(spec: dict, device: torch.device):
    base, _ = load_conditioned_checkpoint(ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth", device)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    checkpoint = torch.load(ROOT / spec["model_checkpoint"], map_location=device, weights_only=False)
    architecture = checkpoint["spec"]
    model = PrototypeGatedResidualMixtureOfExperts(
        base, expert_count=int(architecture["expert_count"]),
        hidden_dim=int(architecture["expert_hidden_dim"]),
        hidden_layers=int(architecture["expert_hidden_layers"]),
        feature_scale=float(architecture["feature_scale"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def waveform_losses(model, sources: np.ndarray, data: np.ndarray,
                    times: np.ndarray, receivers: np.ndarray, device: torch.device) -> np.ndarray:
    downsampled = data[::2, ::8]
    sampled_times = times[::8]
    locations = (receivers[::2] / DOMAIN[:3]).astype(np.float32)
    points = np.concatenate((
        np.broadcast_to(locations[:, None, :], (*downsampled.shape, 3)),
        np.broadcast_to((sampled_times / DOMAIN[3])[None, :, None], (*downsampled.shape, 1)),
    ), axis=-1).reshape(-1, 4)
    x = torch.as_tensor(points.astype(np.float32), device=device)
    observed = torch.as_tensor(downsampled, device=device)
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    if rms <= 0:
        raise ValueError("Empty observed waveform")
    values = []
    with torch.inference_mode():
        for source in sources:
            src = torch.as_tensor((source / DOMAIN).astype(np.float32), device=device)
            prediction = model(x, src.expand(len(points), -1), freeze_base=True)[0]
            values.append(float(((prediction.reshape(downsampled.shape) - observed) / rms).square().mean()))
    return np.asarray(values)


def score_choice(source: np.ndarray, truth: np.ndarray) -> dict:
    delta = source - truth
    return {"estimated_source_m_s": source.tolist(),
            "component_error_m_s": delta.tolist(),
            "location_error_m": float(np.linalg.norm(delta[:3])),
            "origin_time_error_s": float(abs(delta[3]))}


def run_event(path: Path, fields, model, picker, spec, device, alphas) -> dict:
    started = time.perf_counter()
    with np.load(path) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float64)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
    if not np.array_equal(receivers, np.load(ROOT / spec["receivers"])):
        raise ValueError(f"Receiver layout differs: {path}")
    picks = pick_receivers(data, times, picker, spec)
    result = {"observation": str(path), "observation_sha256": digest(path),
              "picker": picker,
              "valid_picks": int(np.isfinite(picks).sum())}
    search = source_search(fields, picks, spec)
    if search is None:
        with np.load(path) as archive:
            result["source_truth_m_s_post_search"] = np.asarray(
                archive["source_truth"], dtype=np.float64).tolist()
        result.update({"status": "insufficient_valid_picks", "methods": {},
                       "runtime_seconds": time.perf_counter() - started})
        return result
    sources, arrival_rmse, _ = search
    wave = waveform_losses(model, sources, data, times, receivers, device)
    arrival_loss = (arrival_rmse / spec["arrival_scale_s"]) ** 2
    branches = {"arrival": arrival_rmse, "waveform": wave}
    branches.update({f"hybrid_{alpha:g}": wave + alpha * arrival_loss for alpha in alphas})
    chosen = {name: int(np.argmin(scores)) for name, scores in branches.items()}
    with np.load(path) as archive:
        truth = np.asarray(archive["source_truth"], dtype=np.float64)
    result["source_truth_m_s_post_search"] = truth.tolist()
    methods = {}
    for name, scores in branches.items():
        idx = chosen[name]
        methods[name] = {**score_choice(sources[idx], truth),
                         "waveform_loss": float(wave[idx]), "arrival_rmse_s": float(arrival_rmse[idx]),
                         "selection_score": float(scores[idx])}
    result.update({"status": "ok", "candidate_count": len(sources),
                   "methods": methods, "runtime_seconds": time.perf_counter() - started})
    return result


def summarize(rows: list[dict], method: str) -> dict:
    successes = [row["methods"][method] for row in rows if row["status"] == "ok"]
    return {"event_count": len(rows), "failure_count": len(rows) - len(successes),
            "mean_location_error_m": float(np.mean([r["location_error_m"] for r in successes])) if successes else None,
            "median_location_error_m": float(np.median([r["location_error_m"] for r in successes])) if successes else None,
            "mean_origin_time_error_s": float(np.mean([r["origin_time_error_s"] for r in successes])) if successes else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("development", "test"))
    args = parser.parse_args()
    spec = load_protocol()
    output = ROOT / spec["development_result" if args.stage == "development" else "test_result"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    decision = ROOT / spec["development_result"]
    if args.stage == "test":
        selected = json.loads(decision.read_text(encoding="utf-8"))
        if selected["protocol_sha256"] != digest(PROTOCOL):
            raise ValueError("Protocol changed after development selection")
        manifest = json.loads((ROOT / "experiments/arrival_constrained_fresh_test_manifest.json")
                              .read_text(encoding="utf-8"))
        if (manifest["protocol_sha256"] != digest(PROTOCOL)
                or manifest["development_decision_sha256"] != digest(decision)):
            raise ValueError("Frozen protocol or development decision changed after FD generation")
        picker_alpha = [(selected["selected"]["picker"], selected["selected"]["alpha"])]
    else:
        picker_alpha = [(picker, alpha) for picker in spec["picker_candidates"]
                        for alpha in spec["alpha_candidates"]]
    paths = event_files(spec, args.stage)
    expected = 16 if args.stage == "development" else len(spec["test_events"])
    if len(paths) != expected or any(not path.is_file() for path in paths):
        raise ValueError(f"Expected {expected} existing {args.stage} FD records")
    if args.stage == "test":
        if len(manifest["events"]) != expected:
            raise ValueError("FD manifest event count does not match locked protocol")
        for path, event, registered in zip(paths, spec["test_events"], manifest["events"]):
            if (registered["id"] != event["id"] or digest(path) != registered["sha256"]
                    or digest(path.with_suffix(".json")) != registered["fd_report_sha256"]):
                raise ValueError(f"FD observation changed after registration: {path}")
            with np.load(path) as archive:
                if not np.allclose(archive["source_truth"], event["source_m_s"], atol=1e-5):
                    raise ValueError(f"Test source does not match protocol: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(spec, device)
    velocity = np.asarray(np.load(ROOT / spec["velocity"], mmap_mode="r")
                          [::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"], ::spec["fmm_velocity_stride"]],
                          dtype=np.float32)
    setup_start = time.perf_counter()
    fields = receiver_fields(velocity, np.load(ROOT / spec["receivers"]), float(spec["fmm_spacing_m"]))
    setup_seconds = time.perf_counter() - setup_start
    all_results = {}
    for picker in dict.fromkeys(item[0] for item in picker_alpha):
        alphas = [alpha for p, alpha in picker_alpha if p == picker]
        rows = []
        for path in paths:
            row = run_event(path, fields, model, picker, spec, device, alphas)
            rows.append(row)
            print(f"{path.name} {picker}: {row['status']} ({row['valid_picks']} picks)", flush=True)
        all_results[picker] = {"events": rows, "summary": {name: summarize(rows, name)
                               for name in ("arrival", "waveform", *(f"hybrid_{a:g}" for a in alphas))}}
    result = {"stage": args.stage, "protocol_sha256": digest(PROTOCOL),
              "model_sha256": spec["model_sha256"], "velocity_sha256": spec["velocity_sha256"],
              "fmm_setup_seconds": setup_seconds, "fmm_cache_mib": fields.nbytes / 1048576,
              "results": all_results}
    if args.stage == "development":
        selected = min(picker_alpha, key=lambda pair: (
            all_results[pair[0]]["summary"][f"hybrid_{pair[1]:g}"]["failure_count"],
            all_results[pair[0]]["summary"][f"hybrid_{pair[1]:g}"]["mean_location_error_m"] or float("inf"),
            all_results[pair[0]]["summary"][f"hybrid_{pair[1]:g}"]["mean_origin_time_error_s"] or float("inf")))
        result["selected"] = {"picker": selected[0], "alpha": selected[1]}
    else:
        result["selected"] = selected["selected"]
        result["development_decision_sha256"] = digest(decision)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selected": result["selected"],
                      "summary": {p: r["summary"] for p, r in all_results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
