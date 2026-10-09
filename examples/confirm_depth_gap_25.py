"""One-time registered FD forward and truth-independent regional inversion audit."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from invert_source_conditioned import load_observations, waveform_mse
from source_conditioned_pinn import load_conditioned_checkpoint


PROTOCOL = ROOT / "experiments/depth_gap_independent_confirmation_protocol.json"
DATA = ROOT / "data/observations/depth_gap_independent_confirmation"
RESULTS = ROOT / "experiments"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_and_verify(protocol: dict, training: list[dict]) -> list[Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    entries = protocol["events"]
    names = [item["file"] for item in entries]
    sources = [tuple(item["source_m_s"]) for item in entries]
    if len(set(names)) != len(names) or len(set(sources)) != len(sources):
        raise ValueError("Confirmation source filenames and coordinates must be unique")
    for item in entries:
        for previous in training:
            if np.allclose(item["source_m_s"], previous["source_truth"], atol=1e-4, rtol=0):
                raise ValueError("A confirmation source is a receiver-training source")
        if item["region"] not in (*protocol["preassigned_inversion_regions"], "remote"):
            raise ValueError(f"Unknown registered region: {item['region']}")
        if item["region"] != "remote":
            bounds = np.asarray(protocol["preassigned_inversion_regions"][item["region"]]["bounds_m_s"])
            if np.any(np.asarray(item["source_m_s"]) < bounds[:, 0]) or np.any(
                np.asarray(item["source_m_s"]) > bounds[:, 1]
            ):
                raise ValueError("Preassigned regional bounds do not include registered source")
    paths = []
    for item in entries:
        path = DATA / item["file"]
        if not path.exists():
            print(f"FD generating {item['file']}", flush=True)
            subprocess.run([sys.executable, str(ROOT / "examples/generate_fd_observation.py"),
                            "--duration", "2", "--output", str(path),
                            "--source", *(str(value) for value in item["source_m_s"])],
                           cwd=ROOT, check=True)
        report_path = path.with_suffix(".json")
        if not report_path.is_file():
            raise ValueError(f"Missing FD provenance: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report["frequency_hz"] != 5.0 or report["duration_s"] != 2.0
                or report["boundary"] != "zero Dirichlet on all six faces"):
            raise ValueError(f"FD generation parameters differ: {path}")
        with np.load(path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.allclose(archive["source_truth"], item["source_m_s"], atol=1e-5, rtol=0)
                    or not np.isfinite(archive["data"]).all()):
                raise ValueError(f"Unexpected source, shape or FD values: {path}")
        if sha256(path) in {row["sha256"] for row in training}:
            raise ValueError("Exact confirmation data appear in checkpoint training set")
        paths.append(path)
    return paths


def coarse_grid(bounds: np.ndarray, shape: list[int]) -> np.ndarray:
    if bounds.shape != (4, 2) or len(shape) != 4 or min(shape) < 2:
        raise ValueError("Expected four physical parameter ranges and 4-D grid")
    axes = [np.linspace(low, high, count, dtype=np.float32)
            for (low, high), count in zip(bounds, shape)]
    return np.asarray(list(itertools.product(*axes)), dtype=np.float32)


def coarse_search(model, receivers, times, observed, scale, bounds,
                  config, rms):
    candidates = coarse_grid(bounds, config["coarse_grid_xyzt"])
    xyz = torch.as_tensor(receivers[::config["coarse_receiver_stride"]] / scale[:3],
                          device=observed.device)
    time_t = torch.as_tensor(times[::config["coarse_time_stride"]] / scale[3],
                             device=observed.device)
    measured = torch.as_tensor(
        observed.cpu().numpy()[::config["coarse_receiver_stride"],
                               ::config["coarse_time_stride"]], device=observed.device)
    n_receiver, n_time = len(xyz), len(time_t)
    points = torch.cat((xyz[:, None, :].expand(n_receiver, n_time, 3),
                        time_t[None, :, None].expand(n_receiver, n_time, 1)), dim=-1)
    points = points.reshape(-1, 4)
    losses = []
    with torch.inference_mode():
        for start in range(0, len(candidates), 12):
            batch = torch.as_tensor(candidates[start:start+12] / scale,
                                    device=observed.device)
            prediction = model(points.repeat(len(batch), 1),
                               batch.repeat_interleave(len(points), dim=0))
            prediction = prediction.reshape(len(batch), n_receiver, n_time)
            values = (((prediction - measured[None, :, :]) / rms) ** 2).mean((1, 2))
            losses.extend(values.cpu().tolist())
    indices = np.argsort(losses, kind="stable")[:config["restarts"]]
    return [{"source_m_s": candidates[index].tolist(), "coarse_normalized_mse": losses[index]}
            for index in indices]


def refine(model, receivers, times, observed, scale, bounds, config,
           rms, candidates):
    xyz = torch.as_tensor(receivers[::config["refine_receiver_stride"]] / scale[:3],
                          device=observed.device)
    time_t = torch.as_tensor(times[::config["refine_time_stride"]] / scale[3],
                             device=observed.device)
    selected = observed[::config["refine_receiver_stride"],
                        ::config["refine_time_stride"]]
    low = torch.as_tensor(bounds[:, 0] / scale, device=observed.device)
    high = torch.as_tensor(bounds[:, 1] / scale, device=observed.device)
    results = []
    for candidate in candidates:
        initial = torch.as_tensor(np.asarray(candidate["source_m_s"], dtype=np.float32) / scale,
                                  device=observed.device)
        fraction = torch.nn.Parameter(((initial - low) / (high - low)).clamp(0., 1.))
        optimizer = torch.optim.Adam((fraction,), lr=config["refine_adam_learning_rate_on_box_fractions"])
        best = (float("inf"), initial.detach().clone())
        for _ in range(config["refine_steps"]):
            optimizer.zero_grad()
            source = low + (high - low) * fraction
            misfit = waveform_mse(model, xyz, time_t, selected, source, rms)
            value = float(misfit.detach())
            if value < best[0]:
                best = value, source.detach().clone()
            misfit.backward()
            optimizer.step()
            with torch.no_grad():
                fraction.clamp_(0., 1.)
        with torch.no_grad():
            source = low + (high - low) * fraction
            value = float(waveform_mse(model, xyz, time_t, selected, source, rms))
            if value < best[0]:
                best = value, source.detach().clone()
        estimate = (best[1].cpu().numpy() * scale).astype(np.float64)
        results.append({"initial_source_m_s": candidate["source_m_s"],
                        "coarse_normalized_mse": candidate["coarse_normalized_mse"],
                        "estimated_source_m_s": estimate.tolist(),
                        "normalized_mse": best[0],
                        "on_region_boundary": bool(np.any(np.isclose(
                            estimate, bounds[:, 0], atol=[1, 1, 1, .001])
                            | np.isclose(estimate, bounds[:, 1], atol=[1, 1, 1, .001])))})
    best = min(results, key=lambda row: row["normalized_mse"])
    return best, results, xyz, time_t, selected


def inversion_without_truth(model, data: np.ndarray, times: np.ndarray,
                            receivers: np.ndarray, region: dict,
                            config: dict, scale: np.ndarray) -> dict:
    """Never accepts source_truth or any event-specific truth-derived bounds."""
    bounds = np.asarray(region["bounds_m_s"], dtype=np.float32)
    device = next(model.parameters()).device
    observations = torch.as_tensor(data, device=device)
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    if rms <= 0 or not np.isfinite(rms):
        raise ValueError("Invalid full-record observation RMS")
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    candidates = coarse_search(model, receivers, times, observations, scale, bounds, config, rms)
    best, refinements, xyz, time_t, selected = refine(
        model, receivers, times, observations, scale, bounds, config, rms, candidates)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"bounds_m_s": bounds.tolist(), "grid_shape_xyzt": config["coarse_grid_xyzt"],
            "grid_candidates": int(np.prod(config["coarse_grid_xyzt"])),
            "coarse_starts": candidates, "refinements": refinements,
            "estimated_source_m_s": best["estimated_source_m_s"],
            "final_normalized_mse": best["normalized_mse"],
            "source_truth_used_in_search": False,
            "runtime_seconds": time.perf_counter() - start}, (rms, xyz, time_t, selected)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checkpoint = ROOT / protocol["checkpoint"]
    if sha256(checkpoint).lower() != protocol["checkpoint_sha256"].lower():
        raise ValueError("Frozen checkpoint hash does not match preregistered SHA-256")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_conditioned_checkpoint(checkpoint, device)
    training = metadata["training_record"]["receiver_observations"]
    if len(training) != 25 or metadata["frequency_hz"] != 5:
        raise ValueError("Checkpoint must contain the registered 25-source 5 Hz model")
    model.eval().requires_grad_(False)
    cfg = metadata["config"]
    scale = np.asarray([cfg["x_length_m"], cfg["y_length_m"],
                        cfg["z_length_m"], cfg["time_length_s"]], dtype=np.float32)
    files = generate_and_verify(protocol, training)
    forward_output = RESULTS / "depth_gap_independent_forward.json"
    subprocess.run([sys.executable, str(ROOT / "examples/audit_forward_error_decomposition.py"),
                    str(checkpoint), *(str(path) for path in files),
                    "--evaluation-label", "synthetic-confirmation",
                    "--output", str(forward_output)], cwd=ROOT, check=True)
    forward = json.loads(forward_output.read_text(encoding="utf-8"))
    records = []
    for item, path, prediction in zip(protocol["events"], files, forward["events"]):
        if prediction["file_in_receiver_training_set"] or prediction["source_in_receiver_training_set"]:
            raise ValueError("Confirmation event overlaps training data")
        result = {"event": item["file"], "role": item["role"], "region": item["region"],
                  "observation": str(path), "sha256": sha256(path),
                  "raw_waveform_relative_l2": prediction["original_amplitude_relative_l2"],
                  "fd_runtime_s": json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["runtime_seconds"]}
        if item["region"] == "remote":
            result["source_truth_m_s"] = item["source_m_s"]
            result["inversion"] = None
        else:
            data, times, receivers, truth, _ = load_observations(path)
            if not np.allclose(truth, item["source_m_s"], atol=1e-5, rtol=0):
                raise ValueError("Unexpected FD event truth")
            inverse, (rms, xyz, time_t, selected) = inversion_without_truth(
                model, data, times, receivers,
                protocol["preassigned_inversion_regions"][item["region"]],
                protocol["inverse"], scale)
            estimate = np.asarray(inverse["estimated_source_m_s"])
            delta = estimate - truth
            with torch.inference_mode():
                true_source = torch.as_tensor(truth / scale, device=device)
                true_loss = float(waveform_mse(model, xyz, time_t, selected, true_source, rms))
            inverse.update({"source_truth_m_s_post_search": truth.tolist(),
                            "component_error_m_s": delta.tolist(),
                            "location_error_m": float(np.linalg.norm(delta[:3])),
                            "origin_time_error_s": float(abs(delta[3])),
                            "truth_source_normalized_mse": true_loss,
                            "selected_loss_below_truth": bool(inverse["final_normalized_mse"] < true_loss)})
            result["inversion"] = inverse
            result["source_truth_m_s"] = truth.tolist()
        records.append(result)
        individual = RESULTS / f"depth_gap_independent_{path.stem}.json"
        individual.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if result["inversion"]:
            inv = result["inversion"]
            print(f"{path.name}: waveform L2={result['raw_waveform_relative_l2']:.4f}; "
                  f"location error={inv['location_error_m']:.1f} m, "
                  f"time error={inv['origin_time_error_s']:.4f} s", flush=True)
        else:
            print(f"{path.name}: waveform L2={result['raw_waveform_relative_l2']:.4f}; "
                  "forward-only remote source", flush=True)
    gates = protocol["gates"]
    local = [row for row in records if row["region"] != "remote"]
    remote = [row for row in records if row["region"] == "remote"]
    checks = {
        "local_forward": all(row["raw_waveform_relative_l2"] < gates[
            "all_six_local_raw_relative_l2_strictly_below"] for row in local),
        "remote_forward": all(row["raw_waveform_relative_l2"] < gates[
            "both_remote_raw_relative_l2_strictly_below"] for row in remote),
        "local_location": all(row["inversion"]["location_error_m"] < gates[
            "all_six_local_location_error_m_strictly_below"] for row in local),
        "local_origin_time": all(row["inversion"]["origin_time_error_s"] < gates[
            "all_six_local_origin_time_error_s_strictly_below"] for row in local),
    }
    report = {"purpose": "preregistered one-time synthetic FD confirmation",
              "protocol": str(PROTOCOL.resolve()), "checkpoint_sha256": sha256(checkpoint),
              "all_gates_passed": all(checks.values()), "gates": checks,
              "events": records,
              "limitations": "Synthetic 5 Hz same-model evaluation after extensive development; not field validation, multi-seed uncertainty or training-source-independent FD cost advantage."}
    target = RESULTS / "depth_gap_independent_results.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {target}; gates={checks}; all_pass={all(checks.values())}", flush=True)


if __name__ == "__main__":
    main()
