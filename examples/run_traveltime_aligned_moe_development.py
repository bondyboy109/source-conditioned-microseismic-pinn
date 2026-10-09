"""Train and gate the registered FMM-aligned MoE on development data."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.invert_fmm_receiver_fields import interpolate_fields, receiver_fields
from source_conditioned_pinn import load_conditioned_checkpoint
from traveltime_aligned_moe import (
    ReceiverTravelTimeInterpolator,
    TravelTimeAlignedPrototypeMoE,
)

PROTOCOL = ROOT / "experiments/traveltime_aligned_moe_development_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_observations(paths: list[Path]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    events = []
    common_receivers = None
    common_times = None
    for path in paths:
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            source = np.asarray(archive["source_truth"], dtype=np.float32)
        if (data.shape != (len(receivers), len(times))
                or source.shape != (4,) or not np.isfinite(data).all()):
            raise ValueError(f"Invalid observation: {path}")
        if common_receivers is None:
            common_receivers = receivers
            common_times = times
        elif (not np.array_equal(receivers, common_receivers)
              or not np.array_equal(times, common_times)):
            raise ValueError(f"Receiver geometry or time samples changed: {path}")
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
        if not np.isfinite(rms) or rms <= 0:
            raise ValueError(f"Observation has zero or invalid RMS: {path}")
        threshold = 0.05 * float(np.max(np.abs(data)))
        active = np.flatnonzero(np.abs(data).reshape(-1) >= threshold)
        if len(active) == 0:
            raise ValueError(f"Observation has no active samples: {path}")
        events.append({"path": path, "sha256": sha256(path), "data": data,
                       "source": source, "rms": rms, "active": active})
    return events, common_receivers, common_times


def build_or_load_fields(spec: dict, receivers: np.ndarray) -> tuple[np.ndarray, float]:
    cache = ROOT / spec["fmm"]["cache_file"]
    spacing = float(spec["fmm"]["grid_spacing_m"])
    stride = int(spec["fmm"]["velocity_grid_stride"])
    velocity_path = ROOT / spec["velocity_file"]
    expected_shape = (len(receivers), 201, 201, 47)
    if cache.is_file():
        fields = np.load(cache, mmap_mode="r")
        if fields.shape != expected_shape or fields.dtype != np.float32:
            raise ValueError("Cached FMM field geometry changed")
        return np.asarray(fields), 0.0
    velocity = np.load(velocity_path, mmap_mode="r")
    downsampled = np.ascontiguousarray(velocity[::stride, ::stride, ::stride],
                                      dtype=np.float32)
    started = time.perf_counter()
    fields = receiver_fields(downsampled, receivers, spacing)
    elapsed = time.perf_counter() - started
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, fields)
    return fields, elapsed


def sample_training_batch(events: list[dict], receivers: np.ndarray,
                          times: np.ndarray, arrivals: np.ndarray,
                          spec: dict, rng: np.random.Generator,
                          domain: np.ndarray, device: torch.device):
    training = spec["training"]
    batch_size = int(training["batch_size"])
    event_count = int(training["events_per_batch"])
    chosen = rng.choice(len(events), size=event_count, replace=False)
    counts = np.full(event_count, batch_size // event_count, dtype=int)
    counts[:batch_size % event_count] += 1
    fractions = training["sampling"]
    coordinates, sources, indices, targets = [], [], [], []
    nt = len(times)
    for event_index, count in zip(chosen, counts):
        event = events[int(event_index)]
        n_arrival = int(round(count * fractions["fmm_arrival_window_fraction"]))
        n_active = int(round(count * fractions["observed_active_fraction"]))
        n_uniform = count - n_arrival - n_active
        receiver_parts, time_parts = [], []
        if n_arrival:
            receiver_id = rng.integers(0, len(receivers), size=n_arrival)
            center = event["source"][3] + arrivals[int(event_index), receiver_id]
            sampled_time = center + rng.uniform(
                -float(fractions["arrival_half_width_s"]),
                float(fractions["arrival_half_width_s"]), size=n_arrival)
            time_id = np.searchsorted(times, sampled_time).clip(0, nt - 1)
            receiver_parts.append(receiver_id)
            time_parts.append(time_id)
        if n_active:
            flat = rng.choice(event["active"], size=n_active, replace=True)
            receiver_parts.append(flat // nt)
            time_parts.append(flat % nt)
        if n_uniform:
            receiver_parts.append(rng.integers(0, len(receivers), size=n_uniform))
            time_parts.append(rng.integers(0, nt, size=n_uniform))
        receiver_id = np.concatenate(receiver_parts).astype(np.int64)
        time_id = np.concatenate(time_parts).astype(np.int64)
        order = rng.permutation(count)
        receiver_id, time_id = receiver_id[order], time_id[order]
        point = np.concatenate((receivers[receiver_id], times[time_id, None]), axis=1)
        coordinates.append(point / domain)
        sources.append(np.broadcast_to(event["source"] / domain, (count, 4)))
        indices.append(receiver_id)
        targets.append(event["data"][receiver_id, time_id] / event["rms"])
    return (
        torch.as_tensor(np.concatenate(coordinates), device=device,
                        dtype=torch.float32),
        torch.as_tensor(np.concatenate(sources), device=device,
                        dtype=torch.float32),
        torch.as_tensor(np.concatenate(indices), device=device,
                        dtype=torch.long),
        torch.as_tensor(np.concatenate(targets), device=device,
                        dtype=torch.float32).unsqueeze(1),
    )


def aligned_prediction(model: TravelTimeAlignedPrototypeMoE, event: dict,
                       receivers: np.ndarray, times: np.ndarray,
                       domain: np.ndarray, device: torch.device) -> np.ndarray:
    shape = event["data"].shape
    coordinates = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*shape, 3)),
        np.broadcast_to(times[None, :, None], (*shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    sources = np.broadcast_to(event["source"] / domain,
                              (len(coordinates), 4)).astype(np.float32)
    receiver_index = np.repeat(np.arange(len(receivers)), len(times))
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coordinates), 8192):
            points = torch.as_tensor(coordinates[start:start + 8192],
                                     device=device, dtype=torch.float32)
            source = torch.as_tensor(sources[start:start + 8192],
                                     device=device, dtype=torch.float32)
            index = torch.as_tensor(receiver_index[start:start + 8192],
                                    device=device, dtype=torch.long)
            output.append(model(points, source, index)[0].squeeze(1).cpu().numpy())
    return np.concatenate(output).reshape(shape)


def base_prediction(model: torch.nn.Module, event: dict, receivers: np.ndarray,
                    times: np.ndarray, domain: np.ndarray,
                    device: torch.device) -> np.ndarray:
    shape = event["data"].shape
    coordinates = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*shape, 3)),
        np.broadcast_to(times[None, :, None], (*shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    sources = np.broadcast_to(event["source"] / domain,
                              (len(coordinates), 4)).astype(np.float32)
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coordinates), 8192):
            points = torch.as_tensor(coordinates[start:start + 8192],
                                     device=device, dtype=torch.float32)
            source = torch.as_tensor(sources[start:start + 8192],
                                     device=device, dtype=torch.float32)
            output.append(model(points, source).squeeze(1).cpu().numpy())
    return np.concatenate(output).reshape(shape)


def relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm(estimate.astype(np.float64)
                                - reference.astype(np.float64))
                 / np.linalg.norm(reference.astype(np.float64)))


def source_loss_gradient(model: TravelTimeAlignedPrototypeMoE, event: dict,
                         receivers: np.ndarray, times: np.ndarray,
                         domain: np.ndarray, device: torch.device) -> np.ndarray:
    data = event["data"] / event["rms"]
    energy = np.sum(data.astype(np.float64) ** 2, axis=1)
    receiver_id = int(np.argmax(energy))
    time_id = np.linspace(0, len(times) - 1, 512).round().astype(int)
    point = np.concatenate((
        np.broadcast_to(receivers[receiver_id], (len(time_id), 3)),
        times[time_id, None]), axis=1) / domain
    source_value = event["source"] / domain
    source = torch.tensor(np.broadcast_to(source_value, (len(time_id), 4)).copy(),
                          device=device, dtype=torch.float32,
                          requires_grad=True)
    prediction = model(
        torch.as_tensor(point, device=device, dtype=torch.float32), source,
        torch.full((len(time_id),), receiver_id, device=device, dtype=torch.long))[0]
    target = torch.as_tensor(data[receiver_id, time_id], device=device,
                             dtype=torch.float32).unsqueeze(1)
    gradient = torch.autograd.grad((prediction - target).square().mean(), source)[0]
    return gradient.abs().mean(dim=0).detach().cpu().numpy()


def write_report(path: Path, result: dict) -> None:
    summary = result["development_summary"]
    gates = result["development_gate_results"]
    lines = [
        "# FMM-traveltime-aligned MoE development result", "",
        "This is a development-only waveform test. The registered eight-event ensemble confirmation was excluded from training and model selection.", "",
        "| Metric | Frozen base | Aligned MoE |", "|---|---:|---:|",
        f"| Mean full relative L2 | {summary['base_mean_full_l2']:.4f} | {summary['aligned_mean_full_l2']:.4f} |",
        f"| Median full relative L2 | {summary['base_median_full_l2']:.4f} | {summary['aligned_median_full_l2']:.4f} |",
        f"| Events below zero-prediction L2 | -- | {summary['aligned_events_below_one']}/16 |",
        f"| Events improved | -- | {summary['events_improved']}/16 |", "",
        f"Mean gate use: `{summary['mean_gate']}`.", "",
        "## Predeclared gates", "", "| Gate | Result |", "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |"
                 for name, passed in gates.items())
    lines.extend(["", f"All gates passed: **{result['all_development_gates_passed']}**.", "",
                  result["decision"], "", f"Limitations: {result['limitations']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result_path = ROOT / spec["result_file"]
    report_path = ROOT / spec["report_file"]
    checkpoint_path = ROOT / spec["output_checkpoint"]
    if result_path.exists() or checkpoint_path.exists():
        raise FileExistsError("Refusing to overwrite registered development outputs")
    if not spec["protocol_written_before_training"]:
        raise ValueError("Protocol was not frozen before training")
    velocity_path = ROOT / spec["velocity_file"]
    manifest_path = ROOT / spec["training_manifest_checkpoint"]
    if (sha256(velocity_path) != spec["velocity_sha256"]
            or sha256(manifest_path) != spec["training_manifest_checkpoint_sha256"]):
        raise ValueError("Registered velocity or training manifest changed")
    payload = torch.load(manifest_path, map_location="cpu", weights_only=False)
    manifest = payload.get("training_record", {}).get("receiver_observations", [])
    if len(manifest) != int(spec["training_observation_count"]):
        raise ValueError("Training observation count changed")
    excluded = (ROOT / spec["excluded_confirmation_directory"]).resolve()
    training_paths = []
    for row in manifest:
        path = Path(row["path"]).resolve()
        if excluded in path.parents or path == excluded:
            raise ValueError("Independent confirmation leaked into training")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Training observation changed: {path}")
        training_paths.append(path)
    development_paths = []
    for directory in spec["development_sets"]:
        development_paths.extend(sorted((ROOT / directory).glob("*.npz")))
    if len(development_paths) != int(spec["development_event_count"]):
        raise ValueError("Development event count changed")
    seed = int(spec["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_events, receivers, times = load_observations(training_paths)
    development_events, dev_receivers, dev_times = load_observations(development_paths)
    if not np.array_equal(receivers, dev_receivers) or not np.array_equal(times, dev_times):
        raise ValueError("Development acquisition differs from training acquisition")
    fields, fmm_seconds = build_or_load_fields(spec, receivers)
    spacing = float(spec["fmm"]["grid_spacing_m"])
    source_xyz = np.stack([event["source"][:3] for event in training_events])
    arrivals = interpolate_fields(fields, source_xyz, spacing)
    model_spec = spec["model"]
    domain = np.asarray([*model_spec["spatial_lengths_m"],
                         model_spec["time_length_s"]], dtype=np.float32)
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)), spacing).to(device)
    model = TravelTimeAlignedPrototypeMoE(
        interpolator, torch.from_numpy(receivers),
        tuple(model_spec["spatial_lengths_m"]),
        time_length_s=float(model_spec["time_length_s"]),
        frequency_hz=float(model_spec["frequency_hz"]),
        expert_count=int(model_spec["expert_count"]),
        hidden_dim=int(model_spec["hidden_dim"]),
        hidden_layers=int(model_spec["hidden_layers"]),
        prototype_centers=torch.tensor(model_spec["prototype_centers"]),
        prototype_scales=tuple(model_spec["prototype_scales"]),
        temperature=float(model_spec["prototype_temperature"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(spec["training"]["learning_rate"]))
    history = []
    started = time.perf_counter()
    for update in range(1, int(spec["training"]["updates"]) + 1):
        model.train()
        points, sources, receiver_index, target = sample_training_batch(
            training_events, receivers, times, arrivals, spec, rng, domain, device)
        optimizer.zero_grad(set_to_none=True)
        prediction, gate, _ = model(points, sources, receiver_index)
        waveform_loss = (prediction - target).square().mean()
        gate_mean = gate.mean(dim=0)
        balance_loss = ((gate_mean - 1.0 / model.expert_count) ** 2).mean()
        separation_loss = model.prototype_separation_loss()
        loss = (waveform_loss
                + float(spec["training"]["gate_balance_weight"]) * balance_loss
                + float(spec["training"]["prototype_separation_weight"])
                * separation_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                      float(spec["training"]["gradient_clip_norm"]))
        optimizer.step()
        low, high = model_spec["prototype_bounds_normalized"]
        with torch.no_grad():
            model.prototype_centers.clamp_(float(low), float(high))
        if update == 1 or update % 100 == 0:
            row = {"update": update, "loss": float(loss.detach()),
                   "waveform": float(waveform_loss.detach()),
                   "gate_balance": float(balance_loss.detach()),
                   "prototype_separation": float(separation_loss.detach()),
                   "mean_gate": gate_mean.detach().cpu().tolist()}
            history.append(row)
            print(row, flush=True)
    training_seconds = time.perf_counter() - started
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    field_cache = ROOT / spec["fmm"]["cache_file"]
    torch.save({"state_dict": model.state_dict(), "protocol": spec,
                "protocol_sha256": sha256(PROTOCOL),
                "field_cache_sha256": sha256(field_cache),
                "training_history": history,
                "training_seconds": training_seconds}, checkpoint_path)

    base_model, base_payload = load_conditioned_checkpoint(manifest_path, device)
    base_domain = np.asarray([base_payload["config"][key] for key in
                              ("x_length_m", "y_length_m", "z_length_m",
                               "time_length_s")], dtype=np.float32)
    events = []
    gradient_rows = []
    gate_rows = []
    for event in development_events:
        aligned = aligned_prediction(model, event, receivers, times, domain, device)
        # The archived base was trained with a 9-s normalization domain even
        # though these receiver records were later cropped to 0--2 s. Evaluate
        # each model with its own frozen normalization convention.
        base = base_prediction(base_model, event, receivers, times, base_domain, device)
        normalized = event["data"] / event["rms"]
        aligned_l2 = relative_l2(normalized, aligned)
        base_l2 = relative_l2(event["data"], base)
        source_norm = torch.as_tensor((event["source"] / domain)[None, :3],
                                      device=device, dtype=torch.float32)
        gate = model.source_gate(source_norm).detach().cpu().numpy()[0]
        gradient = source_loss_gradient(model, event, receivers, times, domain, device)
        gate_rows.append(gate)
        gradient_rows.append(gradient)
        events.append({"event": event["path"].name,
                       "observation": str(event["path"]),
                       "sha256": event["sha256"],
                       "source_m_s": event["source"].tolist(),
                       "base_full_relative_l2": base_l2,
                       "aligned_full_relative_l2": aligned_l2,
                       "aligned_below_zero_prediction": aligned_l2 < 1.0,
                       "aligned_improves_base": aligned_l2 < base_l2,
                       "mean_gate": gate.tolist(),
                       "absolute_source_loss_gradient": gradient.tolist()})
        print(f"{event['path'].name}: base={base_l2:.3f}, aligned={aligned_l2:.3f}",
              flush=True)
    base_values = np.asarray([row["base_full_relative_l2"] for row in events])
    aligned_values = np.asarray([row["aligned_full_relative_l2"] for row in events])
    mean_gate = np.mean(gate_rows, axis=0)
    median_gradient = np.median(gradient_rows, axis=0)
    thresholds = spec["development_gates"]
    gate_results = {
        "mean_l2_improves_by_ten_percent": bool(
            aligned_values.mean() / base_values.mean()
            < thresholds["mean_full_relative_l2_fraction_of_frozen_base_below"]),
        "at_least_twelve_events_improve": bool(
            np.sum(aligned_values < base_values)
            >= thresholds["events_improved_over_frozen_base_at_least"]),
        "at_least_twelve_events_beat_zero_prediction": bool(
            np.sum(aligned_values < 1.0)
            >= thresholds["events_below_zero_prediction_l2_at_least"]),
        "prototype_gate_does_not_collapse": bool(
            mean_gate.max() < thresholds["maximum_mean_gate_probability_below"]),
        "all_source_gradient_components_present": bool(np.all(
            median_gradient > thresholds["all_four_source_gradient_components_above"])),
    }
    summary = {
        "base_mean_full_l2": float(base_values.mean()),
        "base_median_full_l2": float(np.median(base_values)),
        "aligned_mean_full_l2": float(aligned_values.mean()),
        "aligned_median_full_l2": float(np.median(aligned_values)),
        "relative_mean_improvement": float((base_values.mean() - aligned_values.mean())
                                           / base_values.mean()),
        "events_improved": int(np.sum(aligned_values < base_values)),
        "aligned_events_below_one": int(np.sum(aligned_values < 1.0)),
        "mean_gate": mean_gate.tolist(),
        "median_absolute_source_loss_gradient": median_gradient.tolist(),
    }
    passed = bool(all(gate_results.values()))
    result = {
        "purpose": spec["purpose"], "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "training_observation_count": len(training_events),
        "development_event_count": len(development_events),
        "independent_confirmation_used": False,
        "device": str(device), "fmm_setup_seconds": fmm_seconds,
        "field_cache": str(field_cache.resolve()),
        "field_cache_sha256": sha256(field_cache),
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "development_summary": summary,
        "development_gate_results": gate_results,
        "all_development_gates_passed": passed,
        "events": events,
        "decision": ("Proceed to registered development localization without changing the model."
                     if passed else
                     "Stop before localization; revise the forward model using training and development data only."),
        "limitations": spec["limitations"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)
    print(json.dumps({"development_summary": summary,
                      "development_gate_results": gate_results,
                      "all_development_gates_passed": passed,
                      "training_seconds": training_seconds}, indent=2), flush=True)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
