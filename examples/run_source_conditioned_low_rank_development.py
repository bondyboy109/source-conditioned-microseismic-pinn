"""Train and evaluate the factorized source-conditioned waveform operator."""

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

from examples.invert_fmm_receiver_fields import interpolate_fields
from examples.run_traveltime_aligned_moe_development import (
    base_prediction, build_or_load_fields, load_observations,
    relative_l2, sample_training_batch,
)
from source_conditioned_pinn import load_conditioned_checkpoint
from source_conditioned_low_rank import SourceConditionedLowRankOperator
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator

PROTOCOL = ROOT / "experiments/source_conditioned_low_rank_development_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction(model, event, receivers, times, domain, device):
    shape = event["data"].shape
    coordinates = np.concatenate((
        np.broadcast_to(receivers[:, None, :], (*shape, 3)),
        np.broadcast_to(times[None, :, None], (*shape, 1)),
    ), axis=-1).reshape(-1, 4) / domain
    sources = np.broadcast_to(event["source"] / domain,
                              (len(coordinates), 4)).astype(np.float32)
    receiver_ids = np.repeat(np.arange(len(receivers)), len(times))
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coordinates), 8192):
            points = torch.as_tensor(coordinates[start:start + 8192],
                                     device=device, dtype=torch.float32)
            source = torch.as_tensor(sources[start:start + 8192],
                                     device=device, dtype=torch.float32)
            ids = torch.as_tensor(receiver_ids[start:start + 8192],
                                  device=device, dtype=torch.long)
            values.append(model(points, source, ids)[0].squeeze(1).cpu().numpy())
    return np.concatenate(values).reshape(shape)


def source_loss_gradient(model, event, receivers, times, domain, device):
    data = event["data"] / event["rms"]
    energy = np.sum(data.astype(np.float64) ** 2, axis=1)
    receiver_id = int(np.argmax(energy))
    time_id = np.linspace(0, len(times) - 1, 512).round().astype(int)
    point = np.concatenate((
        np.broadcast_to(receivers[receiver_id], (len(time_id), 3)),
        times[time_id, None]), axis=1) / domain
    source = torch.tensor(
        np.broadcast_to(event["source"] / domain, (len(time_id), 4)).copy(),
        device=device, dtype=torch.float32, requires_grad=True)
    output = model(
        torch.as_tensor(point, device=device, dtype=torch.float32), source,
        torch.full((len(time_id),), receiver_id, device=device, dtype=torch.long),
    )[0]
    target = torch.as_tensor(data[receiver_id, time_id], device=device,
                             dtype=torch.float32).unsqueeze(1)
    gradient = torch.autograd.grad((output - target).square().mean(), source)[0]
    return gradient.abs().mean(dim=0).detach().cpu().numpy()


def write_report(path: Path, result: dict) -> None:
    summary = result["development_summary"]
    gates = result["development_gate_results"]
    lines = [
        "# Source-conditioned low-rank operator development result", "",
        "The model was selected using the 16-event development set only. The independent confirmation set was excluded.", "",
        "| Metric | Frozen base | Low-rank operator |",
        "|---|---:|---:|",
        f"| Mean full relative L2 | {summary['base_mean_full_l2']:.4f} | {summary['operator_mean_full_l2']:.4f} |",
        f"| Median full relative L2 | {summary['base_median_full_l2']:.4f} | {summary['operator_median_full_l2']:.4f} |",
        f"| Events below zero-prediction L2 | -- | {summary['operator_events_below_one']}/16 |",
        f"| Events improved over frozen base | -- | {summary['events_improved']}/16 |",
        "", f"Median absolute source-loss gradient: `{summary['median_absolute_source_loss_gradient']}`.", "",
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
    if result_path.exists() or report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("Refusing to overwrite registered low-rank outputs")
    velocity_path = ROOT / spec["velocity_file"]
    manifest_path = ROOT / spec["training_manifest_checkpoint"]
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
    arrivals = interpolate_fields(fields,
                                  np.stack([event["source"][:3] for event in training_events]),
                                  spacing)
    model_spec = spec["model"]
    domain = np.asarray([*model_spec["spatial_lengths_m"],
                         model_spec["time_length_s"]], dtype=np.float32)
    interpolator = ReceiverTravelTimeInterpolator(
        torch.from_numpy(np.array(fields, copy=True)), spacing).to(device)
    model = SourceConditionedLowRankOperator(
        interpolator, torch.from_numpy(receivers),
        tuple(model_spec["spatial_lengths_m"]),
        time_length_s=float(model_spec["time_length_s"]),
        frequency_hz=float(model_spec["frequency_hz"]),
        rank=int(model_spec["rank"]), hidden_dim=int(model_spec["hidden_dim"]),
        hidden_layers=int(model_spec["hidden_layers"]),
        causal_width_s=float(model_spec["causal_width_s"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(spec["training"]["learning_rate"]))
    history = []
    started = time.perf_counter()
    for update in range(1, int(spec["training"]["updates"]) + 1):
        model.train()
        points, sources, receiver_ids, target = sample_training_batch(
            training_events, receivers, times, arrivals, spec, rng, domain, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(points, sources, receiver_ids)[0]
        waveform_loss = (output - target).square().mean()
        coefficient_loss = model.coefficient_regularization()
        loss = waveform_loss + float(spec["training"]["coefficient_weight"]) * coefficient_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                      float(spec["training"]["gradient_clip_norm"]))
        optimizer.step()
        if update == 1 or update % 100 == 0:
            row = {"update": update, "loss": float(loss.detach()),
                   "waveform": float(waveform_loss.detach()),
                   "coefficient": float(coefficient_loss.detach())}
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
                              ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                             dtype=np.float32)
    rows = []
    gradients = []
    for event in development_events:
        estimate = prediction(model, event, receivers, times, domain, device)
        base = base_prediction(base_model, event, receivers, times,
                               base_domain, device)
        normalized = event["data"] / event["rms"]
        operator_l2 = relative_l2(normalized, estimate)
        base_l2 = relative_l2(event["data"], base)
        gradient = source_loss_gradient(model, event, receivers, times, domain, device)
        gradients.append(gradient)
        rows.append({"event": event["path"].name,
                     "observation": str(event["path"]),
                     "sha256": event["sha256"],
                     "source_m_s": event["source"].tolist(),
                     "base_full_relative_l2": base_l2,
                     "operator_full_relative_l2": operator_l2,
                     "operator_below_zero_prediction": operator_l2 < 1.0,
                     "operator_improves_base": operator_l2 < base_l2,
                     "absolute_source_loss_gradient": gradient.tolist()})
        print(f"{event['path'].name}: base={base_l2:.3f}, operator={operator_l2:.3f}", flush=True)
    base_values = np.asarray([row["base_full_relative_l2"] for row in rows])
    operator_values = np.asarray([row["operator_full_relative_l2"] for row in rows])
    median_gradient = np.median(np.asarray(gradients), axis=0)
    thresholds = spec["development_gates"]
    gate_results = {
        "mean_l2_improves_by_ten_percent": bool(
            operator_values.mean() / base_values.mean()
            < thresholds["mean_full_relative_l2_fraction_of_frozen_base_below"]),
        "at_least_twelve_events_improve": bool(
            np.sum(operator_values < base_values)
            >= thresholds["events_improved_over_frozen_base_at_least"]),
        "at_least_twelve_events_beat_zero_prediction": bool(
            np.sum(operator_values < 1.0)
            >= thresholds["events_below_zero_prediction_l2_at_least"]),
        "all_source_gradient_components_present": bool(np.all(
            median_gradient > thresholds["all_four_source_gradient_components_above"])),
    }
    summary = {
        "base_mean_full_l2": float(base_values.mean()),
        "base_median_full_l2": float(np.median(base_values)),
        "operator_mean_full_l2": float(operator_values.mean()),
        "operator_median_full_l2": float(np.median(operator_values)),
        "relative_mean_improvement": float((base_values.mean() - operator_values.mean())
                                           / base_values.mean()),
        "events_improved": int(np.sum(operator_values < base_values)),
        "operator_events_below_one": int(np.sum(operator_values < 1.0)),
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
        "events": rows,
        "decision": ("Proceed to independent localization only if this result remains unchanged under the frozen confirmation protocol."
                     if passed else
                     "Stop before localization; revise the forward operator using training and development data only."),
        "limitations": spec["limitations"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)
    print(json.dumps({"development_summary": summary,
                      "development_gate_results": gate_results,
                      "all_development_gates_passed": passed,
                      "training_seconds": training_seconds}, indent=2), flush=True)


if __name__ == "__main__":
    main()
