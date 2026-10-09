"""Train identical receiver-supervised heads with learned vs FMM travel times."""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_fmm_carrier_waveform_swap import frozen_head
from examples.invert_fmm_receiver_fields import interpolate_fields, receiver_fields
from examples.run_fmm_seeded_waveform_development import hash_file
from receiver_training import ReceiverTrainingData
from source_conditioned_pinn import (
    EikonalTravelTimeCarrierWavePINN, SourceConditionedConfig,
    load_conditioned_checkpoint,
)

PROTOCOL = ROOT / "experiments/fmm_receiver_head_paired_protocol.json"


def check(protocol: dict) -> tuple[dict, list[Path], list[Path]]:
    files = (("base_receiver_checkpoint", "base_receiver_sha256"),
             ("velocity_file", "velocity_sha256"),
             ("fmm_reference", "fmm_reference_sha256"),
             ("development_protocol", "development_protocol_sha256"),
             ("development_results", "development_results_sha256"))
    for name, digest in files:
        if hash_file(ROOT / protocol[name]) != protocol[digest]:
            raise ValueError(f"Frozen input changed: {protocol[name]}")
    if (ROOT / protocol["result_file"]).exists() or (ROOT / protocol["output_directory"]).exists():
        raise FileExistsError("Paired pilot already exists; refusing to overwrite")
    if (protocol["training_sources"] != 29 or protocol["events_per_batch"] != 29
            or protocol["batch_size"] != 3712 or protocol["batch_size"] % 29 != 0
            or not protocol["no_pde_ic_bc_loss"]):
        raise ValueError("Paired sampling must use the fixed receiver-only budget")
    checkpoint = torch.load(ROOT / protocol["base_receiver_checkpoint"],
                            map_location="cpu", weights_only=False)
    manifest = checkpoint["training_record"]["receiver_observations"]
    if (len(manifest) != 29 or checkpoint["frequency_hz"] != 5
            or checkpoint["architecture"]["kind"] != "eikonal_travel_time_carrier"
            or checkpoint["epoch"] != 5000 or checkpoint["training_record"]["seed"] != 42
            or checkpoint["training_record"]["receiver_batch_size"] != protocol["batch_size"]):
        raise ValueError("Receiver checkpoint has an unexpected training manifest")
    training = [Path(item["path"]) for item in manifest]
    for item, path in zip(manifest, training):
        if hash_file(path) != item["sha256"]:
            raise ValueError(f"Training record changed: {path}")
    development = json.loads((ROOT / protocol["development_results"]).read_text())
    dev_registration = json.loads((ROOT / protocol["development_protocol"]).read_text())
    if (development["protocol_sha256"] != protocol["development_protocol_sha256"]
            or len(development["events"]) != 12
            or development["checkpoint_sha256"]["eikonal"]
            != "ce744ba5e7249850fe99644449855c886fc255aebacfacab86f76d59990f2ec7"):
        raise ValueError("Directional development events are not the archived group")
    dev = [ROOT / "data/observations/eikonal_directional_grid" / row["file"]
           for row in development["events"]]
    if set(training) & set(dev):
        raise ValueError("Training / development overlap")
    for row, path in zip(development["events"], dev):
        if (hash_file(path) != row["sha256"]
                or row["anchor"] not in {x["name"] for x in dev_registration["anchors"]}):
            raise ValueError(f"Development observation changed: {path}")
    return checkpoint, training, dev


def fmm_tables(protocol: dict, training: list[Path], dev: list[Path],
               device: torch.device) -> tuple[torch.Tensor, np.ndarray, dict]:
    reference = json.loads((ROOT / protocol["fmm_reference"]).read_text())
    if (len(reference["events"]) != 37 or reference["fmm_grid_spacing_m"] != 80
            or [r["observation_sha256"] for r in reference["events"][:29]]
            != [hash_file(p) for p in training]):
        raise ValueError("Previous FMM table differs from the 29 training files")
    with np.load(training[0]) as archive:
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
        times = np.asarray(archive["times"], dtype=np.float32)
    sources = []
    for path in training + dev:
        with np.load(path) as archive:
            if (archive["data"].shape != (25, 1001)
                    or not np.array_equal(archive["times"], times)
                    or not np.array_equal(archive["receiver_coords"], receivers)):
                raise ValueError(f"Different FD geometry: {path}")
            sources.append(np.asarray(archive["source_truth"][:3], dtype=np.float64))
    velocity = np.load(ROOT / protocol["velocity_file"], mmap_mode="r")
    if velocity.shape != (401, 401, 94):
        raise ValueError("Velocity grid differs from FMM baseline")
    started = time.perf_counter()
    fields = receiver_fields(np.ascontiguousarray(velocity[::2, ::2, ::2],
                                                  dtype=np.float32), receivers, 80.)
    elapsed = time.perf_counter() - started
    lookup = interpolate_fields(fields, np.asarray(sources), 80.)
    earlier = np.asarray([r["fmm_traveltime_s"] for r in reference["events"][:29]])
    if not np.allclose(lookup[:29], earlier, atol=1e-6, rtol=0):
        raise ValueError("New FMM receiver fields disagree with archived training times")
    info = {"setup_seconds": elapsed, "field_cache_mib": fields.nbytes / 1048576,
            "training_fmm_reference_sha256": protocol["fmm_reference_sha256"]}
    return torch.as_tensor(lookup[:29], dtype=torch.float32, device=device), lookup[29:], info


def sampled_batch(data: ReceiverTrainingData, numpy_rng, torch_rng,
                  travel: torch.Tensor, event_count: int, count: int):
    positions, values, sources, scales, travel_times = [], [], [], [], []
    for index in numpy_rng.choice(event_count, event_count, replace=False):
        x, y, src, active, rms = data.events[index]
        half = count // 2
        ids = torch.cat((active[torch.randint(len(active), (half,), device=x.device,
                                               generator=torch_rng)],
                         torch.randint(len(x), (count - half,), device=x.device,
                                       generator=torch_rng)))
        positions.append(x[ids])
        values.append(y[ids])
        sources.append(src.expand(count, 4))
        scales.append(torch.full((count,), rms, device=x.device))
        travel_times.append(travel[index, ids // 1001])
    return (torch.cat(positions), torch.cat(values), torch.cat(sources),
            torch.cat(scales), torch.cat(travel_times)[:, None])


def raw_prediction(model, path: Path, scale: np.ndarray,
                   travel: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    with np.load(path) as archive:
        fd = np.asarray(archive["data"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float32)
        receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
        source = np.asarray(archive["source_truth"], dtype=np.float32)
    source_t = torch.as_tensor(source / scale, device=device)
    receiver_t = torch.as_tensor(receivers / scale[:3], device=device)
    times_t = torch.as_tensor(times / scale[3], device=device)
    prediction = np.empty_like(fd)
    model.eval()
    with torch.inference_mode():
        for i in range(len(receivers)):
            points = torch.cat((receiver_t[i].expand(len(times), 3), times_t[:, None]), dim=1)
            condition = source_t.expand(len(times), 4)
            if travel is None:
                values = model(points, condition)
            else:
                arrival = torch.full((len(times), 1), float(travel[i]), device=device)
                values = frozen_head(model, points, condition, arrival)
            prediction[i] = values.squeeze(-1).cpu().numpy()
    return fd.astype(np.float64), prediction.astype(np.float64)


def waveform_error(fd: np.ndarray, prediction: np.ndarray) -> dict:
    active = np.abs(fd) >= .05 * np.max(np.abs(fd))
    return {"raw_l2": float(np.linalg.norm(prediction - fd) / np.linalg.norm(fd)),
            "active_l2": float(np.linalg.norm((prediction - fd)[active]) /
                               np.linalg.norm(fd[active]))}


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checkpoint, training, development = check(protocol)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    travel_train, travel_dev, fmm_cost = fmm_tables(protocol, training, development, device)
    config = SourceConditionedConfig()
    if protocol["seed"] != 42 or protocol["training_epochs"] != 5000:
        raise ValueError("Training budget changed from the registered pilot")
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol["seed"])
    base, metadata = load_conditioned_checkpoint(ROOT / protocol["base_receiver_checkpoint"], device)
    learned = EikonalTravelTimeCarrierWavePINN(
        5., config.time_length_s, eikonal_model=copy.deepcopy(base.eikonal_model),
        reference_velocity_m_s=3500., distance_core_m=20.,
        hidden_dim=protocol["hidden_dim"], num_layers=protocol["num_layers"],
        output_scale=protocol["output_scale"]).to(device)
    del base
    fmm = copy.deepcopy(learned)
    if any(not torch.equal(x, y) for x, y in zip(learned.net.parameters(),
                                                fmm.net.parameters())):
        raise ValueError("Paired heads did not start with identical parameters")
    train_data = ReceiverTrainingData(training, device, config, 5., seed=42)
    numpy_rng = np.random.default_rng(42)
    torch_rng = torch.Generator(device=device).manual_seed(42)
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
                  for name, model in (("learned", learned), ("fmm", fmm))}
    started = time.perf_counter()
    final_losses = {}
    for epoch in range(1, protocol["training_epochs"] + 1):
        points, target, source, rms, travel = sampled_batch(
            train_data, numpy_rng, torch_rng, travel_train, len(training),
            protocol["batch_size"] // len(training))
        for name, model in (("learned", learned), ("fmm", fmm)):
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad()
            prediction = (model(points, source) if name == "learned"
                          else frozen_head(model, points, source, travel))
            loss = ((prediction.squeeze(-1) / rms - target) ** 2).mean()
            (protocol["receiver_loss_weight"] * loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            final_losses[name] = float(loss.detach())
        if epoch == 1 or epoch % 500 == 0:
            print(f"epoch {epoch}/{protocol['training_epochs']}: receiver loss "
                  f"learned={final_losses['learned']:.5f}, fmm={final_losses['fmm']:.5f}",
                  flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                       dtype=np.float32)
    development_history = json.loads((ROOT / protocol["development_results"]).read_text())
    anchors = {item["name"]: ROOT / item["observation"] for item in
               json.loads((ROOT / protocol["development_protocol"]).read_text())["anchors"]}
    anchor_times = {name: travel_train[next(i for i, path in enumerate(training)
                                             if path.resolve() == anchor.resolve())].cpu().numpy()
                    for name, anchor in anchors.items()}
    anchors_data = {name: {mode: raw_prediction(model, path, scale,
                          anchor_times[name] if mode == "fmm" else None)
                           for mode, model in (("learned", learned), ("fmm", fmm))}
                    for name, path in anchors.items()}
    rows = []
    for item, path, arrival in zip(development_history["events"], development, travel_dev):
        variants = {}
        for mode, model in (("learned", learned), ("fmm", fmm)):
            fd, prediction = raw_prediction(model, path, scale,
                                            arrival if mode == "fmm" else None)
            fd_anchor, pred_anchor = anchors_data[item["anchor"]][mode]
            delta_fd, delta_pred = fd - fd_anchor, prediction - pred_anchor
            if np.linalg.norm(delta_fd) == 0 or np.linalg.norm(delta_pred) == 0:
                raise ValueError(f"Zero source-displacement response: {path}, {mode}")
            variants[mode] = {**waveform_error(fd, prediction),
                              "change_relative_l2": float(np.linalg.norm(delta_pred - delta_fd) /
                                                          np.linalg.norm(delta_fd)),
                              "change_cosine": float(np.sum(delta_fd * delta_pred) /
                                                     (np.linalg.norm(delta_fd) *
                                                      np.linalg.norm(delta_pred)))}
        rows.append({"event": item["file"], "sha256": item["sha256"],
                     "anchor": item["anchor"], "axis": item["axis"],
                     "sign": item["sign"], "models": variants})
        print(f"dev {item['file']}: learned L2={variants['learned']['raw_l2']:.3f}, "
              f"FMM L2={variants['fmm']['raw_l2']:.3f}", flush=True)
    training_metrics = {}
    for mode, model in (("learned", learned), ("fmm", fmm)):
        metrics = [waveform_error(*raw_prediction(model, path, scale,
                   travel_train[i].cpu().numpy() if mode == "fmm" else None))
                   for i, path in enumerate(training)]
        training_metrics[mode] = {"median_raw_l2": float(np.median([r["raw_l2"] for r in metrics])),
                                  "mean_raw_l2": float(np.mean([r["raw_l2"] for r in metrics]))}
    out = ROOT / protocol["output_directory"]
    out.mkdir(parents=True, exist_ok=False)
    for name, model in (("learned", learned), ("fmm", fmm)):
        torch.save({"head_state_dict": {key: value.detach().cpu()
                                        for key, value in model.net.state_dict().items()},
                    "seed": 42, "epoch": protocol["training_epochs"],
                    "protocol_sha256": hash_file(PROTOCOL),
                    "frozen_carrier_checkpoint_sha256": protocol["base_receiver_sha256"],
                    "input": name}, out / f"{name}_receiver_head.pth")
    report = {"role": protocol["role"], "protocol_sha256": hash_file(PROTOCOL),
              "training_records_sha256": [row["sha256"] for row in train_data.metadata],
              "development_records_sha256": [row["sha256"] for row in rows],
              "fmm": fmm_cost, "training_seconds_total_pair": training_seconds,
              "training_final_receiver_loss": final_losses,
              "training": training_metrics, "development": rows,
              "model_paths": {name: str((out / f"{name}_receiver_head.pth").resolve())
                              for name in ("learned", "fmm")},
              "limits": protocol["limits"]}
    result = ROOT / protocol["result_file"]
    with result.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
