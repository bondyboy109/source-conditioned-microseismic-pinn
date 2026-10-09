"""Train the source-conditioned PINN with frequency continuation.

Example:
    python train_source_conditioned_pinn.py --epochs 2000

Use --epochs 2 --n-phys 64 --n-ic 32 --n-bc-per-face 8 for a smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from receiver_training import ReceiverTrainingData

from source_conditioned_pinn import (
    HarmonicCarrierWavePINN,
    EikonalTravelTimeCarrierWavePINN,
    VelocityConditionedTravelTimeCarrierWavePINN,
    TravelTimeCarrierWavePINN,
    TranslationInvariantTravelTimeCarrierWavePINN,
    SourceConditionedConfig,
    SourceConditionedWavePINN,
    load_conditioned_checkpoint,
    save_conditioned_checkpoint,
    training_step,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_VELOCITY = ROOT / "data" / "velocity" / "vp_model.npy"
DEFAULT_OUTPUT = ROOT / "outputs" / "source_conditioned"


class GridVelocity:
    def __init__(self, path: Path, device: torch.device):
        array = np.load(path, mmap_mode="r")
        self.shape = array.shape
        # mmap arrays are read-only; copy before creating a writable tensor.
        self.values = torch.as_tensor(
            np.array(array, dtype=np.float32, copy=True), device=device
        )

    def __call__(self, xyz: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(
            [self.shape[0] - 1, self.shape[1] - 1, self.shape[2] - 1],
            device=xyz.device,
        )
        index = (xyz.clamp(0.0, 1.0) * scale).long()
        return self.values[index[:, 0], index[:, 1], index[:, 2]].unsqueeze(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity", type=Path, default=DEFAULT_VELOCITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequencies", type=float, nargs="+",
                        default=[5, 10, 15, 20, 30, 40, 60])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--n-phys", type=int, default=4096)
    parser.add_argument("--n-ic", type=int, default=2048)
    parser.add_argument("--n-bc-per-face", type=int, default=341)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden width of the trainable wavefield network")
    parser.add_argument("--num-layers", type=int, default=5,
                        help="Number of sinusoidal hidden layers")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--harmonic-carrier", action="store_true",
                        help="Experiment with frequency-specific sine/cosine output")
    parser.add_argument("--travel-time-carrier", action="store_true",
                        help="Diagnostic relative-source carrier with straight-ray time hint")
    parser.add_argument("--velocity-conditioned-carrier", action="store_true",
                        help="Add heterogeneous velocity-path features to the travel-time carrier")
    parser.add_argument("--translation-invariant-carrier", action="store_true",
                        help="Uniform-medium calibration carrier without absolute source coordinates")
    parser.add_argument("--reference-velocity", type=float, default=3500.0,
                        help="Fixed straight-ray time-hint velocity in m/s")
    parser.add_argument("--distance-core-m", type=float, default=1.0,
                        help="Smooth distance core for a travel-time carrier")
    parser.add_argument("--onset-gate", action="store_true",
                        help="Train with a source-dependent smooth earliest-signal gate")
    parser.add_argument("--onset-max-speed", type=float,
                        help="Maximum velocity in m/s for the onset gate")
    parser.add_argument("--onset-support-m", type=float, default=200.0)
    parser.add_argument("--onset-lead-s", type=float, default=0.20)
    parser.add_argument("--onset-rise-s", type=float, default=0.10)
    parser.add_argument("--ray-travel-time", action="store_true",
                        help="Integrate slowness along source-to-point paths using the velocity grid")
    parser.add_argument("--ray-samples", type=int, default=16)
    parser.add_argument("--eikonal-checkpoint", type=Path,
                        help="Frozen differentiable source-conditioned traveltime checkpoint")
    parser.add_argument("--wavefield-scale", type=float, default=1.0,
                        help="Physical output per unit network output; fixed before validation")
    parser.add_argument("--hard-initial", action="store_true",
                        help="Experimental zero initial-value/derivative carrier gate")
    parser.add_argument("--source-near-fraction", type=float, default=0.25)
    parser.add_argument("--source-exact-fraction", type=float, default=0.0,
                        help="Fraction of physics points placed exactly at sampled sources")
    parser.add_argument("--source-spatial-jitter-sigma", type=float, default=2.0)
    parser.add_argument("--source-temporal-half-width-periods", type=float, default=2.0)
    parser.add_argument("--receiver-observations", type=Path, nargs="+", default=[])
    parser.add_argument("--receiver-batch-size", type=int, default=1024)
    parser.add_argument("--receiver-events-per-batch", type=int, default=1,
                        help="Distinct training events represented in every receiver batch")
    parser.add_argument("--receiver-loss-weight", type=float, default=0.1)
    parser.add_argument("--balance-loss-gradients", action="store_true",
                        help="Balance last-layer physics/condition gradients against receiver gradients")
    parser.add_argument("--receiver-only", action="store_true",
                        help="Supervised representation diagnostic; skip all PDE/IC/BC terms")
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="Initialize weights from a compatible source-conditioned checkpoint")
    parser.add_argument("--allow-reference-velocity-mismatch", action="store_true",
                        help="Controlled sweep: replace a travel-time carrier reference speed after loading")
    return parser.parse_args()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requested_architecture(args) -> str:
    if getattr(args, "velocity_conditioned_carrier", False):
        return "velocity_conditioned_travel_time_carrier"
    if args.eikonal_checkpoint:
        return "eikonal_travel_time_carrier"
    if args.translation_invariant_carrier:
        return "translation_invariant_travel_time_carrier"
    if args.travel_time_carrier:
        return "travel_time_carrier"
    if args.harmonic_carrier:
        return "harmonic_carrier"
    return "standard"


def requested_onset_gate(args) -> dict[str, float] | None:
    if not getattr(args, "onset_gate", False):
        return None
    return {"maximum_speed_m_s": args.onset_max_speed,
            "support_m": args.onset_support_m,
            "lead_s": args.onset_lead_s,
            "rise_s": args.onset_rise_s}


def validate_initial_checkpoint(payload: dict, args, cfg: SourceConditionedConfig) -> None:
    architecture = payload.get("architecture", {})
    if architecture.get("kind", "standard") != requested_architecture(args):
        raise ValueError("Initial checkpoint architecture does not match the requested architecture")
    if not np.isclose(float(payload.get("frequency_hz", np.nan)), args.frequencies[0]):
        raise ValueError("Initial checkpoint frequency does not match the requested frequency")
    stored = payload.get("config", {})
    for key in ("x_length_m", "y_length_m", "z_length_m", "time_length_s"):
        if not np.isclose(float(stored.get(key, np.nan)), float(getattr(cfg, key))):
            raise ValueError(f"Initial checkpoint domain mismatch: {key}")
    if not np.isclose(float(architecture.get("output_scale", np.nan)), args.wavefield_scale):
        raise ValueError("Initial checkpoint output scale does not match --wavefield-scale")
    if bool(architecture.get("hard_initial", False)) != args.hard_initial:
        raise ValueError("Initial checkpoint hard-initial setting does not match")
    stored_gate = architecture.get("onset_gate")
    desired_gate = requested_onset_gate(args)
    if ((stored_gate is None) != (desired_gate is None) or
            (stored_gate is not None and (set(stored_gate) != set(desired_gate)
             or any(not np.isclose(stored_gate[key], value, rtol=0, atol=1e-9)
                    for key, value in desired_gate.items())))):
        raise ValueError("Initial checkpoint onset gate does not match")
    if requested_architecture(args) in (
            "travel_time_carrier", "translation_invariant_travel_time_carrier"):
        if (not getattr(args, "allow_reference_velocity_mismatch", False) and
                not np.isclose(float(architecture.get("reference_velocity_m_s", np.nan)),
                               args.reference_velocity)):
            raise ValueError("Initial checkpoint reference velocity does not match")
    if requested_architecture(args) in (
            "travel_time_carrier", "translation_invariant_travel_time_carrier"):
        default_core = (1.0 if requested_architecture(args)
                        == "translation_invariant_travel_time_carrier" else 0.0)
        if not np.isclose(float(architecture.get("distance_core_m", default_core)),
                          args.distance_core_m):
            raise ValueError("Initial checkpoint distance core does not match")
        if bool(architecture.get("ray_travel_time", False)) != args.ray_travel_time:
            raise ValueError("Initial checkpoint ray-travel-time setting does not match")
        if args.ray_travel_time and int(architecture.get("ray_samples", 16)) != args.ray_samples:
            raise ValueError("Initial checkpoint ray sample count does not match")


def main():
    args = parse_args()
    if not np.isfinite(args.wavefield_scale) or args.wavefield_scale <= 0:
        raise ValueError("--wavefield-scale must be finite and positive")
    if args.hidden_dim < 8 or args.num_layers < 1:
        raise ValueError("--hidden-dim must be >=8 and --num-layers must be positive")
    if sum((args.harmonic_carrier, args.travel_time_carrier,
            args.translation_invariant_carrier)) > 1:
        raise ValueError("Choose only one carrier architecture")
    if args.hard_initial and not (args.harmonic_carrier or args.travel_time_carrier
                                  or args.translation_invariant_carrier):
        raise ValueError("--hard-initial requires a carrier architecture")
    if not np.isfinite(args.reference_velocity) or args.reference_velocity <= 0:
        raise ValueError("--reference-velocity must be finite and positive")
    if not np.isfinite(args.distance_core_m) or args.distance_core_m < 0:
        raise ValueError("--distance-core-m must be finite and nonnegative")
    if args.ray_samples < 2:
        raise ValueError("--ray-samples must be at least 2")
    if args.ray_travel_time and not args.travel_time_carrier:
        raise ValueError("--ray-travel-time requires --travel-time-carrier")
    if args.eikonal_checkpoint and not args.travel_time_carrier:
        raise ValueError("--eikonal-checkpoint requires --travel-time-carrier")
    if args.velocity_conditioned_carrier and not args.travel_time_carrier:
        raise ValueError("--velocity-conditioned-carrier requires --travel-time-carrier")
    if args.velocity_conditioned_carrier and args.eikonal_checkpoint is None:
        raise ValueError("--velocity-conditioned-carrier requires --eikonal-checkpoint")
    if args.eikonal_checkpoint and args.ray_travel_time:
        raise ValueError("Choose either the eikonal network or straight-ray integration")
    if args.ray_travel_time and args.receiver_only is False and args.distance_core_m <= 0:
        raise ValueError("Ray travel-time physics training requires a positive distance core")
    if not 0 <= args.source_near_fraction <= 1:
        raise ValueError("--source-near-fraction must be between 0 and 1")
    if not 0 <= args.source_exact_fraction <= args.source_near_fraction:
        raise ValueError("--source-exact-fraction must lie between 0 and source-near-fraction")
    if args.source_spatial_jitter_sigma <= 0 or args.source_temporal_half_width_periods <= 0:
        raise ValueError("Source sampling widths must be positive")
    if args.receiver_observations and len(args.frequencies) != 1:
        raise ValueError("Receiver waveforms are frequency-specific; use one frequency per run")
    if args.receiver_observations and (args.receiver_batch_size < 2 or args.receiver_loss_weight <= 0):
        raise ValueError("Receiver batch size must be >=2 and weight positive")
    if args.receiver_observations and not 1 <= args.receiver_events_per_batch <= len(args.receiver_observations):
        raise ValueError("--receiver-events-per-batch must lie within the observation count")
    if args.receiver_observations and args.receiver_batch_size < 2 * args.receiver_events_per_batch:
        raise ValueError("Receiver batch size must provide at least two samples per event")
    if args.balance_loss_gradients and not args.receiver_observations:
        raise ValueError("--balance-loss-gradients requires --receiver-observations")
    if args.receiver_only and not args.receiver_observations:
        raise ValueError("--receiver-only requires --receiver-observations")
    if args.receiver_only and args.balance_loss_gradients:
        raise ValueError("--receiver-only and --balance-loss-gradients cannot be combined")
    if args.travel_time_carrier and not args.receiver_only and args.distance_core_m <= 0:
        raise ValueError("Physics training requires a positive travel-distance core")
    if args.onset_gate:
        gate = requested_onset_gate(args)
        if (not args.travel_time_carrier or args.distance_core_m <= 0
                or any(value is None or not np.isfinite(value) for value in gate.values())
                or gate["maximum_speed_m_s"] <= 0 or gate["support_m"] < 0
                or gate["lead_s"] < 0 or gate["rise_s"] <= 0):
            raise ValueError("Onset gate needs --travel-time-carrier, positive core and valid parameters")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output.mkdir(parents=True, exist_ok=True)
    velocity = (GridVelocity(args.velocity, device)
                if (not args.receiver_only or args.ray_travel_time) else None)
    cfg = replace(
        SourceConditionedConfig(),
        source_near_fraction=args.source_near_fraction,
        source_exact_fraction=args.source_exact_fraction,
        source_spatial_jitter_sigma=args.source_spatial_jitter_sigma,
        source_temporal_half_width_periods=args.source_temporal_half_width_periods,
    )
    initial_record = None
    if args.initial_checkpoint:
        initial_path = args.initial_checkpoint.resolve()
        model, initial_payload = load_conditioned_checkpoint(initial_path, device)
        validate_initial_checkpoint(initial_payload, args, cfg)
        if (args.allow_reference_velocity_mismatch and
                hasattr(model, "reference_velocity_m_s")):
            model.reference_velocity_m_s = float(args.reference_velocity)
        initial_record = {
            "path": str(initial_path),
            "sha256": checkpoint_sha256(initial_path),
            "epoch": int(initial_payload["epoch"]),
            "training_type": initial_payload.get("training_record", {}).get(
                "training_type"),
        }
    else:
        carrier_options = {"output_scale": args.wavefield_scale,
                           "hard_initial": args.hard_initial,
                           "hidden_dim": args.hidden_dim,
                           "num_layers": args.num_layers}
        if args.translation_invariant_carrier:
            model = TranslationInvariantTravelTimeCarrierWavePINN(
                args.frequencies[0], cfg.time_length_s,
                reference_velocity_m_s=args.reference_velocity,
                x_length_m=cfg.x_length_m, y_length_m=cfg.y_length_m,
                z_length_m=cfg.z_length_m,
                distance_core_m=args.distance_core_m,
                ray_travel_time=args.ray_travel_time,
                ray_samples=args.ray_samples, **carrier_options)
        elif args.travel_time_carrier:
            model_class = TravelTimeCarrierWavePINN
            extra_options = {}
            if args.eikonal_checkpoint:
                from eikonal_traveltime import load_eikonal_checkpoint
                eikonal, _ = load_eikonal_checkpoint(args.eikonal_checkpoint, device)
                model_class = (VelocityConditionedTravelTimeCarrierWavePINN
                               if args.velocity_conditioned_carrier
                               else EikonalTravelTimeCarrierWavePINN)
                extra_options["eikonal_model"] = eikonal
            model = model_class(
                args.frequencies[0], cfg.time_length_s,
                reference_velocity_m_s=args.reference_velocity,
                x_length_m=cfg.x_length_m, y_length_m=cfg.y_length_m,
                z_length_m=cfg.z_length_m,
                distance_core_m=args.distance_core_m,
                ray_travel_time=args.ray_travel_time,
                ray_samples=args.ray_samples,
                onset_gate=requested_onset_gate(args),
                **extra_options, **carrier_options)
        elif args.harmonic_carrier:
            model = HarmonicCarrierWavePINN(args.frequencies[0], cfg.time_length_s,
                                            **carrier_options)
        else:
            model = SourceConditionedWavePINN(
                hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                output_scale=args.wavefield_scale)
        model = model.to(device)
    if args.ray_travel_time or args.velocity_conditioned_carrier:
        model.set_velocity_grid(velocity.values)
    observations = (ReceiverTrainingData(args.receiver_observations, device, cfg,
                                        args.frequencies[0], seed=args.seed)
                    if args.receiver_observations else None)
    summary = {
        "run": {
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "seed": args.seed,
            "parameter_count": sum(p.numel() for p in model.parameters()
                                   if p.requires_grad),
            "frozen_parameter_count": sum(p.numel() for p in model.parameters()
                                          if not p.requires_grad),
            "n_phys": 0 if args.receiver_only else args.n_phys,
            "n_ic": 0 if args.receiver_only else args.n_ic,
            "n_bc_per_face": 0 if args.receiver_only else args.n_bc_per_face,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "architecture": requested_architecture(args),
            "reference_velocity_m_s": (args.reference_velocity
                                         if args.travel_time_carrier
                                         or args.translation_invariant_carrier else None),
            "reference_velocity_override": bool(args.allow_reference_velocity_mismatch),
            "distance_core_m": (args.distance_core_m
                                 if args.travel_time_carrier
                                 or args.translation_invariant_carrier else None),
            "ray_travel_time": args.ray_travel_time,
            "ray_samples": args.ray_samples if args.ray_travel_time else None,
            "eikonal_checkpoint": (str(args.eikonal_checkpoint.resolve())
                                    if args.eikonal_checkpoint else None),
            "wavefield_scale": args.wavefield_scale,
            "hard_initial": args.hard_initial,
            "onset_gate": requested_onset_gate(args),
            "source_near_fraction": args.source_near_fraction,
            "source_exact_fraction": args.source_exact_fraction,
            "source_spatial_jitter_sigma": args.source_spatial_jitter_sigma,
            "source_temporal_half_width_periods": args.source_temporal_half_width_periods,
            "receiver_observations": observations.metadata if observations else [],
            "receiver_batch_size": args.receiver_batch_size if observations else None,
            "receiver_sampling_seed": args.seed if observations else None,
            "receiver_events_per_batch": (args.receiver_events_per_batch
                                           if observations else None),
            "receiver_loss_weight": args.receiver_loss_weight if observations else None,
            "balance_loss_gradients": args.balance_loss_gradients,
            "receiver_only": args.receiver_only,
            "training_type": ("supervised_receiver_diagnostic" if args.receiver_only
                              else "hybrid_physics_receiver" if observations
                              else "physics_only"),
            "initial_checkpoint": initial_record,
        }
    }

    for frequency in args.frequencies:
        if (args.harmonic_carrier or args.travel_time_carrier
                or args.translation_invariant_carrier):
            model.frequency_hz = float(frequency)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate
        )
        history = []
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            metrics = training_step(
                model=model,
                optimizer=optimizer,
                velocity_lookup=velocity,
                frequency=frequency,
                n_phys=args.n_phys,
                n_ic=args.n_ic,
                n_bc_per_face=args.n_bc_per_face,
                device=device,
                cfg=cfg,
                receiver_batch=observations.batch(
                    args.receiver_batch_size, args.receiver_events_per_batch)
                if observations else None,
                receiver_loss_weight=args.receiver_loss_weight if observations else 0.0,
                balance_loss_gradients=args.balance_loss_gradients,
                receiver_only=args.receiver_only,
            )
            history.append(metrics)
            if epoch == 1 or epoch % 100 == 0:
                print(
                    f"{frequency:g} Hz epoch {epoch}/{args.epochs}: "
                    f"total={metrics['total']:.6e}"
                )
            if epoch % args.save_every == 0:
                save_conditioned_checkpoint(
                    args.output / f"conditioned_{frequency:g}Hz_epoch_{epoch}.pth",
                    model, optimizer, frequency, epoch, cfg, history, summary["run"]
                )

        elapsed = time.perf_counter() - started
        final_path = args.output / f"conditioned_{frequency:g}Hz.pth"
        save_conditioned_checkpoint(
            final_path, model, optimizer, frequency, args.epochs, cfg, history,
            summary["run"]
        )
        summary[f"{frequency:g}Hz"] = {
            "final": history[-1],
            "seconds": elapsed,
            "checkpoint": str(final_path),
            "source_conditioned": True,
            "input_dim": 8,
        }
        if device.type == "cuda":
            summary[f"{frequency:g}Hz"]["peak_cuda_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
        with (args.output / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
