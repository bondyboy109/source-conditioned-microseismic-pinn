"""Source-conditioned acoustic PINN for source-parameter inversion.

This module is a replacement interface for the four-input wavefield PINN.
The network input is [x, y, z, t, xs, ys, zs, ts], all normalized to [0, 1].
The source parameters are sampled during training and remain differentiable
during inversion. Existing four-input checkpoints are intentionally
incompatible and must not be loaded into this model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features,
                                            1.0 / in_features)
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SourceConditionedWavePINN(nn.Module):
    """u_theta(x,y,z,t;xs,ys,zs,ts)."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 5,
                 output_scale: float = 1.0):
        super().__init__()
        if not math.isfinite(output_scale) or output_scale <= 0:
            raise ValueError("output_scale must be finite and positive")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_scale = float(output_scale)
        layers = [SineLayer(8, hidden_dim, is_first=True)]
        layers.extend(
            SineLayer(hidden_dim, hidden_dim, is_first=False)
            for _ in range(num_layers - 1)
        )
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, space_time: torch.Tensor,
                source: torch.Tensor) -> torch.Tensor:
        if space_time.shape[-1] != 4 or source.shape[-1] != 4:
            raise ValueError("space_time and source must each have four columns")
        return self.output_scale * self.net(torch.cat((space_time, source), dim=-1))


class HarmonicCarrierWavePINN(SourceConditionedWavePINN):
    """Experimental frequency-specific carrier with learned spatial envelopes."""

    def __init__(self, frequency_hz: float, time_length_s: float,
                 hidden_dim: int = 128, num_layers: int = 5,
                 output_scale: float = 1.0, hard_initial: bool = False):
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers,
                         output_scale=output_scale)
        self.net[-1] = nn.Linear(hidden_dim, 2)
        self.frequency_hz = float(frequency_hz)
        self.time_length_s = float(time_length_s)
        self.hard_initial = bool(hard_initial)

    def forward(self, space_time: torch.Tensor,
                source: torch.Tensor) -> torch.Tensor:
        if space_time.shape[-1] != 4 or source.shape[-1] != 4:
            raise ValueError("space_time and source must each have four columns")
        amplitude = self.net(torch.cat((space_time, source), dim=-1))
        phase = (2.0 * math.pi * self.frequency_hz * self.time_length_s
                 * (space_time[:, 3:4] - source[:, 3:4]))
        wavefield = self.output_scale * (
            amplitude[:, :1] * torch.cos(phase)
            + amplitude[:, 1:] * torch.sin(phase))
        if self.hard_initial:
            physical_t = space_time[:, 3:4] * self.time_length_s
            gate = (physical_t / (physical_t + 0.1)).square()
            wavefield = gate * wavefield
        return wavefield


class TravelTimeCarrierWavePINN(HarmonicCarrierWavePINN):
    """Diagnostic carrier with relative coordinates and a straight-ray time hint."""

    def __init__(self, frequency_hz: float, time_length_s: float,
                 reference_velocity_m_s: float = 3500.0,
                 x_length_m: float = 16000.0, y_length_m: float = 16000.0,
                 z_length_m: float = 3720.0, distance_core_m: float = 0.0,
                 ray_travel_time: bool = False, ray_samples: int = 16,
                 onset_gate: dict[str, float] | None = None,
                 **kwargs):
        super().__init__(frequency_hz, time_length_s, **kwargs)
        if not math.isfinite(reference_velocity_m_s) or reference_velocity_m_s <= 0:
            raise ValueError("reference_velocity_m_s must be finite and positive")
        self.reference_velocity_m_s = float(reference_velocity_m_s)
        if not math.isfinite(distance_core_m) or distance_core_m < 0:
            raise ValueError("distance_core_m must be finite and nonnegative")
        self.distance_core_m = float(distance_core_m)
        if ray_samples < 2:
            raise ValueError("ray_samples must be at least 2")
        self.ray_travel_time = bool(ray_travel_time)
        self.ray_samples = int(ray_samples)
        self.velocity_grid = None
        self.spatial_lengths_m = (float(x_length_m), float(y_length_m),
                                  float(z_length_m))
        if any(not math.isfinite(length) or length <= 0
               for length in self.spatial_lengths_m):
               raise ValueError("Spatial lengths must be finite and positive")
        self.onset_gate = dict(onset_gate) if onset_gate is not None else None
        if self.onset_gate is not None:
            if (set(self.onset_gate) != {"maximum_speed_m_s", "support_m", "lead_s", "rise_s"}
                    or any(not math.isfinite(value) for value in self.onset_gate.values())
                    or self.onset_gate["maximum_speed_m_s"] <= 0
                    or self.onset_gate["support_m"] < 0
                    or self.onset_gate["lead_s"] < 0
                    or self.onset_gate["rise_s"] <= 0
                    or self.distance_core_m <= 0):
                raise ValueError("Onset gate requires finite parameters and a positive distance core")

    def _apply_onset_gate(self, wavefield: torch.Tensor, space_time: torch.Tensor,
                          source: torch.Tensor) -> torch.Tensor:
        if self.onset_gate is None:
            return wavefield
        lengths = space_time.new_tensor(self.spatial_lengths_m)
        squared = (((space_time[:, :3] - source[:, :3]) * lengths).square()
                   .sum(dim=-1, keepdim=True))
        distance_m = torch.sqrt(squared + self.distance_core_m ** 2) - self.distance_core_m
        earliest_s = (source[:, 3:4] * self.time_length_s
                      - self.onset_gate["lead_s"]
                      + torch.clamp(distance_m - self.onset_gate["support_m"], min=0)
                      / self.onset_gate["maximum_speed_m_s"])
        phase = torch.clamp((space_time[:, 3:4] * self.time_length_s - earliest_s)
                            / self.onset_gate["rise_s"], 0, 1)
        gate = phase**3 * (10 + phase * (-15 + 6 * phase))
        return wavefield * gate

    def set_velocity_grid(self, values: torch.Tensor) -> None:
        if values.ndim != 3 or not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Velocity grid must be a finite positive 3-D tensor")
        # grid_sample expects [depth, height, width] and coordinates [x,y,z].
        self.velocity_grid = values.detach().to(dtype=torch.float32,
                                                device=next(self.parameters()).device
                                                ).permute(2, 1, 0).contiguous()[None, None]

    def _travel_time(self, relative_xyz: torch.Tensor,
                     source: torch.Tensor) -> torch.Tensor:
        lengths = relative_xyz.new_tensor(self.spatial_lengths_m)
        squared_distance_m = (relative_xyz * lengths).square().sum(
            dim=-1, keepdim=True)
        if self.distance_core_m > 0:
            distance_m = torch.sqrt(squared_distance_m + self.distance_core_m ** 2)
            distance_m = distance_m - self.distance_core_m
        else:
            distance_m = torch.sqrt(squared_distance_m)
        if not self.ray_travel_time:
            return distance_m / self.reference_velocity_m_s
        if self.velocity_grid is None:
            raise RuntimeError("ray_travel_time requires set_velocity_grid()")
        fractions = (torch.arange(self.ray_samples, device=relative_xyz.device,
                                  dtype=relative_xyz.dtype) + 0.5) / self.ray_samples
        points = source[:, None, :3] + fractions[None, :, None] * relative_xyz[:, None, :]
        grid = points.reshape(1, 1, 1, -1, 3) * 2.0 - 1.0
        sampled = F.grid_sample(self.velocity_grid, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        sampled = sampled.reshape(-1, self.ray_samples)
        return distance_m * torch.mean(1.0 / sampled, dim=1, keepdim=True)

    def forward(self, space_time: torch.Tensor,
                source: torch.Tensor) -> torch.Tensor:
        if space_time.shape[-1] != 4 or source.shape[-1] != 4:
            raise ValueError("space_time and source must each have four columns")
        relative_xyz = space_time[:, :3] - source[:, :3]
        lengths = space_time.new_tensor(self.spatial_lengths_m)
        squared_distance_m = (relative_xyz * lengths).square().sum(
            dim=-1, keepdim=True)
        if self.distance_core_m > 0:
            core = self.distance_core_m
            distance_m = torch.sqrt(squared_distance_m + core ** 2) - core
        else:
            # Retain exact legacy checkpoint behaviour for supervised audits.
            distance_m = torch.sqrt(squared_distance_m)
        travel_s = self._travel_time(relative_xyz, source)
        delayed_time = (space_time[:, 3:4] - source[:, 3:4]
                        - travel_s / self.time_length_s)
        features = torch.cat((relative_xyz, delayed_time, source[:, :3],
                              travel_s / self.time_length_s), dim=-1)
        amplitude = self.net(features)
        phase = 2.0 * math.pi * self.frequency_hz * self.time_length_s * delayed_time
        wavefield = self.output_scale * (
            amplitude[:, :1] * torch.cos(phase)
            + amplitude[:, 1:] * torch.sin(phase))
        if self.hard_initial:
            physical_t = space_time[:, 3:4] * self.time_length_s
            gate = (physical_t / (physical_t + 0.1)).square()
            wavefield = gate * wavefield
        return self._apply_onset_gate(wavefield, space_time, source)


class TranslationInvariantTravelTimeCarrierWavePINN(TravelTimeCarrierWavePINN):
    """Uniform-medium calibration model using only relative source features."""

    def __init__(self, *args, distance_core_m: float = 1.0, **kwargs):
        super().__init__(*args, distance_core_m=distance_core_m, **kwargs)
        if not math.isfinite(distance_core_m) or distance_core_m <= 0:
            raise ValueError("distance_core_m must be finite and positive")
        self.net[0] = SineLayer(5, self.hidden_dim, is_first=True)

    def forward(self, space_time: torch.Tensor,
                source: torch.Tensor) -> torch.Tensor:
        if space_time.shape[-1] != 4 or source.shape[-1] != 4:
            raise ValueError("space_time and source must each have four columns")
        relative_xyz = space_time[:, :3] - source[:, :3]
        lengths = space_time.new_tensor(self.spatial_lengths_m)
        squared_distance_m = (relative_xyz * lengths).square().sum(dim=-1,
                                                                     keepdim=True)
        # A finite core keeps second derivatives bounded at source coincidence.
        # It is checkpointed because it changes both phase and curvature.
        core = self.distance_core_m
        distance_m = torch.sqrt(squared_distance_m + core ** 2) - core
        travel_s = distance_m / self.reference_velocity_m_s
        delayed_time = (space_time[:, 3:4] - source[:, 3:4]
                        - travel_s / self.time_length_s)
        features = torch.cat((relative_xyz, delayed_time,
                              travel_s / self.time_length_s), dim=-1)
        amplitude = self.net(features)
        phase = 2.0 * math.pi * self.frequency_hz * self.time_length_s * delayed_time
        wavefield = self.output_scale * (
            amplitude[:, :1] * torch.cos(phase)
            + amplitude[:, 1:] * torch.sin(phase))
        if self.hard_initial:
            physical_t = space_time[:, 3:4] * self.time_length_s
            gate = (physical_t / (physical_t + 0.1)).square()
            wavefield = gate * wavefield
        return self._apply_onset_gate(wavefield, space_time, source)


class EikonalTravelTimeCarrierWavePINN(TravelTimeCarrierWavePINN):
    """Carrier whose differentiable time hint comes from a frozen eikonal net."""

    def __init__(self, *args, eikonal_model: nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.eikonal_model = eikonal_model
        for parameter in self.eikonal_model.parameters():
            parameter.requires_grad_(False)

    def _travel_time(self, relative_xyz: torch.Tensor,
                     source: torch.Tensor) -> torch.Tensor:
        xyz = source[:, :3] + relative_xyz
        return self.eikonal_model(xyz, source[:, :3])


class VelocityConditionedTravelTimeCarrierWavePINN(EikonalTravelTimeCarrierWavePINN):
    """Travel-time carrier augmented with sampled heterogeneous velocities.

    The path samples are diagnostic local features for the trainable wavefield
    operator. They do not replace the frozen eikonal travel-time estimate.
    """

    path_sample_count = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net[0] = SineLayer(18, self.hidden_dim, is_first=True)

    def _velocity_features(self, relative_xyz: torch.Tensor,
                           source: torch.Tensor) -> torch.Tensor:
        if self.velocity_grid is None:
            raise RuntimeError("Velocity-conditioned carrier requires set_velocity_grid()")
        fractions = (torch.arange(self.path_sample_count,
                                  device=relative_xyz.device,
                                  dtype=relative_xyz.dtype) + 0.5) / self.path_sample_count
        points = source[:, None, :3] + fractions[None, :, None] * relative_xyz[:, None, :]
        grid = points.reshape(1, 1, 1, -1, 3) * 2.0 - 1.0
        sampled = F.grid_sample(self.velocity_grid, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        sampled = sampled.reshape(-1, self.path_sample_count)
        endpoint = torch.cat((source[:, None, :3],
                              source[:, None, :3] + relative_xyz[:, None, :]), dim=1)
        endpoint_grid = endpoint.reshape(1, 1, 1, -1, 3) * 2.0 - 1.0
        endpoint_velocity = F.grid_sample(
            self.velocity_grid, endpoint_grid, mode="bilinear",
            padding_mode="border", align_corners=True).reshape(-1, 2)
        # PyTorch does not provide second derivatives for 3-D grid_sample.
        # Treat interpolated velocity as a frozen operator feature so the
        # acoustic PDE can still differentiate the wavefield twice.
        return (torch.cat((sampled, endpoint_velocity), dim=-1) / 6000.0).detach()

    def forward(self, space_time: torch.Tensor,
                source: torch.Tensor) -> torch.Tensor:
        if space_time.shape[-1] != 4 or source.shape[-1] != 4:
            raise ValueError("space_time and source must each have four columns")
        relative_xyz = space_time[:, :3] - source[:, :3]
        lengths = space_time.new_tensor(self.spatial_lengths_m)
        squared_distance_m = (relative_xyz * lengths).square().sum(
            dim=-1, keepdim=True)
        core = self.distance_core_m
        distance_m = torch.sqrt(squared_distance_m + core ** 2) - core
        travel_s = self._travel_time(relative_xyz, source)
        delayed_time = (space_time[:, 3:4] - source[:, 3:4]
                        - travel_s / self.time_length_s)
        features = torch.cat((relative_xyz, delayed_time, source[:, :3],
                              travel_s / self.time_length_s,
                              self._velocity_features(relative_xyz, source)), dim=-1)
        amplitude = self.net(features)
        phase = 2.0 * math.pi * self.frequency_hz * self.time_length_s * delayed_time
        wavefield = self.output_scale * (
            amplitude[:, :1] * torch.cos(phase)
            + amplitude[:, 1:] * torch.sin(phase))
        if self.hard_initial:
            physical_t = space_time[:, 3:4] * self.time_length_s
            gate = (physical_t / (physical_t + 0.1)).square()
            wavefield = gate * wavefield
        return self._apply_onset_gate(wavefield, space_time, source)


@dataclass(frozen=True)
class SourceConditionedConfig:
    source_sigma_m: float = 50.0
    source_near_fraction: float = 0.25
    source_exact_fraction: float = 0.0
    source_spatial_jitter_sigma: float = 2.0
    source_temporal_half_width_periods: float = 2.0
    lambda_phys: float = 1.0
    lambda_ic: float = 10.0
    lambda_bc: float = 10.0
    source_time_min: float = 0.05 / 9.0
    source_time_max: float = 0.50 / 9.0
    source_x_min: float = 2000.0 / 16000.0
    source_x_max: float = 14000.0 / 16000.0
    source_y_min: float = 2000.0 / 16000.0
    source_y_max: float = 14000.0 / 16000.0
    source_z_min: float = 100.0 / 3720.0
    source_z_max: float = 1500.0 / 3720.0
    x_length_m: float = 16000.0
    y_length_m: float = 16000.0
    z_length_m: float = 3720.0
    time_length_s: float = 9.0


def ricker(t: torch.Tensor, frequency: float) -> torch.Tensor:
    a = (math.pi * frequency) ** 2
    return (1.0 - 2.0 * a * t.square()) * torch.exp(-a * t.square())


def sample_source_condition(n: int, device: torch.device,
                            cfg: SourceConditionedConfig | None = None) -> torch.Tensor:
    """Sample source parameters independently for a training batch."""
    cfg = cfg or SourceConditionedConfig()
    source = torch.rand(n, 4, device=device)
    source[:, 0] = (
        cfg.source_x_min
        + source[:, 0] * (cfg.source_x_max - cfg.source_x_min)
    )
    source[:, 1] = (
        cfg.source_y_min
        + source[:, 1] * (cfg.source_y_max - cfg.source_y_min)
    )
    source[:, 2] = (
        cfg.source_z_min
        + source[:, 2] * (cfg.source_z_max - cfg.source_z_min)
    )
    source[:, 3] = (
        cfg.source_time_min
        + source[:, 3] * (cfg.source_time_max - cfg.source_time_min)
    )
    return source


def source_term(space_time: torch.Tensor, source: torch.Tensor,
                frequency: float, cfg: SourceConditionedConfig) -> torch.Tensor:
    x, y, z, t = space_time.split(1, dim=-1)
    xs, ys, zs, ts = source.split(1, dim=-1)
    dx = (x - xs) * cfg.x_length_m
    dy = (y - ys) * cfg.y_length_m
    dz = (z - zs) * cfg.z_length_m
    spatial = torch.exp(
        -(dx.square() + dy.square() + dz.square())
        / (2.0 * cfg.source_sigma_m ** 2)
    )
    return spatial * ricker((t - ts) * cfg.time_length_s, frequency)


def physics_loss(model: SourceConditionedWavePINN, space_time: torch.Tensor,
                 source: torch.Tensor, velocity: torch.Tensor,
                 frequency: float, cfg: SourceConditionedConfig) -> torch.Tensor:
    """Compute the acoustic residual while preserving source conditioning."""
    space_time = space_time.clone().detach().requires_grad_(True)
    u = model(space_time, source)
    grads = torch.autograd.grad(
        u, space_time, torch.ones_like(u), create_graph=True
    )[0]
    second = []
    for axis in range(4):
        first_axis = grads[:, axis:axis + 1]
        second_axis = torch.autograd.grad(
            first_axis, space_time, torch.ones_like(first_axis),
            create_graph=True, retain_graph=True
        )[0][:, axis:axis + 1]
        second.append(second_axis)
    u_xx, u_yy, u_zz, u_tt = second
    laplacian = (
        u_xx / cfg.x_length_m ** 2
        + u_yy / cfg.y_length_m ** 2
        + u_zz / cfg.z_length_m ** 2
    )
    residual = u_tt / cfg.time_length_s ** 2 - velocity.square() * laplacian
    residual = residual - source_term(space_time, source, frequency, cfg)
    return residual.square().mean()


def initial_condition_loss(model: SourceConditionedWavePINN,
                           xyz: torch.Tensor, source: torch.Tensor,
                           frequency: float,
                           cfg: SourceConditionedConfig) -> torch.Tensor:
    """Enforce zero initial displacement and velocity on comparable scales.

    Autograd differentiates with respect to normalized time. Dividing by
    ``2*pi*f*T`` converts that derivative to physical time and normalizes it
    by the characteristic angular frequency, so the displacement and velocity
    penalties are both dimensionless wave-amplitude terms.
    """
    time = torch.zeros(xyz.shape[0], 1, device=xyz.device)
    space_time = torch.cat((xyz, time), dim=-1).requires_grad_(True)
    u = model(space_time, source) / model.output_scale
    u_t = torch.autograd.grad(
        u, space_time, torch.ones_like(u), create_graph=True
    )[0][:, 3:4]
    normalized_velocity = u_t / (2.0 * math.pi * frequency * cfg.time_length_s)
    return u.square().mean() + normalized_velocity.square().mean()


def boundary_condition_loss(model: SourceConditionedWavePINN,
                            n_per_face: int, device: torch.device,
                            cfg: SourceConditionedConfig) -> torch.Tensor:
    """Enforce the current zero-Dirichlet condition on all six faces."""
    losses = []
    for axis in range(3):
        for value in (0.0, 1.0):
            space_time = torch.rand(n_per_face, 4, device=device)
            space_time[:, axis] = value
            source = sample_source_condition(n_per_face, device, cfg)
            losses.append((model(space_time, source) / model.output_scale).square().mean())
    return torch.stack(losses).mean()


def gradient_balance_weight(loss: torch.Tensor, reference: torch.Tensor,
                            parameters: tuple[torch.nn.Parameter, ...]) -> float:
    """Match a loss's last-layer gradient norm to the receiver reference."""
    def norm(value: torch.Tensor) -> torch.Tensor:
        gradients = torch.autograd.grad(value, parameters, retain_graph=True,
                                        allow_unused=True)
        return torch.sqrt(sum((g.detach().square().sum() for g in gradients
                               if g is not None), value.new_zeros(())))

    reference_norm = norm(reference)
    loss_norm = norm(loss)
    if not torch.isfinite(reference_norm) or not torch.isfinite(loss_norm):
        raise FloatingPointError("Nonfinite training loss gradient")
    if loss_norm.item() == 0:
        return 0.0
    return float((reference_norm / loss_norm).clamp(1e-12, 1e12))


def training_step(model: SourceConditionedWavePINN,
                  optimizer: torch.optim.Optimizer,
                  velocity_lookup,
                  frequency: float,
                  n_phys: int,
                  n_ic: int,
                  n_bc_per_face: int,
                  device: torch.device,
                  cfg: SourceConditionedConfig,
                  receiver_batch=None,
                  receiver_loss_weight: float = 0.0,
                  balance_loss_gradients: bool = False,
                  receiver_only: bool = False):
    """Run a source-conditioned update and return scalar diagnostics."""
    model.train()
    optimizer.zero_grad()

    if receiver_only:
        if receiver_batch is None or receiver_loss_weight <= 0:
            raise ValueError("Receiver-only training requires receiver observations")
        space_time_r, target_r, source_r, rms = receiver_batch
        source_values = (source_r.expand(len(target_r), 4)
                         if source_r.ndim == 1 else source_r)
        prediction = model(space_time_r, source_values)
        receiver_loss = ((prediction.squeeze(-1) / rms - target_r) ** 2).mean()
        total = receiver_loss_weight * receiver_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return {"total": float(total.detach()),
                "receivers": float(receiver_loss.detach())}

    space_time = torch.rand(n_phys, 4, device=device)
    source_phys = sample_source_condition(n_phys, device, cfg)
    n_near = int(round(n_phys * cfg.source_near_fraction))
    if n_near:
        length_scale = torch.tensor(
            [cfg.x_length_m, cfg.y_length_m, cfg.z_length_m],
            device=device,
        )
        spatial_jitter = (
            torch.randn(n_near, 3, device=device)
            * (cfg.source_spatial_jitter_sigma * cfg.source_sigma_m / length_scale)
        )
        space_time[:n_near, :3] = (
            source_phys[:n_near, :3] + spatial_jitter
        ).clamp_(0.0, 1.0)
        temporal_half_width = (
            cfg.source_temporal_half_width_periods / (frequency * cfg.time_length_s)
        )
        temporal_jitter = (
            2.0 * torch.rand(n_near, device=device) - 1.0
        ) * temporal_half_width
        space_time[:n_near, 3] = (
            source_phys[:n_near, 3] + temporal_jitter
        ).clamp_(0.0, 1.0)
    n_exact = int(round(n_phys * cfg.source_exact_fraction))
    if n_exact:
        if n_exact > n_near:
            raise ValueError("Exact-source samples must be a subset of near-source samples")
        space_time[:n_exact, :3] = source_phys[:n_exact, :3]
        # Include the Ricker peak explicitly while retaining off-peak samples.
        n_peak = max(1, n_exact // 4)
        space_time[:n_peak, 3] = source_phys[:n_peak, 3]
    velocity = velocity_lookup(space_time[:, :3])
    loss_phys = physics_loss(
        model, space_time, source_phys, velocity, frequency, cfg
    )

    xyz_ic = torch.rand(n_ic, 3, device=device)
    source_ic = sample_source_condition(n_ic, device, cfg)
    loss_ic = initial_condition_loss(model, xyz_ic, source_ic, frequency, cfg)
    loss_bc = boundary_condition_loss(
        model, n_bc_per_face, device, cfg
    )
    components = {"physics": cfg.lambda_phys * loss_phys,
                  "initial": cfg.lambda_ic * loss_ic,
                  "boundary": cfg.lambda_bc * loss_bc}
    if receiver_batch is not None:
        space_time_r, target_r, source_r, rms = receiver_batch
        source_values = (source_r.expand(len(target_r), 4)
                         if source_r.ndim == 1 else source_r)
        predicted_r = model(space_time_r, source_values)
        loss_receivers = ((predicted_r.squeeze(-1) / rms - target_r) ** 2).mean()
        components["receivers"] = receiver_loss_weight * loss_receivers
    weights = {name: 1.0 for name in components}
    if balance_loss_gradients:
        if "receivers" not in components:
            raise ValueError("Gradient balancing requires receiver observations")
        last_layer = tuple(model.net[-1].parameters())
        for name in ("physics", "initial", "boundary"):
            weights[name] = gradient_balance_weight(
                components[name], components["receivers"], last_layer)
    total = sum(weights[name] * value for name, value in components.items())
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    metrics = {
        "physics": float(loss_phys.detach()),
        "initial": float(loss_ic.detach()),
        "boundary": float(loss_bc.detach()),
        "total": float(total.detach()),
    }
    if receiver_batch is not None:
        metrics["receivers"] = float(loss_receivers.detach())
    if balance_loss_gradients:
        metrics["gradient_balance_weights"] = weights
    return metrics


def save_conditioned_checkpoint(path, model: SourceConditionedWavePINN,
                                optimizer: torch.optim.Optimizer | None,
                                frequency: float, epoch: int,
                                cfg: SourceConditionedConfig,
                                history: list[dict] | None = None,
                                training_record: dict | None = None) -> None:
    if isinstance(model, HarmonicCarrierWavePINN) and model.frequency_hz != float(frequency):
        raise ValueError("Harmonic carrier frequency must match checkpoint stage")
    kind = ("velocity_conditioned_travel_time_carrier"
            if isinstance(model, VelocityConditionedTravelTimeCarrierWavePINN)
            else "eikonal_travel_time_carrier"
            if isinstance(model, EikonalTravelTimeCarrierWavePINN)
            else "translation_invariant_travel_time_carrier"
            if isinstance(model, TranslationInvariantTravelTimeCarrierWavePINN)
            else "travel_time_carrier" if isinstance(model, TravelTimeCarrierWavePINN)
            else "harmonic_carrier" if isinstance(model, HarmonicCarrierWavePINN)
            else "standard")
    payload = {
        "model_type": "source_conditioned_wave_pinn",
        "input_dim": 8,
        "architecture": {
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "output_scale": model.output_scale,
            **({"version": 3,
                "distance_core_m": model.distance_core_m} if isinstance(
                model, TranslationInvariantTravelTimeCarrierWavePINN) else {}),
            "hard_initial": model.hard_initial if isinstance(model, HarmonicCarrierWavePINN)
            else False,
            "kind": kind,
            **({"reference_velocity_m_s": model.reference_velocity_m_s}
               if isinstance(model, TravelTimeCarrierWavePINN) else {}),
            **({"distance_core_m": model.distance_core_m}
               if isinstance(model, TravelTimeCarrierWavePINN) else {}),
            **({"ray_travel_time": model.ray_travel_time,
                "ray_samples": model.ray_samples,
                "onset_gate": model.onset_gate}
               if isinstance(model, TravelTimeCarrierWavePINN) else {}),
            **({"eikonal_architecture": {
                "hidden_dim": model.eikonal_model.hidden_dim,
                "num_layers": model.eikonal_model.num_layers,
                "velocity_min": model.eikonal_model.velocity_min,
                "velocity_max": model.eikonal_model.velocity_max,
                "domain_m": model.eikonal_model.domain_m,
                "core_m": model.eikonal_model.core_m,
                "distance_formula": model.eikonal_model.distance_formula,
            }} if isinstance(model, EikonalTravelTimeCarrierWavePINN) else {}),
        },
        "frequency_hz": float(frequency),
        "epoch": int(epoch),
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "history": history or [],
        "training_record": training_record or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_conditioned_checkpoint(path, device: torch.device):
    # Checkpoints contain metadata and optimizer state in addition to tensors.
    # Load only checkpoints created by this project or another trusted source.
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_type") != "source_conditioned_wave_pinn":
        raise ValueError(
            "Checkpoint is not source-conditioned; four-input legacy "
            "checkpoints are incompatible."
        )
    if payload.get("input_dim") != 8:
        raise ValueError("Source-conditioned checkpoint must have input_dim=8")
    architecture = payload.get("architecture", {})
    kind = architecture.get("kind", "standard")
    if (kind == "translation_invariant_travel_time_carrier"
            and architecture.get("version") not in (2, 3)):
        raise ValueError("Translation-invariant checkpoint predates the "
                         "smooth source-distance correction; retrain before loading")
    if kind not in ("standard", "harmonic_carrier", "travel_time_carrier",
                    "eikonal_travel_time_carrier",
                    "velocity_conditioned_travel_time_carrier",
                    "translation_invariant_travel_time_carrier"):
        raise ValueError(f"Unknown source-conditioned architecture: {kind}")
    model_class = {"standard": SourceConditionedWavePINN,
                   "harmonic_carrier": HarmonicCarrierWavePINN,
                   "travel_time_carrier": TravelTimeCarrierWavePINN,
                   "eikonal_travel_time_carrier": EikonalTravelTimeCarrierWavePINN,
                   "velocity_conditioned_travel_time_carrier":
                       VelocityConditionedTravelTimeCarrierWavePINN,
                   "translation_invariant_travel_time_carrier":
                       TranslationInvariantTravelTimeCarrierWavePINN}[kind]
    kwargs = {}
    if kind != "standard":
        kwargs = {"frequency_hz": float(payload["frequency_hz"]),
                  "time_length_s": float(payload["config"]["time_length_s"]),
                  "hard_initial": bool(architecture.get("hard_initial", False))}
    if kind in ("travel_time_carrier", "eikonal_travel_time_carrier",
                "velocity_conditioned_travel_time_carrier",
                "translation_invariant_travel_time_carrier"):
        default_core = (1.0 if kind == "translation_invariant_travel_time_carrier"
                        else 0.0)
        kwargs.update(reference_velocity_m_s=float(architecture["reference_velocity_m_s"]),
                      x_length_m=float(payload["config"]["x_length_m"]),
                      y_length_m=float(payload["config"]["y_length_m"]),
                      z_length_m=float(payload["config"]["z_length_m"]),
                      distance_core_m=float(architecture.get(
                          "distance_core_m", default_core)),
                      ray_travel_time=bool(architecture.get("ray_travel_time", False)),
                      ray_samples=int(architecture.get("ray_samples", 16)),
                      onset_gate=architecture.get("onset_gate"))
    if kind in ("eikonal_travel_time_carrier",
                "velocity_conditioned_travel_time_carrier"):
        from eikonal_traveltime import FactorizedEikonalNet
        kwargs["eikonal_model"] = FactorizedEikonalNet(
            **architecture["eikonal_architecture"])
    model = model_class(
        hidden_dim=int(architecture.get("hidden_dim", 128)),
        num_layers=int(architecture.get("num_layers", 5)),
        output_scale=float(architecture.get("output_scale", 1.0)),
        **kwargs,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def conditioned_prediction(model: SourceConditionedWavePINN,
                           receiver_xyz: torch.Tensor,
                           times: torch.Tensor,
                           source: torch.Tensor) -> torch.Tensor:
    """Evaluate all receiver-time samples for a differentiable inversion."""
    n_receiver = receiver_xyz.shape[0]
    n_time = times.shape[0]
    xyz = receiver_xyz[:, None, :].expand(n_receiver, n_time, 3)
    t = times[None, :, None].expand(n_receiver, n_time, 1)
    st = torch.cat((xyz, t), dim=-1).reshape(-1, 4)
    src = source.reshape(1, 4).expand(st.shape[0], 4)
    return model(st, src).reshape(n_receiver, n_time)


def invert_source(model: SourceConditionedWavePINN,
                  receiver_xyz: torch.Tensor, times: torch.Tensor,
                  observed: torch.Tensor, initial_source: torch.Tensor,
                  iterations: int = 300, learning_rate: float = 5e-3):
    """Optimize all four source parameters through the conditioned PINN."""
    model.eval()
    source = nn.Parameter(initial_source.detach().clone())
    optimizer = torch.optim.Adam((source,), lr=learning_rate)
    history = []
    for _ in range(iterations):
        optimizer.zero_grad()
        prediction = conditioned_prediction(model, receiver_xyz, times, source)
        loss = (prediction - observed).square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            source.clamp_(0.0, 1.0)
        history.append(float(loss.detach()))
    return source.detach(), history
