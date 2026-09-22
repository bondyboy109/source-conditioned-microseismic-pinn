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

    def __init__(self, hidden_dim: int = 128, num_layers: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        return self.net(torch.cat((space_time, source), dim=-1))


@dataclass(frozen=True)
class SourceConditionedConfig:
    source_sigma_m: float = 50.0
    source_near_fraction: float = 0.25
    lambda_phys: float = 1.0
    lambda_ic: float = 10.0
    lambda_bc: float = 10.0
    source_time_min: float = 0.05
    source_time_max: float = 0.50
    source_x_min: float = 0.10
    source_x_max: float = 0.70
    source_y_min: float = 0.10
    source_y_max: float = 0.70
    source_z_min: float = 100.0 / 4650.0
    source_z_max: float = 1500.0 / 4650.0
    x_length_m: float = 20000.0
    y_length_m: float = 20000.0
    z_length_m: float = 4650.0
    time_length_s: float = 1.0


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
                           xyz: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Enforce u=0 and du/dt=0 at normalized time zero."""
    time = torch.zeros(xyz.shape[0], 1, device=xyz.device)
    space_time = torch.cat((xyz, time), dim=-1).requires_grad_(True)
    u = model(space_time, source)
    u_t = torch.autograd.grad(
        u, space_time, torch.ones_like(u), create_graph=True
    )[0][:, 3:4]
    return u.square().mean() + u_t.square().mean()


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
            losses.append(model(space_time, source).square().mean())
    return torch.stack(losses).mean()


def training_step(model: SourceConditionedWavePINN,
                  optimizer: torch.optim.Optimizer,
                  velocity_lookup,
                  frequency: float,
                  n_phys: int,
                  n_ic: int,
                  n_bc_per_face: int,
                  device: torch.device,
                  cfg: SourceConditionedConfig):
    """Run one source-conditioned PINN update and return scalar diagnostics."""
    model.train()
    optimizer.zero_grad()

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
            * (2.0 * cfg.source_sigma_m / length_scale)
        )
        space_time[:n_near, :3] = (
            source_phys[:n_near, :3] + spatial_jitter
        ).clamp_(0.0, 1.0)
        temporal_half_width = 2.0 / (frequency * cfg.time_length_s)
        temporal_jitter = (
            2.0 * torch.rand(n_near, device=device) - 1.0
        ) * temporal_half_width
        space_time[:n_near, 3] = (
            source_phys[:n_near, 3] + temporal_jitter
        ).clamp_(0.0, 1.0)
    velocity = velocity_lookup(space_time[:, :3])
    loss_phys = physics_loss(
        model, space_time, source_phys, velocity, frequency, cfg
    )

    xyz_ic = torch.rand(n_ic, 3, device=device)
    source_ic = sample_source_condition(n_ic, device, cfg)
    loss_ic = initial_condition_loss(model, xyz_ic, source_ic)
    loss_bc = boundary_condition_loss(
        model, n_bc_per_face, device, cfg
    )
    total = (
        cfg.lambda_phys * loss_phys
        + cfg.lambda_ic * loss_ic
        + cfg.lambda_bc * loss_bc
    )
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {
        "physics": float(loss_phys.detach()),
        "initial": float(loss_ic.detach()),
        "boundary": float(loss_bc.detach()),
        "total": float(total.detach()),
    }


def save_conditioned_checkpoint(path, model: SourceConditionedWavePINN,
                                optimizer: torch.optim.Optimizer | None,
                                frequency: float, epoch: int,
                                cfg: SourceConditionedConfig,
                                history: list[dict] | None = None) -> None:
    payload = {
        "model_type": "source_conditioned_wave_pinn",
        "input_dim": 8,
        "architecture": {
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
        },
        "frequency_hz": float(frequency),
        "epoch": int(epoch),
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "history": history or [],
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
    model = SourceConditionedWavePINN(
        hidden_dim=int(architecture.get("hidden_dim", 128)),
        num_layers=int(architecture.get("num_layers", 5)),
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
