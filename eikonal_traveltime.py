"""Differentiable source-conditioned first-arrival traveltime model."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


class FactorizedEikonalNet(nn.Module):
    """T(x; xs) = smooth_distance(x,xs) * learned bounded slowness."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 5,
                 velocity_min: float = 2000.0, velocity_max: float = 6500.0,
                 domain_m=(16000.0, 16000.0, 3720.0), core_m: float = 20.0,
                 distance_formula: str = "quadratic_core"):
        super().__init__()
        if velocity_min <= 0 or velocity_max <= velocity_min:
            raise ValueError("Velocity bounds must be positive and ordered")
        if core_m <= 0 or not math.isfinite(core_m):
            raise ValueError("core_m must be finite and positive")
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.velocity_min = float(velocity_min)
        self.velocity_max = float(velocity_max)
        self.domain_m = tuple(float(value) for value in domain_m)
        self.core_m = float(core_m)
        if distance_formula not in ("quadratic_core", "sqrt_minus_core"):
            raise ValueError("Unknown distance formula")
        self.distance_formula = distance_formula
        layers = [nn.Linear(6, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, source_xyz: torch.Tensor) -> torch.Tensor:
        if xyz.shape[-1] != 3 or source_xyz.shape[-1] != 3:
            raise ValueError("xyz and source_xyz must have three columns")
        relative = xyz - source_xyz
        lengths = xyz.new_tensor(self.domain_m)
        squared_distance = (relative * lengths).square().sum(dim=-1, keepdim=True)
        denominator = torch.sqrt(squared_distance + self.core_m ** 2)
        if self.distance_formula == "quadratic_core":
            distance = squared_distance / denominator
        else:
            distance = denominator - self.core_m
        raw = self.net(torch.cat((xyz, source_xyz), dim=-1))
        slow_min = 1.0 / self.velocity_max
        slow_max = 1.0 / self.velocity_min
        slowness = slow_min + (slow_max - slow_min) * torch.sigmoid(raw)
        return distance * slowness


def eikonal_loss(model: FactorizedEikonalNet, xyz: torch.Tensor,
                  source_xyz: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    xyz = xyz.clone().detach().requires_grad_(True)
    travel = model(xyz, source_xyz)
    gradient = torch.autograd.grad(travel.sum(), xyz, create_graph=True)[0]
    lengths = xyz.new_tensor(model.domain_m)
    physical_gradient = gradient / lengths
    predicted_slowness = torch.linalg.vector_norm(
        physical_gradient, dim=-1, keepdim=True)
    target_slowness = 1.0 / velocity
    relative_residual = (predicted_slowness - target_slowness) / target_slowness
    return relative_residual.square().mean()


def save_eikonal_checkpoint(path: Path, model: FactorizedEikonalNet,
                            optimizer, step: int, history: list[float],
                            metadata: dict) -> None:
    payload = {
        "model_type": "source_conditioned_eikonal",
        "architecture": {
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "velocity_min": model.velocity_min,
            "velocity_max": model.velocity_max,
            "domain_m": model.domain_m,
            "core_m": model.core_m,
            "distance_formula": model.distance_formula,
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "history": history,
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_eikonal_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_type") != "source_conditioned_eikonal":
        raise ValueError("Not a source-conditioned eikonal checkpoint")
    architecture = dict(payload["architecture"])
    architecture.setdefault("distance_formula", "sqrt_minus_core")
    model = FactorizedEikonalNet(**architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
