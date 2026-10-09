"""Differentiable gated source-region mixture for a frozen/shared wave surrogate."""

from __future__ import annotations

import torch
from torch import nn


class GatedResidualMixtureOfExperts(nn.Module):
    """Shared base wavefield plus source-gated residual experts.

    The gate receives the normalized source parameters and produces a softmax
    mixture. Every expert receives receiver/time coordinates, source
    parameters, and the shared base prediction. The complete map remains
    differentiable with respect to source location and origin time.
    """

    def __init__(self, base_model: nn.Module, expert_count: int = 3,
                 hidden_dim: int = 128, hidden_layers: int = 3,
                 feature_scale: float = 1e6):
        super().__init__()
        if expert_count < 2:
            raise ValueError("expert_count must be at least two")
        if feature_scale <= 0:
            raise ValueError("feature_scale must be positive")
        self.base_model = base_model
        self.expert_count = int(expert_count)
        self.feature_scale = float(feature_scale)
        gate_layers: list[nn.Module] = [nn.Linear(4, hidden_dim), nn.SiLU()]
        gate_layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        gate_output = nn.Linear(hidden_dim, self.expert_count)
        nn.init.zeros_(gate_output.weight)
        nn.init.zeros_(gate_output.bias)
        gate_layers.append(gate_output)
        self.gate = nn.Sequential(*gate_layers)
        self.experts = nn.ModuleList()
        for _ in range(self.expert_count):
            layers: list[nn.Module] = [nn.Linear(9, hidden_dim), nn.SiLU()]
            for _ in range(hidden_layers - 1):
                layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
            output = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            layers.append(output)
            self.experts.append(nn.Sequential(*layers))

    def forward(self, space_time: torch.Tensor, source: torch.Tensor,
                freeze_base: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if space_time.shape[-1] != 4 or source.shape[-1] != 4:
            raise ValueError("space_time and source must have four columns")
        base = self.base_model(space_time, source)
        if freeze_base:
            base_features = base.detach()
        else:
            base_features = base
        expert_input = torch.cat((space_time, source,
                                  base_features * self.feature_scale), dim=-1)
        residuals = torch.cat([expert(expert_input) for expert in self.experts], dim=-1)
        gate = torch.softmax(self.gate(source), dim=-1)
        correction = torch.sum(gate * residuals, dim=-1, keepdim=True)
        return base + correction / self.feature_scale, gate


class PrototypeGatedResidualMixtureOfExperts(GatedResidualMixtureOfExperts):
    """Residual MoE with a differentiable source-prototype gate.

    The prototypes are learned jointly with the residual experts. A positive
    source-scale keeps the four source coordinates comparable, while the
    softmax over negative squared distances provides explicit regional
    specialization without requiring hard source labels.
    """

    def __init__(self, base_model: nn.Module, expert_count: int = 3,
                 hidden_dim: int = 128, hidden_layers: int = 3,
                 feature_scale: float = 1e6,
                 prototype_centers: torch.Tensor | None = None,
                 prototype_scales: tuple[float, float, float, float] =
                 (0.25, 0.25, 0.15, 0.05),
                 temperature: float = 1.0):
        super().__init__(base_model, expert_count=expert_count,
                         hidden_dim=hidden_dim, hidden_layers=hidden_layers,
                         feature_scale=feature_scale)
        if len(prototype_scales) != 4 or any(float(value) <= 0 for value in prototype_scales):
            raise ValueError("prototype_scales must contain four positive values")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if prototype_centers is None:
            default = torch.tensor([
                [0.25, 0.25, 0.16, 0.015],
                [0.50, 0.50, 0.25, 0.030],
                [0.75, 0.70, 0.32, 0.045],
            ], dtype=torch.float32)
            if expert_count != len(default):
                prototype_centers = torch.rand(expert_count, 4)
            else:
                prototype_centers = default
        if prototype_centers.shape != (expert_count, 4):
            raise ValueError("prototype_centers must have shape (expert_count, 4)")
        self.prototype_centers = nn.Parameter(prototype_centers.clone().detach())
        self.register_buffer("prototype_scales", torch.tensor(prototype_scales, dtype=torch.float32))
        self.temperature = float(temperature)

    def source_gate(self, source: torch.Tensor) -> torch.Tensor:
        scaled = (source[:, None, :] - self.prototype_centers[None, :, :]) / self.prototype_scales
        logits = -scaled.square().sum(dim=-1) / self.temperature
        return torch.softmax(logits, dim=-1)

    def forward(self, space_time: torch.Tensor, source: torch.Tensor,
                freeze_base: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if space_time.shape[-1] != 4 or source.shape[-1] != 4:
            raise ValueError("space_time and source must have four columns")
        base = self.base_model(space_time, source)
        base_features = base.detach() if freeze_base else base
        expert_input = torch.cat((space_time, source,
                                  base_features * self.feature_scale), dim=-1)
        residuals = torch.cat([expert(expert_input) for expert in self.experts], dim=-1)
        gate = self.source_gate(source)
        correction = torch.sum(gate * residuals, dim=-1, keepdim=True)
        return base + correction / self.feature_scale, gate
