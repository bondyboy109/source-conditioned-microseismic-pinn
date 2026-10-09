"""Differentiable FMM-aligned receiver-waveform mixture of experts."""

from __future__ import annotations

import math

import torch
from torch import nn


class ReceiverTravelTimeInterpolator(nn.Module):
    """Trilinearly sample reciprocal receiver travel-time fields.

    Fields use the axis order ``[receiver, x, y, z]``. Cell selection is
    piecewise constant, while the interpolation weights retain gradients with
    respect to the source coordinates inside each cell.
    """

    def __init__(self, fields_s: torch.Tensor, spacing_m: float):
        super().__init__()
        fields = torch.as_tensor(fields_s, dtype=torch.float32)
        if fields.ndim != 4 or min(fields.shape[1:]) < 2:
            raise ValueError("fields_s must have shape [receiver, x, y, z]")
        if not torch.isfinite(fields).all() or (fields < 0).any():
            raise ValueError("Travel-time fields must be finite and nonnegative")
        if not math.isfinite(spacing_m) or spacing_m <= 0:
            raise ValueError("spacing_m must be finite and positive")
        self.register_buffer("fields_s", fields, persistent=False)
        self.spacing_m = float(spacing_m)

    @property
    def receiver_count(self) -> int:
        return int(self.fields_s.shape[0])

    def forward(self, source_xyz_m: torch.Tensor,
                receiver_index: torch.Tensor) -> torch.Tensor:
        if source_xyz_m.ndim != 2 or source_xyz_m.shape[1] != 3:
            raise ValueError("source_xyz_m must have shape [N, 3]")
        if receiver_index.ndim != 1 or len(receiver_index) != len(source_xyz_m):
            raise ValueError("receiver_index must have shape [N]")
        receiver_index = receiver_index.to(device=source_xyz_m.device,
                                           dtype=torch.long)
        if ((receiver_index < 0).any()
                or (receiver_index >= self.receiver_count).any()):
            raise ValueError("receiver_index is outside the travel-time fields")
        position = source_xyz_m / self.spacing_m
        upper_index = source_xyz_m.new_tensor(self.fields_s.shape[1:]) - 1
        if (position < 0).any() or (position > upper_index).any():
            raise ValueError("source coordinates are outside the travel-time grid")
        lower = torch.minimum(torch.floor(position).long(),
                              upper_index.long() - 1)
        fraction = position - lower.to(position.dtype)
        values = torch.zeros(len(source_xyz_m), device=source_xyz_m.device,
                             dtype=source_xyz_m.dtype)
        fields = self.fields_s.to(device=source_xyz_m.device,
                                  dtype=source_xyz_m.dtype)
        for ox in (0, 1):
            for oy in (0, 1):
                for oz in (0, 1):
                    offsets = source_xyz_m.new_tensor((ox, oy, oz),
                                                      dtype=torch.long)
                    index = lower + offsets
                    selector = source_xyz_m.new_tensor((ox, oy, oz))
                    weight = torch.where(selector.bool(), fraction,
                                         1.0 - fraction).prod(dim=1)
                    values = values + weight * fields[
                        receiver_index, index[:, 0], index[:, 1], index[:, 2]]
        return values.unsqueeze(1)


class TravelTimeAlignedPrototypeMoE(nn.Module):
    """Source-region MoE evaluated in heterogeneous retarded time."""

    def __init__(self, interpolator: ReceiverTravelTimeInterpolator,
                 receiver_xyz_m: torch.Tensor,
                 spatial_lengths_m: tuple[float, float, float],
                 time_length_s: float = 2.0, frequency_hz: float = 5.0,
                 expert_count: int = 3, hidden_dim: int = 128,
                 hidden_layers: int = 4,
                 prototype_centers: torch.Tensor | None = None,
                 prototype_scales: tuple[float, float, float] =
                 (0.30, 0.30, 0.20),
                 temperature: float = 1.0,
                 geometric_attenuation: bool = False):
        super().__init__()
        receivers = torch.as_tensor(receiver_xyz_m, dtype=torch.float32)
        if receivers.shape != (interpolator.receiver_count, 3):
            raise ValueError("receiver_xyz_m must match the receiver fields")
        if (len(spatial_lengths_m) != 3
                or any(not math.isfinite(v) or v <= 0
                       for v in spatial_lengths_m)):
            raise ValueError("spatial_lengths_m must contain three positive values")
        if not math.isfinite(time_length_s) or time_length_s <= 0:
            raise ValueError("time_length_s must be finite and positive")
        if not math.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        if expert_count < 2 or hidden_layers < 1 or hidden_dim < 1:
            raise ValueError("Invalid expert architecture")
        if (len(prototype_scales) != 3
                or any(not math.isfinite(v) or v <= 0
                       for v in prototype_scales)):
            raise ValueError("prototype_scales must contain three positive values")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.interpolator = interpolator
        self.register_buffer("receiver_xyz_m", receivers)
        self.register_buffer("spatial_lengths_m",
                             torch.tensor(spatial_lengths_m, dtype=torch.float32))
        self.register_buffer("prototype_scales",
                             torch.tensor(prototype_scales, dtype=torch.float32))
        self.time_length_s = float(time_length_s)
        self.frequency_hz = float(frequency_hz)
        self.expert_count = int(expert_count)
        self.temperature = float(temperature)
        self.geometric_attenuation = bool(geometric_attenuation)
        if prototype_centers is None:
            prototype_centers = torch.rand(expert_count, 3)
        centers = torch.as_tensor(prototype_centers, dtype=torch.float32)
        if centers.shape != (expert_count, 3):
            raise ValueError("prototype_centers must have shape [expert_count, 3]")
        self.prototype_centers = nn.Parameter(centers.clone())
        self.experts = nn.ModuleList()
        for _ in range(expert_count):
            layers: list[nn.Module] = []
            width_in = 12 if self.geometric_attenuation else 10
            for _ in range(hidden_layers):
                layers.extend((nn.Linear(width_in, hidden_dim), nn.SiLU()))
                width_in = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1))
            self.experts.append(nn.Sequential(*layers))

    def source_gate(self, source_xyz_normalized: torch.Tensor) -> torch.Tensor:
        difference = ((source_xyz_normalized[:, None, :]
                       - self.prototype_centers[None, :, :])
                      / self.prototype_scales)
        logits = -difference.square().sum(dim=-1) / self.temperature
        return torch.softmax(logits, dim=-1)

    def forward(self, space_time_normalized: torch.Tensor,
                source_normalized: torch.Tensor,
                receiver_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (space_time_normalized.ndim != 2
                or space_time_normalized.shape[1] != 4
                or source_normalized.shape != space_time_normalized.shape):
            raise ValueError("space_time and source must both have shape [N, 4]")
        receiver_index = receiver_index.to(device=space_time_normalized.device,
                                           dtype=torch.long)
        lengths = self.spatial_lengths_m.to(space_time_normalized)
        source_xyz_m = source_normalized[:, :3] * lengths
        travel_s = self.interpolator(source_xyz_m, receiver_index)
        receiver_xyz = self.receiver_xyz_m.to(space_time_normalized)[receiver_index]
        receiver_normalized = receiver_xyz / lengths
        relative_xyz = receiver_normalized - source_normalized[:, :3]
        if self.geometric_attenuation:
            relative_xyz_m = relative_xyz * lengths
            distance_m = torch.linalg.vector_norm(relative_xyz_m, dim=1, keepdim=True)
            characteristic_length = torch.linalg.vector_norm(lengths)
            distance_normalized = distance_m / characteristic_length
            inverse_distance = 1.0 / (1.0 + distance_normalized)
        retarded_s = (space_time_normalized[:, 3:4] * self.time_length_s
                      - source_normalized[:, 3:4] * self.time_length_s
                      - travel_s)
        phase = 2.0 * math.pi * self.frequency_hz * retarded_s
        feature_parts = [
            relative_xyz,
            source_normalized[:, :3],
            travel_s / self.time_length_s,
            retarded_s / self.time_length_s,
            torch.sin(phase),
            torch.cos(phase),
        ]
        if self.geometric_attenuation:
            feature_parts.extend((distance_normalized, inverse_distance))
        features = torch.cat(feature_parts, dim=1)
        expert_values = torch.cat([expert(features) for expert in self.experts],
                                  dim=1)
        gate = self.source_gate(source_normalized[:, :3])
        prediction = torch.sum(gate * expert_values, dim=1, keepdim=True)
        return prediction, gate, travel_s

    def prototype_separation_loss(self) -> torch.Tensor:
        scaled = self.prototype_centers / self.prototype_scales
        distances = torch.cdist(scaled, scaled).square()
        mask = ~torch.eye(self.expert_count, device=distances.device,
                          dtype=torch.bool)
        return torch.exp(-distances[mask]).mean()
