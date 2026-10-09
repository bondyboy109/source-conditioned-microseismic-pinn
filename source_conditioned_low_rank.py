"""Factorized source-conditioned receiver waveform operator.

The operator separates propagation from waveform shape:

    u(r, t; s) = sum_k c_k(s, r) b_k(t - t_s - tau_FMM(s, r)).

FMM travel time therefore controls the retarded-time coordinate explicitly,
while the geometry network predicts source-receiver response coefficients.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from traveltime_aligned_moe import ReceiverTravelTimeInterpolator


def _mlp(input_dim: int, hidden_dim: int, hidden_layers: int,
         output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class SourceConditionedLowRankOperator(nn.Module):
    """Differentiable low-rank source-to-receiver waveform surrogate."""

    def __init__(self, interpolator: ReceiverTravelTimeInterpolator,
                 receiver_xyz_m: torch.Tensor,
                 spatial_lengths_m: tuple[float, float, float],
                 time_length_s: float = 2.0, frequency_hz: float = 5.0,
                 rank: int = 8, hidden_dim: int = 128,
                 hidden_layers: int = 3, causal_width_s: float = 0.04):
        super().__init__()
        receivers = torch.as_tensor(receiver_xyz_m, dtype=torch.float32)
        if receivers.shape != (interpolator.receiver_count, 3):
            raise ValueError("receiver_xyz_m must match travel-time fields")
        if len(spatial_lengths_m) != 3 or any(v <= 0 for v in spatial_lengths_m):
            raise ValueError("spatial_lengths_m must contain three positives")
        if time_length_s <= 0 or frequency_hz <= 0 or rank < 1:
            raise ValueError("Invalid time, frequency, or rank")
        if causal_width_s <= 0:
            raise ValueError("causal_width_s must be positive")
        self.interpolator = interpolator
        self.register_buffer("receiver_xyz_m", receivers)
        self.register_buffer("spatial_lengths_m",
                             torch.tensor(spatial_lengths_m, dtype=torch.float32))
        self.time_length_s = float(time_length_s)
        self.frequency_hz = float(frequency_hz)
        self.rank = int(rank)
        self.causal_width_s = float(causal_width_s)
        # Relative geometry, source position, receiver position, FMM time,
        # and a bounded distance coordinate.
        self.geometry_net = _mlp(11, hidden_dim, hidden_layers, rank)
        # The basis is global in time and shared by all source-receiver pairs.
        self.basis_net = _mlp(3, hidden_dim, hidden_layers, rank)

    def forward(self, space_time_normalized: torch.Tensor,
                source_normalized: torch.Tensor,
                receiver_index: torch.Tensor):
        if (space_time_normalized.ndim != 2
                or space_time_normalized.shape[1] != 4
                or source_normalized.shape != space_time_normalized.shape):
            raise ValueError("space_time and source must both have shape [N, 4]")
        receiver_index = receiver_index.to(space_time_normalized.device,
                                           dtype=torch.long)
        lengths = self.spatial_lengths_m.to(space_time_normalized)
        source_xyz_m = source_normalized[:, :3] * lengths
        travel_s = self.interpolator(source_xyz_m, receiver_index)
        receiver_xyz = self.receiver_xyz_m.to(space_time_normalized)[receiver_index]
        receiver_normalized = receiver_xyz / lengths
        relative_xyz = receiver_normalized - source_normalized[:, :3]
        distance_normalized = torch.linalg.vector_norm(
            relative_xyz * lengths, dim=1, keepdim=True) / torch.linalg.vector_norm(lengths)
        retarded_s = (space_time_normalized[:, 3:4] * self.time_length_s
                      - source_normalized[:, 3:4] * self.time_length_s
                      - travel_s)
        phase = 2.0 * math.pi * self.frequency_hz * retarded_s
        geometry = torch.cat((relative_xyz, source_normalized[:, :3],
                              receiver_normalized, travel_s / self.time_length_s,
                              torch.log1p(distance_normalized)), dim=1)
        coefficients = self.geometry_net(geometry) / math.sqrt(self.rank)
        temporal = torch.cat((retarded_s / self.time_length_s,
                              torch.sin(phase), torch.cos(phase)), dim=1)
        basis = self.basis_net(temporal)
        causal_gate = torch.sigmoid(retarded_s / self.causal_width_s)
        prediction = (coefficients * basis).sum(dim=1, keepdim=True) * causal_gate
        return prediction, coefficients, travel_s

    def coefficient_regularization(self) -> torch.Tensor:
        return self.geometry_net[-1].weight.square().mean()
