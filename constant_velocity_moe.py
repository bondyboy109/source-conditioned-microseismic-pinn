"""Source-conditioned MoE using a constant-velocity receiver time shift."""
from __future__ import annotations

import math
import torch
from torch import nn


class ConstantVelocityPrototypeMoE(nn.Module):
    def __init__(self, receiver_xyz_m, spatial_lengths_m, vavg,
                 time_length_s=2.0, frequency_hz=5.0, expert_count=3,
                 hidden_dim=128, hidden_layers=4,
                 prototype_centers=None, prototype_scales=(.30,.30,.20),
                 temperature=1.0):
        super().__init__()
        receivers = torch.as_tensor(receiver_xyz_m, dtype=torch.float32)
        self.register_buffer('receiver_xyz_m', receivers)
        self.register_buffer('spatial_lengths_m', torch.tensor(spatial_lengths_m, dtype=torch.float32))
        self.vavg = float(vavg)
        self.time_length_s = float(time_length_s)
        self.frequency_hz = float(frequency_hz)
        self.expert_count = int(expert_count)
        self.temperature = float(temperature)
        self.register_buffer('prototype_scales', torch.tensor(prototype_scales, dtype=torch.float32))
        centers = torch.as_tensor(prototype_centers, dtype=torch.float32)
        self.prototype_centers = nn.Parameter(centers.clone())
        self.experts = nn.ModuleList()
        for _ in range(self.expert_count):
            layers = []
            width = 10
            for _ in range(hidden_layers):
                layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
                width = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1))
            self.experts.append(nn.Sequential(*layers))

    def source_gate(self, xyz):
        diff = (xyz[:, None, :] - self.prototype_centers[None, :, :]) / self.prototype_scales
        return torch.softmax(-diff.square().sum(-1) / self.temperature, dim=-1)

    def forward(self, space_time_normalized, source_normalized, receiver_index):
        lengths = self.spatial_lengths_m.to(space_time_normalized)
        receiver_index = receiver_index.long()
        receiver_xyz = self.receiver_xyz_m.to(space_time_normalized)[receiver_index]
        source_xyz = source_normalized[:, :3] * lengths
        distance = torch.linalg.vector_norm(receiver_xyz - source_xyz, dim=1, keepdim=True)
        travel_s = distance / self.vavg
        receiver_normalized = receiver_xyz / lengths
        relative_xyz = receiver_normalized - source_normalized[:, :3]
        retarded_s = (space_time_normalized[:, 3:4] * self.time_length_s
                      - source_normalized[:, 3:4] * self.time_length_s - travel_s)
        phase = 2.0 * math.pi * self.frequency_hz * retarded_s
        features = torch.cat((relative_xyz, source_normalized[:, :3],
                              travel_s / self.time_length_s,
                              retarded_s / self.time_length_s,
                              torch.sin(phase), torch.cos(phase)), dim=1)
        values = torch.cat([expert(features) for expert in self.experts], dim=1)
        gate = self.source_gate(source_normalized[:, :3])
        return torch.sum(gate * values, dim=1, keepdim=True), gate, travel_s

    def prototype_separation_loss(self):
        scaled = self.prototype_centers / self.prototype_scales
        distances = torch.cdist(scaled, scaled).square()
        mask = ~torch.eye(self.expert_count, device=distances.device, dtype=torch.bool)
        return torch.exp(-distances[mask]).mean()
