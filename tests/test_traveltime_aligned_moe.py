"""Differentiability and geometry checks for the FMM-aligned MoE."""

import unittest

import numpy as np
import torch

from examples.run_traveltime_aligned_moe_localization import (
    choose_diverse_starts,
    coarse_grid,
)
from traveltime_aligned_moe import (
    ReceiverTravelTimeInterpolator,
    TravelTimeAlignedPrototypeMoE,
)


class TravelTimeAlignedMoETests(unittest.TestCase):
    @staticmethod
    def linear_fields(receiver_count=2):
        grid = torch.stack(torch.meshgrid(
            torch.arange(5, dtype=torch.float32),
            torch.arange(5, dtype=torch.float32),
            torch.arange(5, dtype=torch.float32), indexing="ij"))
        return torch.stack([
            0.1 + (r + 1) * 0.02 * grid[0] + 0.03 * grid[1]
            + 0.04 * grid[2]
            for r in range(receiver_count)
        ])

    def test_trilinear_values_and_spatial_gradients_are_exact_for_linear_field(self):
        interpolator = ReceiverTravelTimeInterpolator(self.linear_fields(), 10.0)
        source = torch.tensor([[12.5, 25.0, 17.5],
                               [21.0, 13.0, 32.0]], requires_grad=True)
        receivers = torch.tensor([0, 1])
        values = interpolator(source, receivers)
        expected = torch.tensor([
            0.1 + 0.02 * 1.25 + 0.03 * 2.5 + 0.04 * 1.75,
            0.1 + 0.04 * 2.1 + 0.03 * 1.3 + 0.04 * 3.2,
        ])
        torch.testing.assert_close(values[:, 0], expected)
        gradient = torch.autograd.grad(values.sum(), source)[0]
        torch.testing.assert_close(
            gradient,
            torch.tensor([[0.002, 0.003, 0.004],
                          [0.004, 0.003, 0.004]]),
            atol=1e-7, rtol=1e-6)

    def test_prediction_has_finite_gradients_to_all_source_parameters(self):
        torch.manual_seed(4)
        interpolator = ReceiverTravelTimeInterpolator(self.linear_fields(), 10.0)
        model = TravelTimeAlignedPrototypeMoE(
            interpolator,
            receiver_xyz_m=torch.tensor([[0.0, 0.0, 0.0],
                                         [40.0, 40.0, 40.0]]),
            spatial_lengths_m=(40.0, 40.0, 40.0),
            time_length_s=2.0, frequency_hz=5.0,
            expert_count=3, hidden_dim=16, hidden_layers=2,
            prototype_centers=torch.tensor([
                [0.2, 0.2, 0.2], [0.7, 0.2, 0.4], [0.5, 0.8, 0.6]]))
        points = torch.tensor([[0.0, 0.0, 0.0, 0.6],
                               [1.0, 1.0, 1.0, 0.7]])
        sources = torch.tensor([[0.3, 0.4, 0.2, 0.10],
                                [0.6, 0.3, 0.7, 0.20]], requires_grad=True)
        prediction, gate, travel = model(points, sources, torch.tensor([0, 1]))
        gradient = torch.autograd.grad(prediction.sum(), sources)[0]
        self.assertEqual(prediction.shape, (2, 1))
        self.assertEqual(gate.shape, (2, 3))
        self.assertEqual(travel.shape, (2, 1))
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertTrue((gradient.abs().sum(dim=0) > 0).all().item())
        torch.testing.assert_close(gate.sum(dim=1), torch.ones(2))

    def test_gate_uses_spatial_source_but_not_origin_time(self):
        interpolator = ReceiverTravelTimeInterpolator(self.linear_fields(1), 10.0)
        model = TravelTimeAlignedPrototypeMoE(
            interpolator, torch.zeros(1, 3), (40.0, 40.0, 40.0),
            expert_count=3, hidden_dim=8, hidden_layers=1,
            prototype_centers=torch.tensor([
                [0.2, 0.2, 0.2], [0.7, 0.2, 0.4], [0.5, 0.8, 0.6]]))
        source_xyz = torch.tensor([[0.35, 0.25, 0.30]])
        first = model.source_gate(source_xyz)
        second = model.source_gate(source_xyz.clone())
        torch.testing.assert_close(first, second)
        self.assertLess(float(first.max()), 1.0)

    def test_localization_grid_and_diverse_start_selection(self):
        bounds = np.asarray([[0, 10], [0, 20], [0, 4], [0, 1]],
                            dtype=np.float32)
        candidates = coarse_grid(bounds, [2, 2, 2, 2])
        self.assertEqual(candidates.shape, (16, 4))
        scores = np.arange(len(candidates), dtype=np.float64)
        selected = choose_diverse_starts(candidates, scores, 3, 1.0)
        self.assertEqual(selected, [0, 2, 4])
        xyz = candidates[selected, :3]
        self.assertTrue(all(np.linalg.norm(xyz[i] - xyz[j]) >= 1.0
                            for i in range(len(xyz)) for j in range(i)))


if __name__ == "__main__":
    unittest.main()
