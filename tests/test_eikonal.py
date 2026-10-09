"""Core properties of the factorized eikonal model."""

import unittest

import torch

from eikonal_traveltime import FactorizedEikonalNet, eikonal_loss


class EikonalTests(unittest.TestCase):
    def test_source_condition_is_exact_and_gradients_are_finite(self):
        model = FactorizedEikonalNet(hidden_dim=8, num_layers=2)
        source = torch.tensor([[0.3, 0.4, 0.2]])
        self.assertEqual(model(source, source).item(), 0.0)
        xyz = torch.tensor([[0.5, 0.4, 0.2]], requires_grad=True)
        gradient = torch.autograd.grad(model(xyz, source).sum(), xyz)[0]
        self.assertTrue(torch.isfinite(gradient).all().item())

    def test_eikonal_loss_is_finite(self):
        model = FactorizedEikonalNet(hidden_dim=8, num_layers=2)
        xyz = torch.rand(16, 3)
        source = torch.rand(16, 3)
        velocity = torch.full((16, 1), 3500.0)
        loss = eikonal_loss(model, xyz, source, velocity)
        self.assertTrue(torch.isfinite(loss).item())

    def test_quadratic_core_has_small_far_field_distance_bias(self):
        model = FactorizedEikonalNet(hidden_dim=8, num_layers=2,
                                    domain_m=(1000, 1000, 1000), core_m=20)
        xyz = torch.tensor([[1.0, 0.0, 0.0]])
        source = torch.zeros_like(xyz)
        relative_distance = 1000.0
        smooth_distance = relative_distance ** 2 / (
            relative_distance ** 2 + 20.0 ** 2) ** 0.5
        # The network slowness is bounded, so recover its implied distance by
        # dividing out the separately evaluated learned slowness.
        raw = model.net(torch.cat((xyz, source), dim=-1))
        slow = 1 / model.velocity_max + (
            1 / model.velocity_min - 1 / model.velocity_max) * torch.sigmoid(raw)
        recovered = (model(xyz, source) / slow).item()
        self.assertAlmostEqual(recovered, smooth_distance, places=3)
        self.assertLess(abs(recovered - relative_distance), 0.21)


if __name__ == "__main__":
    unittest.main()
