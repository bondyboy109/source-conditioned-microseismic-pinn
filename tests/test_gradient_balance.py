"""Checks for the optional receiver/physics gradient balancing rule."""

import unittest

import torch

from source_conditioned_pinn import (SourceConditionedConfig,
                                     SourceConditionedWavePINN,
                                     gradient_balance_weight, training_step)


class GradientBalanceTests(unittest.TestCase):
    def test_matches_last_layer_gradient_magnitude(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        reference = (2.0 * parameter).square().sum()
        physical = (3.0 * parameter).square().sum()
        weight = gradient_balance_weight(physical, reference, (parameter,))
        physical_grad = torch.autograd.grad(physical, parameter, retain_graph=True)[0]
        reference_grad = torch.autograd.grad(reference, parameter)[0]
        self.assertAlmostEqual((weight * physical_grad).norm().item(),
                               reference_grad.norm().item(), places=5)

    def test_zero_gradient_component_is_ignored(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        reference = parameter.square().sum()
        zero = 0.0 * parameter.square().sum()
        self.assertEqual(gradient_balance_weight(zero, reference, (parameter,)), 0.0)

    def test_receiver_only_does_not_evaluate_physics(self):
        model = SourceConditionedWavePINN(hidden_dim=8, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        receiver_batch = (torch.rand(4, 4), torch.zeros(4),
                          torch.tensor([0.5, 0.5, 0.2, 0.01]), 1.0)

        def forbidden_velocity_lookup(_):
            raise AssertionError("Velocity must not be used for receiver-only training")

        metrics = training_step(
            model, optimizer, forbidden_velocity_lookup, frequency=5.0,
            n_phys=0, n_ic=0, n_bc_per_face=0, device=torch.device("cpu"),
            cfg=SourceConditionedConfig(), receiver_batch=receiver_batch,
            receiver_loss_weight=0.1, receiver_only=True)
        self.assertEqual(set(metrics), {"total", "receivers"})
        self.assertAlmostEqual(metrics["total"], 0.1 * metrics["receivers"])

    def test_receiver_only_accepts_per_sample_sources_and_scales(self):
        model = SourceConditionedWavePINN(hidden_dim=8, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        receiver_batch = (
            torch.rand(6, 4), torch.zeros(6), torch.rand(6, 4),
            torch.linspace(0.5, 1.0, 6),
        )
        metrics = training_step(
            model, optimizer, None, frequency=5.0, n_phys=0, n_ic=0,
            n_bc_per_face=0, device=torch.device("cpu"),
            cfg=SourceConditionedConfig(), receiver_batch=receiver_batch,
            receiver_loss_weight=0.1, receiver_only=True)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["total"])))


if __name__ == "__main__":
    unittest.main()
