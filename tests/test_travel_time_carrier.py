"""Source dependence and checkpoint semantics of the diagnostic carrier."""

import tempfile
import unittest
from pathlib import Path

import torch

from source_conditioned_pinn import (SourceConditionedConfig,
                                     EikonalTravelTimeCarrierWavePINN,
                                     TranslationInvariantTravelTimeCarrierWavePINN,
                                     TravelTimeCarrierWavePINN,
                                     load_conditioned_checkpoint,
                                     save_conditioned_checkpoint)
from eikonal_traveltime import FactorizedEikonalNet


class TravelTimeCarrierTests(unittest.TestCase):
    def test_time_hint_and_source_coordinates_are_differentiable(self):
        model = TravelTimeCarrierWavePINN(5.0, 9.0, hidden_dim=8, num_layers=2)
        points = torch.tensor([[0.6, 0.4, 0.1, 0.4],
                               [0.7, 0.2, 0.1, 0.5]])
        sources = torch.tensor([[0.2, 0.2, 0.2, 0.02],
                                [0.3, 0.5, 0.3, 0.03]], requires_grad=True)
        predictions = model(points, sources)
        gradients = torch.autograd.grad(predictions.sum(), sources)[0]
        self.assertEqual(predictions.shape, (2, 1))
        self.assertTrue(torch.isfinite(gradients).all().item())
        self.assertTrue((gradients[:, :3].abs().sum() > 0).item())
        self.assertTrue((gradients[:, 3].abs().sum() > 0).item())

    def test_smoothed_absolute_source_carrier_has_finite_source_curvature(self):
        model = TravelTimeCarrierWavePINN(
            5.0, 9.0, hidden_dim=8, num_layers=2,
            distance_core_m=20.0)
        point = torch.tensor([[0.3, 0.4, 0.2, 0.02]], requires_grad=True)
        source = point.detach().clone()
        gradient = torch.autograd.grad(model(point, source).sum(), point,
                                       create_graph=True)[0]
        for axis in range(4):
            second = torch.autograd.grad(gradient[:, axis].sum(), point,
                                         retain_graph=True)[0][:, axis]
            self.assertTrue(torch.isfinite(second).all().item())

    def test_ray_checkpoint_restores_mode_and_requires_velocity(self):
        model = TravelTimeCarrierWavePINN(
            5.0, 9.0, hidden_dim=8, num_layers=2,
            distance_core_m=20.0, ray_travel_time=True, ray_samples=8)
        model.set_velocity_grid(torch.full((9, 9, 5), 3500.0))
        point = torch.tensor([[0.3, 0.4, 0.2, 0.02]])
        source = torch.tensor([[0.2, 0.3, 0.1, 0.01]])
        expected = model(point, source)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ray.pth"
            save_conditioned_checkpoint(checkpoint, model, None, 5.0, 1,
                                        SourceConditionedConfig())
            restored, payload = load_conditioned_checkpoint(
                checkpoint, torch.device("cpu"))
        self.assertTrue(payload["architecture"]["ray_travel_time"])
        self.assertEqual(payload["architecture"]["ray_samples"], 8)
        with self.assertRaisesRegex(RuntimeError, "set_velocity_grid"):
            restored(point, source)
        restored.set_velocity_grid(torch.full((9, 9, 5), 3500.0))
        torch.testing.assert_close(expected, restored(point, source))

    def test_eikonal_carrier_checkpoint_round_trip(self):
        travel = FactorizedEikonalNet(hidden_dim=8, num_layers=2)
        model = EikonalTravelTimeCarrierWavePINN(
            5.0, 9.0, eikonal_model=travel, hidden_dim=8, num_layers=2,
            distance_core_m=20.0)
        point = torch.tensor([[0.4, 0.5, 0.2, 0.3]])
        source = torch.tensor([[0.2, 0.3, 0.1, 0.02]])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "eikonal_carrier.pth"
            save_conditioned_checkpoint(checkpoint, model, None, 5.0, 1,
                                        SourceConditionedConfig())
            restored, payload = load_conditioned_checkpoint(
                checkpoint, torch.device("cpu"))
        self.assertEqual(payload["architecture"]["kind"],
                         "eikonal_travel_time_carrier")
        torch.testing.assert_close(model(point, source), restored(point, source))

    def test_checkpoint_restores_time_hint_and_predictions(self):
        model = TravelTimeCarrierWavePINN(
            5.0, 9.0, reference_velocity_m_s=3500.0, hidden_dim=8,
            num_layers=2, output_scale=1e-5, hard_initial=True)
        points = torch.tensor([[0.6, 0.4, 0.1, 0.4]])
        sources = torch.tensor([[0.2, 0.2, 0.2, 0.02]])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            save_conditioned_checkpoint(checkpoint, model, None, 5.0, 3,
                                        SourceConditionedConfig(),
                                        training_record={"receiver_only": True})
            restored, payload = load_conditioned_checkpoint(checkpoint,
                                                            torch.device("cpu"))
        self.assertEqual(payload["architecture"]["kind"], "travel_time_carrier")
        self.assertEqual(payload["training_record"]["receiver_only"], True)
        self.assertEqual(restored.reference_velocity_m_s, 3500.0)
        torch.testing.assert_close(model(points, sources), restored(points, sources))

    def test_translation_invariant_model_ignores_common_spatial_shift(self):
        model = TranslationInvariantTravelTimeCarrierWavePINN(
            5.0, 9.0, reference_velocity_m_s=3500.0,
            hidden_dim=8, num_layers=2, output_scale=1e-5,
            distance_core_m=10.0)
        point = torch.tensor([[0.50, 0.45, 0.10, 0.30]])
        source = torch.tensor([[0.25, 0.20, 0.15, 0.02]])
        shift = torch.tensor([[0.10, 0.05, 0.02, 0.00]])
        torch.testing.assert_close(model(point, source),
                                   model(point + shift, source + shift))

    def test_translation_invariant_checkpoint_round_trip(self):
        model = TranslationInvariantTravelTimeCarrierWavePINN(
            5.0, 9.0, reference_velocity_m_s=3500.0,
            hidden_dim=8, num_layers=2, output_scale=1e-5,
            distance_core_m=10.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            save_conditioned_checkpoint(checkpoint, model, None, 5.0, 1,
                                        SourceConditionedConfig())
            restored, payload = load_conditioned_checkpoint(checkpoint,
                                                            torch.device("cpu"))
        self.assertEqual(payload["architecture"]["kind"],
                         "translation_invariant_travel_time_carrier")
        self.assertIsInstance(restored,
                              TranslationInvariantTravelTimeCarrierWavePINN)
        self.assertEqual(payload["architecture"]["version"], 3)
        self.assertEqual(payload["architecture"]["distance_core_m"], 10.0)
        self.assertEqual(restored.distance_core_m, 10.0)

    def test_version_two_checkpoint_restores_one_metre_core(self):
        model = TranslationInvariantTravelTimeCarrierWavePINN(
            5.0, 9.0, hidden_dim=8, num_layers=2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            save_conditioned_checkpoint(checkpoint, model, None, 5.0, 1,
                                        SourceConditionedConfig())
            payload = torch.load(checkpoint, weights_only=False)
            payload["architecture"]["version"] = 2
            del payload["architecture"]["distance_core_m"]
            torch.save(payload, checkpoint)
            restored, _ = load_conditioned_checkpoint(checkpoint,
                                                      torch.device("cpu"))
        self.assertEqual(restored.distance_core_m, 1.0)

    def test_old_translation_checkpoint_is_rejected(self):
        model = TranslationInvariantTravelTimeCarrierWavePINN(
            5.0, 9.0, hidden_dim=8, num_layers=2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            save_conditioned_checkpoint(checkpoint, model, None, 5.0, 1,
                                        SourceConditionedConfig())
            payload = torch.load(checkpoint, weights_only=False)
            del payload["architecture"]["version"]
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "predates"):
                load_conditioned_checkpoint(checkpoint, torch.device("cpu"))

    def test_translation_invariant_source_has_finite_second_derivatives(self):
        model = TranslationInvariantTravelTimeCarrierWavePINN(
            5.0, 9.0, hidden_dim=8, num_layers=2, output_scale=1e-5)
        point = torch.tensor([[0.3, 0.4, 0.2, 0.02]], requires_grad=True)
        source = point.detach().clone()
        gradient = torch.autograd.grad(model(point, source).sum(), point,
                                       create_graph=True)[0]
        for axis in range(4):
            second = torch.autograd.grad(gradient[:, axis].sum(), point,
                                         retain_graph=True)[0][:, axis]
            self.assertTrue(torch.isfinite(second).all().item())


if __name__ == "__main__":
    unittest.main()
