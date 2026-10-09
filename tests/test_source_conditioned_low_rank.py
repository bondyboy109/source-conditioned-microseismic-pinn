import unittest

import torch

from source_conditioned_low_rank import SourceConditionedLowRankOperator
from traveltime_aligned_moe import ReceiverTravelTimeInterpolator


class SourceConditionedLowRankTests(unittest.TestCase):
    def test_source_gradient_and_factorized_output(self):
        fields = torch.ones(2, 5, 5, 5) * 0.1
        interpolator = ReceiverTravelTimeInterpolator(fields, 10.0)
        model = SourceConditionedLowRankOperator(
            interpolator, torch.tensor([[0., 0., 0.], [40., 40., 40.]]),
            (40., 40., 40.), rank=4, hidden_dim=8, hidden_layers=2)
        points = torch.tensor([[0., 0., 0., 0.7], [1., 1., 1., 0.8]])
        source = torch.tensor([[0.3, 0.4, 0.2, 0.10],
                               [0.6, 0.3, 0.7, 0.20]], requires_grad=True)
        prediction, coefficients, travel = model(points, source, torch.tensor([0, 1]))
        gradient = torch.autograd.grad(prediction.sum(), source)[0]
        self.assertEqual(prediction.shape, (2, 1))
        self.assertEqual(coefficients.shape, (2, 4))
        self.assertEqual(travel.shape, (2, 1))
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertTrue((gradient[:, :3].abs().sum(dim=0) > 0).all().item())

    def test_causal_gate_suppresses_prearrival_prediction(self):
        fields = torch.ones(1, 5, 5, 5) * 0.1
        model = SourceConditionedLowRankOperator(
            ReceiverTravelTimeInterpolator(fields, 10.0),
            torch.tensor([[0., 0., 0.]]), (40., 40., 40.), rank=4,
            hidden_dim=8, hidden_layers=1)
        source = torch.tensor([[0.2, 0.2, 0.2, 0.5]])
        before = torch.tensor([[0., 0., 0., 0.0]])
        after = torch.tensor([[0., 0., 0., 0.8]])
        with torch.no_grad():
            early = model(before, source, torch.tensor([0]))[0].abs().item()
            late = model(after, source, torch.tensor([0]))[0].abs().item()
        self.assertLess(early, late + 1e-6)


if __name__ == "__main__":
    unittest.main()
