"""Check regional search plumbing before using new FD confirmation events."""

import unittest

import numpy as np
import torch

from examples.confirm_depth_gap_25 import coarse_grid, inversion_without_truth
from source_conditioned_pinn import conditioned_prediction


class FourParameterToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, space_time, source):
        x, y, t = space_time[:, 0], space_time[:, 1], space_time[:, 3]
        return (source[:, 0] + source[:, 1] * x + source[:, 2] * y
                + source[:, 3] * t + self.anchor * 0).unsqueeze(-1)


class RegisteredSearchTests(unittest.TestCase):
    def test_grid_includes_both_bounds_without_truth_candidate_insertion(self):
        bounds = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.float32)
        candidates = coarse_grid(bounds, [3, 4, 5, 6])
        self.assertEqual(candidates.shape, (360, 4))
        np.testing.assert_array_equal(candidates[0], bounds[:, 0])
        np.testing.assert_array_equal(candidates[-1], bounds[:, 1])

    def test_frozen_regional_search_recovers_identifiable_toy(self):
        model = FourParameterToy()
        model.eval().requires_grad_(False)
        xyz = np.array([[0.0, 0.0, 0.0], [0.7, 0.1, 0.0],
                        [0.0, 0.9, 0.0], [0.8, 0.7, 0.0]], dtype=np.float32)
        times = np.linspace(0, 1, 9, dtype=np.float32)
        truth = torch.tensor([0.5, 0.5, 0.5, 0.5])
        with torch.inference_mode():
            data = conditioned_prediction(model, torch.as_tensor(xyz),
                                          torch.as_tensor(times), truth).numpy()
        config = {"coarse_grid_xyzt": [3, 3, 3, 3],
                  "coarse_receiver_stride": 1, "coarse_time_stride": 1,
                  "restarts": 2, "refine_receiver_stride": 1,
                  "refine_time_stride": 1, "refine_steps": 10,
                  "refine_adam_learning_rate_on_box_fractions": 0.01}
        region = {"bounds_m_s": [[0., 1.]] * 4}
        result, _ = inversion_without_truth(
            model, data, times, xyz, region, config, np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(result["estimated_source_m_s"], truth.numpy(), atol=1e-5)
        self.assertFalse(result["source_truth_used_in_search"])
        self.assertEqual(result["grid_candidates"], 81)


if __name__ == "__main__":
    unittest.main()
