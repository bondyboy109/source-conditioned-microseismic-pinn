"""Validate arrival-candidate ranking and explicit insufficient-pick handling."""

import unittest

import numpy as np

from examples.invert_fmm_receiver_fields import interpolate_fields
from examples.calibrate_arrival_bias_development import estimate_from_picks
from examples.validate_arrival_constrained import source_search


class ArrivalConstrainedValidationTests(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "source_bounds_m": [[2.0, 8.0], [2.0, 8.0], [2.0, 8.0]],
            "coarse_grid": [2, 2, 2], "local_step_m": [1.0, 1.0, 1.0],
            "fmm_spacing_m": 1.0, "origin_bounds_s": [0.05, 0.5],
            "min_valid_picks": 4,
        }
        grid = np.indices((10, 10, 10), dtype=np.float32)
        self.fields = np.stack([
            0.1 + 0.02 * (i + 1) * grid[0] + 0.03 * (i + 2) * grid[1]
            + 0.04 * (6 - i) * grid[2] for i in range(5)
        ])

    def test_finds_known_coarse_source_and_origin(self):
        truth = np.array([[2.0, 2.0, 2.0]])
        picks = interpolate_fields(self.fields, truth, 1.0)[0] + 0.2
        candidates, residuals, _ = source_search(self.fields, picks, self.spec)
        recovered = candidates[int(np.argmin(residuals))]
        np.testing.assert_allclose(recovered, [2.0, 2.0, 2.0, 0.2], atol=1e-5)

    def test_insufficient_picks_reports_failure(self):
        picks = np.array([0.2, np.nan, np.nan, 0.3, np.nan])
        self.assertIsNone(source_search(self.fields, picks, self.spec))

    def test_fixed_picker_bias_corrects_origin_without_truth_in_search(self):
        truth = np.array([2.0, 2.0, 2.0, 0.3])
        bias = -0.1
        picks = interpolate_fields(self.fields, truth[None, :3], 1.0)[0] + truth[3] + bias
        raw = estimate_from_picks(self.fields, picks, truth, self.spec)
        corrected = estimate_from_picks(self.fields, picks - bias, truth, self.spec)
        self.assertEqual(raw["status"], "ok")
        self.assertEqual(corrected["status"], "ok")
        self.assertAlmostEqual(raw["origin_time_error_s"], 0.1)
        self.assertAlmostEqual(corrected["origin_time_error_s"], 0.0)
        self.assertAlmostEqual(corrected["location_error_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
