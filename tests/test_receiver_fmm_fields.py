"""Check reciprocal-field sampling and insufficient-pick failure reporting."""

import unittest

import numpy as np

from examples.invert_fmm_receiver_fields import (
    InsufficientPicks,
    interpolate_fields,
    invert_one,
    score_candidates,
)


class ReciprocalFieldTests(unittest.TestCase):
    def test_interpolation_and_candidate_ranking_on_linear_traveltimes(self):
        grid = np.indices((5, 5, 5), dtype=np.float32)
        fields = np.stack([
            0.1 + (.02 * (i + 1)) * grid[0] + (.03 * (i + 2)) * grid[1]
            + (.04 * (4 - i)) * grid[2]
            for i in range(4)
        ])
        truth = np.array([[1.25, 2.5, 1.75]])
        travel = interpolate_fields(fields, truth, 1.)
        expected = np.asarray([
            0.1 + .02 * (i + 1) * 1.25 + .03 * (i + 2) * 2.5
            + .04 * (4 - i) * 1.75 for i in range(4)
        ])
        np.testing.assert_allclose(travel[0], expected, atol=1e-7)
        observed = travel[0] + .2
        candidates = np.vstack((truth, [2.75, 1.0, 3.]))
        losses, origins = score_candidates(fields, candidates, observed, 1., (.05, .5))
        self.assertLess(losses[0], 1e-10)
        self.assertLess(losses[0], losses[1])
        self.assertAlmostEqual(float(origins[0]), .2, places=5)

    def test_no_valid_picks_is_explicit_failure(self):
        fields = np.zeros((2, 5, 5, 5), dtype=np.float32)
        with self.assertRaises(InsufficientPicks) as raised:
            invert_one(fields, np.zeros((2, 3)), np.zeros((2, 11)),
                       np.linspace(0., 1., 11), 1., (2, 2, 2), 3,
                       0, (.05, .5), .05, 1)
        self.assertEqual(raised.exception.count, 0)


if __name__ == "__main__":
    unittest.main()
