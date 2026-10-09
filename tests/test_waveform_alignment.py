"""Sanity checks for diagnostic lag sign and overlap normalization."""

import unittest

import numpy as np

from examples.diagnose_loo_waveform_alignment import trace_alignment


class WaveformAlignmentTests(unittest.TestCase):
    def test_positive_lag_delays_prediction(self):
        reference = np.zeros(64)
        predicted = np.zeros(64)
        reference[30:34] = [1.0, 2.0, -1.0, -2.0]
        predicted[25:29] = reference[30:34]
        result = trace_alignment(reference, predicted, 8)
        self.assertEqual(result["best_shift_samples"], 5)
        self.assertAlmostEqual(result["best_signed_cosine"], 1.0)
        self.assertAlmostEqual(result["best_gain_and_shift_relative_l2_diagnostic"], 0.0)

    def test_zero_energy_is_rejected(self):
        with self.assertRaises(ValueError):
            trace_alignment(np.zeros(10), np.ones(10), 2)


if __name__ == "__main__":
    unittest.main()
