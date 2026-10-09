"""The inverse objective uses the same full-record RMS scale as training."""

import unittest

import numpy as np
import torch

from invert_source_conditioned import source_gradient, waveform_mse


class TinyPhysicalWavefield(torch.nn.Module):
    def forward(self, space_time, source):
        return (1e-6 * source[:, :1]).contiguous()


class InversionScaleTests(unittest.TestCase):
    def test_objective_and_gradient_share_one_physical_scale(self):
        model = TinyPhysicalWavefield()
        xyz, times = torch.zeros(2, 3), torch.zeros(3)
        observed = torch.full((2, 3), 0.1e-6)
        source = np.array([0.25, 0.3, 0.2, 0.1])
        loss = waveform_mse(model, xyz, times, observed,
                            torch.tensor(source, dtype=torch.float32), 1e-6)
        self.assertAlmostEqual(float(loss), 0.0225, places=5)
        gradient = source_gradient(
            model, xyz, times, observed, source, 1e-6)
        self.assertAlmostEqual(float(gradient[0]), 0.3, places=5)
        np.testing.assert_allclose(gradient[1:], 0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
