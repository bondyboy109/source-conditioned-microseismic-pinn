"""Controls for the optional FD boundary-sensitivity sponge."""

import unittest

import numpy as np
import torch

from examples.generate_fd_observation import sponge_damping


class SpongeTests(unittest.TestCase):
    def test_only_sides_and_bottom_are_damped(self):
        damping = sponge_damping((41, 41, 41), np.array([40.0] * 3),
                                 400.0, 60.0, torch.device("cpu"))
        self.assertEqual(damping.shape, (41, 41, 41))
        self.assertEqual(damping[20, 20, 0].item(), 0.0)
        self.assertEqual(damping[20, 20, 20].item(), 0.0)
        self.assertEqual(damping[20, 20, 40].item(), 60.0)
        self.assertEqual(damping[0, 20, 0].item(), 60.0)
        self.assertEqual(damping[20, 0, 0].item(), 60.0)
        self.assertEqual(damping[10, 20, 0].item(), 0.0)
        self.assertEqual(damping[5, 20, 0].item(), 15.0)


if __name__ == "__main__":
    unittest.main()
