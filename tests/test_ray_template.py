"""Controls for straight-ray template timing and interpolation."""

import unittest

import numpy as np

from examples.compare_ray_shifted_template import (
    interpolate_velocity, ray_travel_times, shifted_template)


class RayTemplateTests(unittest.TestCase):
    def test_constant_velocity_integral_and_grid_edge(self):
        velocity = np.full((3, 3, 3), 4000.0)
        source = np.array([0.0, 0.0, 0.0, 0.1])
        receiver = np.array([[80.0, 0.0, 0.0], [40.0, 40.0, 0.0]])
        np.testing.assert_allclose(interpolate_velocity(velocity, receiver), 4000.0)
        np.testing.assert_allclose(ray_travel_times(velocity, source, receiver),
                                   np.linalg.norm(receiver, axis=1) / 4000.0)

    def test_same_source_reproduces_template(self):
        times = np.arange(20, dtype=np.float64) * 0.1
        traces = np.stack([np.sin(times), np.cos(times)])
        source = np.array([0.0, 0.0, 0.0, 0.1])
        receivers = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
        travel = np.array([0.025, 0.025])
        result = shifted_template(traces, times, source, source, receivers,
                                  travel, travel)
        np.testing.assert_allclose(result, traces, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
