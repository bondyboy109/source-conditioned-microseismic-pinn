"""Receiver mini-batches stay paired when physics uses the global RNG."""

import unittest

import numpy as np
import torch

from receiver_training import ReceiverTrainingData


def sampler(seed):
    data = ReceiverTrainingData.__new__(ReceiverTrainingData)
    data._numpy_rng = np.random.default_rng(seed)
    data._torch_rng = torch.Generator(device="cpu").manual_seed(seed)
    data.events = []
    for event_id in range(3):
        x = torch.arange(80, dtype=torch.float32).reshape(20, 4) + event_id
        y = torch.arange(20, dtype=torch.float32) + event_id
        src = torch.full((4,), float(event_id))
        active = torch.arange(4, 12)
        data.events.append((x, y, src, active, float(event_id + 1)))
    return data


class ReceiverSamplingTests(unittest.TestCase):
    def test_global_rng_consumption_does_not_change_paired_batches(self):
        left, right = sampler(123), sampler(123)
        for _ in range(5):
            expected = left.batch(18, event_count=3)
            np.random.random(31)
            torch.rand(23)
            actual = right.batch(18, event_count=3)
            for first, second in zip(expected, actual):
                torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
