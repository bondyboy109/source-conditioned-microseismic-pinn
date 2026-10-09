"""FD pair supervision must compare the same receiver and time sample."""

import numpy as np
import torch

from examples.run_source_transfer_fd_pair import paired_batch


def test_paired_batch_uses_matched_coordinates_and_per_event_rms():
    receivers = np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32)
    times = np.array([0, 0.1, 0.2], dtype=np.float32)
    a_data = np.array([[1, 2, 4], [2, 3, 5]], dtype=np.float32)
    b_data = np.array([[3, 5, 8], [4, 7, 10]], dtype=np.float32)
    events = [
        {"source": np.array([10, 20, 30, 0.05], dtype=np.float32),
         "data": a_data, "rms": 2.0, "active": np.arange(6)},
        {"source": np.array([30, 20, 30, 0.05], dtype=np.float32),
         "data": b_data, "rms": 4.0, "active": np.arange(6)},
    ]
    domain = np.array([200, 200, 200, 1], dtype=np.float32)
    points, ids, sources_a, sources_b, delta = paired_batch(
        events, receivers, times, {"a": 0, "b": 1}, 40, 0.75,
        np.random.default_rng(17), domain, torch.device("cpu"))
    receiver_index = ids.numpy()
    time_index = np.rint(points.numpy()[:, 3] / 0.1).astype(int)
    expected = b_data[receiver_index, time_index] / 4 - (
        a_data[receiver_index, time_index] / 2)
    np.testing.assert_allclose(delta.numpy().ravel(), expected, atol=1e-6)
    np.testing.assert_allclose(points.numpy()[:, :3] * domain[:3],
                               receivers[receiver_index], atol=1e-6)
    np.testing.assert_allclose(sources_a.numpy() * domain,
                               np.broadcast_to(events[0]["source"], (40, 4)),
                               atol=1e-6)
    np.testing.assert_allclose(sources_b.numpy() * domain,
                               np.broadcast_to(events[1]["source"], (40, 4)),
                               atol=1e-6)
