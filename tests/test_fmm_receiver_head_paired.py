"""Check that FMM receiver times stay paired with sampled FD observations."""

from types import SimpleNamespace

import numpy as np
import torch

from examples.run_fmm_receiver_head_paired import sampled_batch


def test_receiver_time_uses_same_event_and_station_as_waveform_sample():
    events = []
    travel = torch.arange(50, dtype=torch.float32).reshape(2, 25) / 100
    for event in range(2):
        positions = torch.zeros((25 * 1001, 4))
        positions[:, 0] = event
        positions[:, 1] = torch.arange(25).repeat_interleave(1001)
        target = torch.zeros(25 * 1001)
        source = torch.tensor([float(event), 0., 0., 0.])
        active = torch.tensor([1001 * 7 + 100, 1001 * 18 + 300])
        events.append((positions, target, source, active, 1.))
    sampled = sampled_batch(SimpleNamespace(events=events), np.random.default_rng(42),
                            torch.Generator().manual_seed(42), travel, 2, 8)
    points, _, source, _, carrier = sampled
    for i in range(len(points)):
        event, station = int(source[i, 0]), int(points[i, 1])
        assert carrier[i, 0] == travel[event, station]
