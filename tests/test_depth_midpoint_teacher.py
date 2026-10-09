"""Checks for FD midpoint pseudo-label construction."""

import torch

from examples.run_depth_midpoint_teacher_finetune import build_pseudo_events


class DummyData:
    metadata = [{"path": "low.npz"}, {"path": "high.npz"}]
    points = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    events = [
        (points, torch.tensor([1., 2.]), torch.tensor([0., 0., .2, .1]),
         torch.tensor([0, 1]), 2.),
        (points, torch.tensor([3., 4.]), torch.tensor([0., 0., .6, .1]),
         torch.tensor([0, 1]), 4.),
    ]


def test_midpoint_teacher_averages_raw_endpoint_waveforms(tmp_path, monkeypatch):
    low = (tmp_path / "low.npz").resolve()
    high = (tmp_path / "high.npz").resolve()
    data = DummyData()
    data.metadata = [{"path": str(low)}, {"path": str(high)}]
    pairs = [{"low_path": low, "high_path": high}]
    event = build_pseudo_events(data, pairs)[0]
    raw = event[1] * event[4]
    torch.testing.assert_close(raw, torch.tensor([7., 10.]))
    torch.testing.assert_close(event[2], torch.tensor([0., 0., .4, .1]))
