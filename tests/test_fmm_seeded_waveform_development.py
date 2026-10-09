"""Candidate generation and scoring for the development hybrid search."""

import numpy as np
import torch

from examples.run_fmm_seeded_waveform_development import (
    choose_spatial_starts,
    fine_scores,
    proposals,
)
from invert_source_conditioned import waveform_mse


class ToyWavefield(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, points, sources):
        return (points[:, 0] + points[:, 3] + sources[:, 0] * 2
                + sources[:, 3] + self.weight).unsqueeze(-1)


def test_clipped_proposals_have_unique_spatial_locations_and_no_truth_input():
    bounds = np.array([[0., 10.], [0., 10.], [0., 10.], [.1, .3]])
    candidates = proposals(np.array([0., 0., 5.]), bounds,
                           [-2., 0., 2.], np.array([.1, .2, .3]))
    assert len(candidates) == 12 * 3
    assert np.all(candidates[:, :3] >= bounds[:3, 0])
    assert np.all(candidates[:, :3] <= bounds[:3, 1])


def test_spatial_starts_keep_lowest_loss_time_and_diverse_positions():
    candidates = np.array([[1., 0., 0., .1], [1., 0., 0., .2],
                           [1.1, 0., 0., .1], [5., 0., 0., .1]], dtype=np.float32)
    scores = np.array([.3, .1, .2, .4])
    assert choose_spatial_starts(candidates, scores, 3, .5) == [1, 3]


def test_fine_score_matches_the_exact_refinement_objective():
    model = ToyWavefield()
    receivers = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
    times = np.arange(6, dtype=np.float32)
    observed = np.ones((2, 6), dtype=np.float32)
    scale = np.ones(4, dtype=np.float32)
    candidates = np.array([[0., 0., 0., 0.], [1., 0., 0., 2.]], dtype=np.float32)
    scores = fine_scores(model, receivers, times, observed, scale, 2., candidates,
                         receiver_stride=1, time_stride=2, batch_size=2)
    for index, candidate in enumerate(candidates):
        expected = waveform_mse(model, torch.as_tensor(receivers),
                                torch.as_tensor(times[::2]),
                                torch.as_tensor(observed[:, ::2]),
                                torch.as_tensor(candidate), 2.)
        assert np.isclose(scores[index], float(expected))
