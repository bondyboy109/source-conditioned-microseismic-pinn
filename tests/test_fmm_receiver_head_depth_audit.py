"""Check that pre-origin threshold crossings are counted on usable FD stations."""

import numpy as np

from examples.audit_fmm_receiver_head_depth import pick_diagnostic


def test_early_model_pick_is_not_mistaken_for_fd_arrival():
    times = np.arange(11, dtype=np.float64) * .1
    fd = np.zeros((2, len(times)), dtype=np.float64)
    predicted = np.zeros_like(fd)
    fd[0, 6:9] = 1.
    predicted[0, :3] = .2
    predicted[0, 6:9] = 1.
    fd[1, 6:9] = .001  # Too weak to be a usable FD trace.
    predicted[1, :3] = .2
    result = pick_diagnostic(fd, predicted, times, origin=.5)
    assert result["valid_fd_station_count"] == 1
    assert result["model_picks_before_origin_minus_0_10_s"] == 1
    assert result["model_picks_at_t0"] == 1
    assert result["raw_threshold_pick"]["paired_pick_count"] == 1
