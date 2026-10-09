"""Depth objective ranks waveform candidates by normalized misfit."""

import numpy as np

from examples.audit_fmm_receiver_head_depth_objective import rank_errors


def test_true_depth_ranking_and_anchor_loss():
    fd = np.asarray([[1., 2.]], dtype=np.float64)
    candidates = [np.asarray([[2., 2.]]), fd.copy(), np.asarray([[4., 2.]]),
                  np.asarray([[3., 2.]]), np.asarray([[1., 3.]])]
    result = rank_errors(fd, candidates, true_index=1)
    assert result["best_index"] == result["true_index"] == result["true_rank"] == 1
    assert result["true_mse"] == 0
    assert result["anchor_mse"] == .8
