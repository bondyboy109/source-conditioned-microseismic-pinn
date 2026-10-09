"""Check local waveform interpolation metrics."""

import numpy as np

from examples.audit_depth_linear_interpolation import metrics


def test_exact_interpolation_has_zero_waveform_error():
    fd = np.asarray([[1., 2., 3.]], dtype=np.float64)
    result = metrics(fd, fd.copy(), fd.copy())
    assert result["full_l2"] == 0
    assert result["active_l2"] == 0
    assert result["response_l2"] == 0
    assert np.isclose(result["response_cosine"], 1.)
