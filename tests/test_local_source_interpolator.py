"""Checks for the time-aligned local waveform interpolation baseline."""

import numpy as np

from examples.audit_local_source_interpolator import shift_origin


def test_origin_shift_delays_trace_when_target_origin_is_later():
    times = np.arange(5, dtype=np.float64)
    data = np.asarray([[0., 1., 0., 0., 0.]])
    shifted = shift_origin(data, times, training_origin_s=0., target_origin_s=1.)
    np.testing.assert_allclose(shifted, [[0., 0., 1., 0., 0.]])
