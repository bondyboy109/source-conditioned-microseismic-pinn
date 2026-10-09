"""Checks for the frozen MoE stress-test perturbations."""

import numpy as np
import pytest
import torch

from examples.stress_test_traveltime_aligned_moe import (
    add_exact_awgn,
    retained_receivers,
    sampled_waveforms_available,
)
from examples.stress_test_velocity_bias import scale_traveltime_fields


def test_exact_awgn_matches_requested_whole_record_snr():
    data = np.linspace(-2.0, 2.0, 120, dtype=np.float32).reshape(6, 20)
    noisy, measured = add_exact_awgn(data, 10.0, 1234)
    noise = noisy.astype(np.float64) - data.astype(np.float64)
    actual = 20.0 * np.log10(
        np.sqrt(np.mean(data.astype(np.float64) ** 2))
        / np.sqrt(np.mean(noise ** 2)))
    assert measured == pytest.approx(10.0, abs=1e-10)
    assert actual == pytest.approx(10.0, abs=1e-6)


def test_retained_receiver_mask_is_reproducible_and_sized():
    first = retained_receivers(25, 0.40, 99)
    second = retained_receivers(25, 0.40, 99)
    assert len(first) == 15
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == len(first)


def test_masked_sampling_preserves_global_receiver_indices():
    receivers = np.arange(12, dtype=np.float32).reshape(4, 3)
    times = np.arange(6, dtype=np.float32)
    data = np.arange(24, dtype=np.float32).reshape(4, 6) + 1.0
    event = {"receivers": receivers, "times": times, "data": data}
    points, receiver_index, observed = sampled_waveforms_available(
        event, np.asarray([20.0, 20.0, 20.0, 10.0], dtype=np.float32),
        np.asarray([1, 2, 3]), receiver_stride=2, time_stride=3,
        device=torch.device("cpu"))
    assert points.shape == (4, 4)
    assert observed.shape == (4, 1)
    assert receiver_index.tolist() == [1, 1, 3, 3]


def test_uniform_velocity_factor_has_exact_inverse_traveltime_scaling():
    fields = np.asarray([[[[0.9, 1.8], [2.7, 3.6]]]], dtype=np.float32)
    scaled = scale_traveltime_fields(fields, 0.9)
    np.testing.assert_allclose(scaled, fields / 0.9, rtol=1e-6)
    with pytest.raises(ValueError):
        scale_traveltime_fields(fields, 0.0)
