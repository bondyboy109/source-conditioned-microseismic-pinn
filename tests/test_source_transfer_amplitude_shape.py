import numpy as np
import pytest

from examples.audit_source_transfer_amplitude_shape import amplitude_metrics


def test_shared_scale_and_distinct_receiver_scales():
    observed = np.array([[1.0, -2.0, 1.0], [2.0, 1.0, -1.0]])
    shared = amplitude_metrics(observed, observed * 0.25, 1e-4)
    assert shared["best_nonnegative_global_gain"] == pytest.approx(4.0)
    assert shared["prediction_to_observation_energy_ratio"] == pytest.approx(0.0625)
    assert shared["global_gain_relative_l2"] == pytest.approx(0.0)
    distinct = amplitude_metrics(observed, observed * np.array([[0.25], [2.0]]), 1e-4)
    assert distinct["global_gain_relative_l2"] > 0.1
    assert distinct["receiver_gain_oracle_relative_l2"] == pytest.approx(0.0)


def test_polarity_reversal_cannot_be_repaired_by_nonnegative_gain():
    observed = np.array([[1.0, -1.0], [2.0, -2.0]])
    stats = amplitude_metrics(observed, -observed, 1e-4)
    assert stats["waveform_cosine"] == pytest.approx(-1.0)
    assert stats["global_gain_relative_l2"] == pytest.approx(1.0)
    assert stats["receiver_gain_oracle_relative_l2"] == pytest.approx(1.0)


def test_zero_prediction_and_inactive_receiver():
    observed = np.array([[1.0, -1.0], [0.0, 0.0]])
    stats = amplitude_metrics(observed, np.zeros_like(observed), 1e-4)
    assert stats["waveform_cosine"] is None
    assert stats["raw_relative_l2"] == pytest.approx(1.0)
    assert stats["global_gain_relative_l2"] == pytest.approx(1.0)
    assert stats["active_receiver_ids"] == [0]
