"""The diagnostic partitions must reconstruct the existing MSE exactly."""

import numpy as np

from examples.audit_cross_region_objective import loss_parts


def test_window_and_receiver_parts_sum_to_full_record_scaled_mse():
    observed = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.]])
    at_truth = observed + np.array([[0., 1., 0., 1.], [1., 0., 1., 0.]])
    returned = observed + np.array([[1., 1., 1., 1.], [0., 0., 0., 0.]])
    full_record_rms = 3.
    result = loss_parts(observed, at_truth, returned,
                        np.array([0., .5, 1., 1.5]), np.zeros((2, 3)),
                        full_record_rms)
    expected_truth = np.mean(((at_truth - observed) / full_record_rms) ** 2)
    expected_returned = np.mean(((returned - observed) / full_record_rms) ** 2)
    assert np.isclose(result["true_source_normalized_mse"], expected_truth)
    assert np.isclose(result["returned_source_normalized_mse"], expected_returned)
    for part in ("receivers", "windows"):
        assert np.isclose(sum(row["true_source_mse_contribution"]
                              for row in result[part]), expected_truth)
        assert np.isclose(sum(row["returned_minus_truth"]
                              for row in result[part]), expected_returned - expected_truth)
