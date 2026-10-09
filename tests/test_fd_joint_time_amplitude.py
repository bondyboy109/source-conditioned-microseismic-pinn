import numpy as np

from examples.audit_fd_joint_time_amplitude import score_joint


def test_joint_recovers_time_and_gain_with_zero_padding():
    template = np.zeros((2, 30))
    template[0, 5:9] = [1.0, -0.5, 0.2, -0.1]
    template[1, 10:12] = [0.8, -0.4]
    observed = np.zeros((2, 20))
    observed[:, 3:] = 0.25 * template[:, :17]
    result = score_joint(template, observed, 0.01, 0.05, (0.05, 0.15))
    assert result["lag_samples"] == 3
    assert np.isclose(result["nonnegative_gain"], 0.25)
    assert result["normalized_mse"] < 1e-12


def test_joint_rejects_antiphase_without_negative_gain():
    template = np.zeros((1, 12))
    template[0, 2:5] = [1.0, 0.5, 0.25]
    observed = -template[:, :10]
    result = score_joint(template, observed, 0.01, 0.05, (0.05, 0.05))
    assert result["nonnegative_gain"] == 0
    assert np.isclose(result["normalized_mse"], 1.0)
