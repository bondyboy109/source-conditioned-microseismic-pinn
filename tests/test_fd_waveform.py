import numpy as np

from fd_waveform import best_origin_time


def test_best_origin_time_recovers_zero_padded_delay_without_gain_fit():
    template = np.zeros((2, 20), dtype=np.float64)
    template[0, 3:6] = [1.0, -0.5, 0.25]
    template[1, 7:9] = [0.2, -0.4]
    observed = np.zeros_like(template)
    observed[:, 4:] = template[:, :-4]

    result = best_origin_time(template, observed, dt=0.01,
                              reference_origin_s=0.05,
                              bounds_s=(0.05, 0.15))

    assert result["lag_samples"] == 4
    assert np.isclose(result["origin_time_s"], 0.09)
    assert result["relative_l2"] < 1e-12
