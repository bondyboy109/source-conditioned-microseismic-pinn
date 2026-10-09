"""A zero waveform must use the same sampling and normalization as MoE scoring."""

import numpy as np

from examples.audit_refined_moe_truth_waveform import zero_waveform_loss


def test_zero_loss_uses_full_rms_and_downsampled_receiver_records():
    data = np.arange(1, 17, dtype=np.float32).reshape(2, 8)
    expected = np.mean(data[::2, ::8].astype(np.float64) ** 2) / np.mean(
        data.astype(np.float64) ** 2)
    assert np.isclose(zero_waveform_loss(data), expected)
