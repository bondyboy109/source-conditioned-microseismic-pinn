"""Ensure the diagnostic refines candidate spacing without changing its objective."""

import numpy as np

from examples.audit_arrival_spatial_resolution import rank_candidates, refined_search
from examples.invert_fmm_receiver_fields import interpolate_fields


def test_refined_search_finds_off_grid_ideal_arrivals():
    spec = {"source_bounds_m": [[2., 8.]] * 3,
            "coarse_grid": [2, 2, 2], "fmm_spacing_m": 1.,
            "origin_bounds_s": [0.05, 0.5], "min_valid_picks": 4}
    xyz = np.indices((10, 10, 10), dtype=np.float32)
    fields = np.stack([0.1 + 0.011 * (i + 1) * xyz[0]
                       + 0.017 * (i % 3 + 1) * xyz[1]
                       + 0.021 * (5 - i) * xyz[2] for i in range(5)])
    truth = np.array([4.25, 5.0, 3.5])
    picks = interpolate_fields(fields, truth[None, :], 1.)[0] + 0.2
    estimate, rmse, evaluated = refined_search(fields, picks, spec, seeds=3, rounds=3)
    assert evaluated > 8
    assert np.linalg.norm(estimate[:3] - truth) < 2.0
    np.testing.assert_allclose(rmse, rank_candidates(fields, picks,
                                                     estimate[None, :3], spec)[0][0])
