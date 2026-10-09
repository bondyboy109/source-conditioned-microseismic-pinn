"""Shared candidate pool must preserve the frozen refined FMM optimum."""

import numpy as np

from examples.audit_arrival_spatial_resolution import refined_search
from examples.compare_refined_fmm_moe import candidate_pool
from examples.invert_fmm_receiver_fields import interpolate_fields


def test_candidate_union_retains_refined_optimum():
    spec = {"source_bounds_m": [[2., 8.]] * 3, "coarse_grid": [2, 2, 2],
            "fmm_spacing_m": 1., "origin_bounds_s": [0.05, 0.5],
            "min_valid_picks": 4}
    xyz = np.indices((10, 10, 10), dtype=np.float32)
    fields = np.stack([0.1 + 0.011 * (i + 1) * xyz[0]
                       + 0.017 * (i % 3 + 1) * xyz[1]
                       + 0.021 * (5 - i) * xyz[2] for i in range(5)])
    truth = np.array([4.25, 5.0, 3.5])
    picks = interpolate_fields(fields, truth[None, :], 1.)[0] + 0.2
    selected, score, _ = refined_search(fields, picks, spec)
    pool, scores = candidate_pool(fields, picks, spec)
    idx = np.argmin(scores)
    np.testing.assert_allclose(pool[idx], selected, atol=1e-7)
    np.testing.assert_allclose(scores[idx], score, atol=1e-9)
    assert len(pool) > 8
