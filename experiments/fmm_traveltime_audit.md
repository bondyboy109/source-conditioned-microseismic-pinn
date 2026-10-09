# Numerical fast-marching traveltime audit

The learned eikonal carrier was compared with a numerical heterogeneous
first-arrival calculation using `scikit-fmm`. The FMM used the full 40 m
velocity grid, the same velocity model as the FD observations, and trilinear
interpolation at the 25 receivers. First arrivals were picked with the same 5%
peak and three-consecutive-sample rule used by the learned-interface audit.

| Independent event | FMM MAE (s) | FMM RMSE (s) | FMM 95th percentile (s) |
| --- | ---: | ---: | ---: |
| 01 | 0.209 | 0.283 | 0.598 |
| 02 | 0.157 | 0.205 | 0.437 |
| 03 | 0.305 | 0.402 | 0.683 |
| **Mean** | **0.224** | **0.296** | **0.573** |

The learned eikonal interface gave a mean first-arrival RMSE of `0.347 s` on
the same three events. Numerical FMM therefore improves the mean RMSE by about
14.6% relative to the learned eikonal approximation. The improvement supports
the hypothesis that the learned factorized carrier misses heterogeneous
curved-ray effects. FMM is a first-arrival baseline only; it does not reproduce
full waveform amplitudes, multipath interference, or FD boundary effects.

The raw audit is `outputs/fmm_independent_test.json`. The next experiment is a
FMM-based travel-time localization baseline using the same receiver geometry
and source bounds.
