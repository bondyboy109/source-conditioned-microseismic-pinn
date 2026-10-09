# FMM arrival constraint on local FD joint fits

The two events, local grids, picker bias, and FMM fields have already been inspected or developed. This is an in-sample diagnostic, not an independent localization result. The amplitude is a free scalar, and the FMM and FD discretizations differ.

| Weight | Median eloc (m) | Median et (s) | Bound hits | Median waveform MSE | Median gain |
|---:|---:|---:|---:|---:|---:|
| 0 | 102.3 | 0.0020 | 0 | 0.3030 | 0.8195 |
| 0.1 | 102.3 | 0.0030 | 0 | 0.3049 | 0.8133 |
| 0.25 | 102.3 | 0.0030 | 0 | 0.3049 | 0.8133 |
| 1 | 124.2 | 0.0040 | 0 | 0.3257 | 0.7483 |

| Event | Weight | Estimate (x,y,z,t) | eloc (m) | et (s) | Bound | Waveform MSE | Arrival loss | Gain |
|---|---:|---|---:|---:|---|---:|---:|---:|
| spatial_01 | 0 | (3700.0, 8100.0, 916.667, 0.272) | 103.5 | 0.0020 | False | 0.1522 | 0.2281 | 0.8768 |
| spatial_01 | 0.1 | (3700.0, 8100.0, 916.667, 0.272) | 103.5 | 0.0020 | False | 0.1522 | 0.2281 | 0.8768 |
| spatial_01 | 0.25 | (3700.0, 8100.0, 916.667, 0.272) | 103.5 | 0.0020 | False | 0.1522 | 0.2281 | 0.8768 |
| spatial_01 | 1 | (3700.0, 8100.0, 916.667, 0.274) | 103.5 | 0.0040 | False | 0.1614 | 0.2179 | 0.8725 |
| spatial_02 | 0 | (12300.0, 6400.0, 395.0, 0.358) | 101.1 | 0.0020 | False | 0.4538 | 0.2354 | 0.7621 |
| spatial_02 | 0.1 | (12300.0, 6200.0, 395.0, 0.364) | 101.1 | 0.0040 | False | 0.4575 | 0.1941 | 0.7498 |
| spatial_02 | 0.25 | (12300.0, 6200.0, 395.0, 0.364) | 101.1 | 0.0040 | False | 0.4575 | 0.1941 | 0.7498 |
| spatial_02 | 1 | (12300.0, 6200.0, 275.0, 0.364) | 145.0 | 0.0040 | False | 0.4900 | 0.1310 | 0.6242 |

The spatial grid and source observations were already inspected. The 0-weight row fits time and a nonnegative scalar gain using FD waveforms; all other weights add corrected receiver picks from the frozen FMM fields. FD time shifting and the FMM arrival residual use the same trial origin time. The tabulated waveform MSE excludes the arrival penalty; total loss includes it.

On these two previously inspected and already well-localized events, arrival regularization did not reduce the location error. Weights 0.1 and 0.25 increased median origin-time error, and weight 1 worsened both errors and waveform fit. There were no bound hits even at zero weight, so this small local pilot does not test whether the penalty prevents the bound failures previously seen in the separate 75-node fresh-source check. Do not treat this weight sweep as a new independent confirmation or transfer its favorable local-grid errors to a global search.
