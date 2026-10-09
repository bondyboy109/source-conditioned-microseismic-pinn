# Independent adaptive-fusion confirmation

Synthetic confirmation on the same Overthrust velocity model and receiver geometry. Source locations are separated from all archived source truths, but this is not field-data or cross-geology validation. The targeted SNR levels were selected using pick counts on prior development events only. The 500 m threshold is an operational metric, not a universal microseismic tolerance.

Success means a finite estimate with location error <= 500 m. Unavailable estimates count as failures in the all-condition success rate.

| SNR | Method | Available | <=500 m | Success rate | Median eloc (m) | Median et (s) |
|---|---|---:|---:|---:|---:|---:|
| clean | waveform_only | 8/8 | 1/8 | 0.125 | 1134.8 | 0.1247 |
| clean | fixed_hybrid | 8/8 | 6/8 | 0.750 | 126.2 | 0.0066 |
| clean | adaptive | 8/8 | 5/8 | 0.625 | 278.0 | 0.0089 |
| clean | pure_fmm | 8/8 | 8/8 | 1.000 | 143.5 | 0.0121 |
| 1 | waveform_only | 24/24 | 3/24 | 0.125 | 1197.3 | 0.0700 |
| 1 | fixed_hybrid | 0/24 | 0/24 | 0.000 | NA | NA |
| 1 | adaptive | 2/24 | 1/24 | 0.042 | 1116.6 | 0.1300 |
| 1 | pure_fmm | 2/24 | 1/24 | 0.042 | 1116.6 | 0.1300 |
| 2 | waveform_only | 24/24 | 3/24 | 0.125 | 1188.6 | 0.1112 |
| 2 | fixed_hybrid | 1/24 | 0/24 | 0.000 | 2229.6 | 0.0587 |
| 2 | adaptive | 8/24 | 3/24 | 0.125 | 1279.8 | 0.1900 |
| 2 | pure_fmm | 8/24 | 3/24 | 0.125 | 1279.8 | 0.1900 |
| 3 | waveform_only | 24/24 | 3/24 | 0.125 | 1185.6 | 0.1160 |
| 3 | fixed_hybrid | 4/24 | 0/24 | 0.000 | 1454.0 | 0.1500 |
| 3 | adaptive | 15/24 | 2/24 | 0.083 | 1434.3 | 0.1900 |
| 3 | pure_fmm | 15/24 | 4/24 | 0.167 | 1456.0 | 0.1900 |

Adaptive-minus-comparator paired location-error changes are in metres; negative values favor adaptive fusion.

Overall conditions evaluated: **80**. 
Runtime total: **452.4 s**.

Confirmation script SHA-256: `86e9011eb3d3177fd7f97e55192090ebcb8d478c10f09e57f3ee6e3573b60ca3`.

The experiment is synthetic, uses one Overthrust model and one surface array, and cannot establish field-data or cross-geology performance.

## Pick-count gate coverage

| Condition | 0-3 picks | 4-5 picks | 6-11 picks | 12+ picks |
|---|---:|---:|---:|---:|
| clean | 0/8 | 0/8 | 4/8 | 4/8 |
| 1 | 22/24 | 2/24 | 0/24 | 0/24 |
| 2 | 16/24 | 7/24 | 1/24 | 0/24 |
| 3 | 9/24 | 11/24 | 4/24 | 0/24 |

The 4-5-pick subset is the prespecified fallback transition band. Adaptive estimates in this band are identical to the pure-FMM branch by design; the relevant comparison is whether those FMM estimates are available and accurate while fixed hybrid abstains.

## Transition-band Results

| SNR (dB) | Conditions with 4-5 picks | Adaptive/FMM available | Within 500 m | Median eloc (m) | Fixed hybrid available |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2/2 | 1/2 | 1116.6 | 0/2 |
| 2 | 7 | 7/7 | 3/7 | 941.5 | 0/7 |
| 3 | 11 | 11/11 | 2/11 | 1456.0 | 0/11 |
| **All** | **20** | **20/20** | **6/20** | **1445.1** | **0/20** |

For these 20 conditions, the adaptive branch and pure FMM returned exactly the same source estimates, as specified by the fallback rule. The fallback therefore improves estimate availability relative to the fixed hybrid gate, but only 30% of estimates met the 500 m criterion; it provides no accuracy improvement over pure FMM.
