# Independent adaptive-fusion confirmation

Synthetic confirmation on the same Overthrust velocity model and receiver geometry. The eight source locations are spatially separated from all archived source truths, but this is not field-data or cross-geology validation. The 500 m success threshold is a prespecified operational metric, not a universal microseismic tolerance.

Success means a finite estimate with location error <= 500 m. Unavailable estimates count as failures in the all-condition success rate.

| SNR | Method | Available | <=500 m | Success rate | Median eloc (m) | Median et (s) |
|---|---|---:|---:|---:|---:|---:|
| clean | waveform_only | 8/8 | 0/8 | 0.000 | 1102.5 | 0.1000 |
| clean | fixed_hybrid | 8/8 | 8/8 | 1.000 | 160.8 | 0.0148 |
| clean | adaptive | 8/8 | 8/8 | 1.000 | 188.1 | 0.0114 |
| clean | pure_fmm | 8/8 | 8/8 | 1.000 | 165.5 | 0.0135 |
| 0 | waveform_only | 24/24 | 0/24 | 0.000 | 1875.7 | 0.0978 |
| 0 | fixed_hybrid | 0/24 | 0/24 | 0.000 | NA | NA |
| 0 | adaptive | 0/24 | 0/24 | 0.000 | NA | NA |
| 0 | pure_fmm | 0/24 | 0/24 | 0.000 | NA | NA |
| 10 | waveform_only | 24/24 | 0/24 | 0.000 | 1165.4 | 0.1300 |
| 10 | fixed_hybrid | 24/24 | 7/24 | 0.292 | 560.8 | 0.1974 |
| 10 | adaptive | 24/24 | 4/24 | 0.167 | 609.9 | 0.1984 |
| 10 | pure_fmm | 24/24 | 19/24 | 0.792 | 317.1 | 0.2200 |
| 20 | waveform_only | 24/24 | 0/24 | 0.000 | 1100.9 | 0.1001 |
| 20 | fixed_hybrid | 24/24 | 14/24 | 0.583 | 436.9 | 0.1765 |
| 20 | adaptive | 24/24 | 15/24 | 0.625 | 456.4 | 0.1713 |
| 20 | pure_fmm | 24/24 | 21/24 | 0.875 | 268.6 | 0.1839 |

## Pick-count gate coverage

| Condition | 0-3 picks | 4-5 picks | 6-11 picks | 12+ picks |
|---|---:|---:|---:|---:|
| Clean | 0/8 | 0/8 | 2/8 | 6/8 |
| 0 dB | 24/24 | 0/24 | 0/24 | 0/24 |
| 10 dB | 0/24 | 0/24 | 18/24 | 6/24 |
| 20 dB | 0/24 | 0/24 | 9/24 | 15/24 |

No observation produced 4-5 valid picks. At 0 dB, all 24 conditions had fewer than four picks, so the FMM fallback could not form an identifiable estimate. At 10 and 20 dB, all conditions had at least seven picks and the adaptive method used its hybrid branch. The experiment therefore does not establish a benefit specifically in the 4-5-pick fallback regime.

Adaptive-minus-comparator paired location-error changes are in metres; negative values favor adaptive fusion.

Overall conditions evaluated: **80**. 
Runtime total: **704.8 s**.

Evaluation script SHA-256: `fba979271eb126094209d229ddff098606ff147d1ac4c730b14bf435f7275ea5`.

The experiment is synthetic, uses one Overthrust model and one surface array, and cannot establish field-data or cross-geology performance.
