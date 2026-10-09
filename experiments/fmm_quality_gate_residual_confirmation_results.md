# Arrival-residual FMM gate: locked source-held-out follow-up

The residual feature was a predeclared development candidate, selected for this follow-up after the azimuth feature failed independent generalization. No prior holdout observations are used to fit this threshold.

Synthetic source-held-out confirmation on the same Overthrust model and receiver geometry. The rule only accepts or abstains from FMM outputs; it cannot improve the underlying estimate. Observations share source waveforms across noise conditions and are clustered by source. The new residual candidate was selected after prior holdouts were reviewed, but its threshold uses development data only.

Frozen rule: **accept FMM estimate only if arrival_rmse_s <= 0.09508478690085563**

| Metric | Always accept FMM | Residual gate |
|---|---:|---:|
| Eligible cases | 31 | 11 |
| Coverage | 1.000 | 0.355 |
| Success precision | 0.290 | 0.364 |
| Successful estimates retained | 9 | 4 |
| Median location error (m) | 947.207 | 947.207 |

## By SNR

| SNR | Eligible | Accepted | FMM successes / eligible | Accepted precision |
|---:|---:|---:|---:|---:|
| clean | 1 | 1 | 1/1 | 1.000 |
| 1 | 1 | 0 | 1/1 | NA |
| 2 | 11 | 4 | 3/11 | 0.000 |
| 3 | 18 | 6 | 4/18 | 0.500 |

## By source

Noise conditions are repeated measurements nested within each source; the independent source count is 12.

| Source | Eligible conditions | Accepted | FMM successes | Accepted successes |
|---|---:|---:|---:|---:|
| r3_01_sw_shallow | 4 | 1 | 2 | 1 |
| r3_02_sw_mid | 2 | 2 | 2 | 2 |
| r3_03_sw_deep | 3 | 3 | 0 | 0 |
| r3_04_se_shallow | 3 | 0 | 2 | 0 |
| r3_05_se_mid | 4 | 1 | 0 | 0 |
| r3_06_se_deep | 3 | 3 | 0 | 0 |
| r3_07_nw_shallow | 2 | 0 | 2 | 0 |
| r3_08_nw_mid | 1 | 0 | 0 | 0 |
| r3_09_nw_deep | 3 | 0 | 0 | 0 |
| r3_10_ne_shallow | 3 | 0 | 0 | 0 |
| r3_11_ne_mid | 2 | 0 | 0 | 0 |
| r3_12_ne_deep | 1 | 1 | 1 | 1 |

The threshold was fitted on development data only. This follow-up tests a predeclared alternative feature after the first gate failed on prior holdouts, so it is independent of threshold fitting but not a pristine first-choice preregistration. The gate only abstains; it never changes the FMM estimate. Interpret precision together with coverage and retained successes.

Total observations evaluated: **120**; independent source locations: **12**.
