# Locked FMM quality-gate confirmation

Synthetic independent-source confirmation on one Overthrust model and one fixed receiver geometry. The gate can abstain on poor FMM estimates but cannot improve the underlying FMM estimates. Results do not establish field-data or cross-geology performance.

Frozen rule: **accept FMM estimate only if max_azimuth_gap_deg <= 168.387367228**

| Metric | Always accept FMM | Quality gate |
|---|---:|---:|
| Eligible four-to-five-pick cases | 15 | 5 |
| Coverage | 1.000 | 0.333 |
| Success precision among accepted | 0.200 | 0.600 |
| Successes retained | 3 | 3 |
| Median location error (m) | 1278.400 | 449.596 |

## By SNR

| SNR | Eligible | Accepted | Successes / eligible | Precision among accepted |
|---:|---:|---:|---:|---:|
| clean | 0 | 0 | 0/0 | NA |
| 1 | 2 | 1 | 1/2 | 1.000 |
| 2 | 5 | 3 | 1/5 | 0.333 |
| 3 | 8 | 1 | 1/8 | 1.000 |

## By source

The noise realizations are clustered within source locations, so the pooled conditions are not independent source replicates.

| Source | Eligible | Accepted | All-FMM successes | Accepted successes |
|---|---:|---:|---:|---:|
| qg_01_west_shallow | 3 | 1 | 1 | 1 |
| qg_02_north_mid | 1 | 1 | 1 | 1 |
| qg_03_south_mid | 2 | 0 | 0 | 0 |
| qg_04_east_mid | 4 | 3 | 1 | 1 |
| qg_05_center_deep | 1 | 0 | 0 | 0 |
| qg_06_southeast_deep | 1 | 0 | 0 | 0 |
| qg_07_southwest_deep | 3 | 0 | 0 | 0 |
| qg_08_northeast_mid | 0 | 0 | 0 | 0 |

## Interpretation

The gate only abstains from selected FMM outputs; it does not modify or improve the FMM source estimates. It retained all three successful estimates, but accepted outputs came from only three source locations and two of five accepted estimates from qg_04 exceeded 500 m. The apparent precision increase is therefore preliminary and should be interpreted with the 33% coverage and source-level clustering, not as evidence of robust generalization.

Total observations evaluated: **80**.
