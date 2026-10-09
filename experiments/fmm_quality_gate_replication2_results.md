# Second locked FMM quality-gate replication

This replication was planned after the first confirmation was reviewed; source selection uses geometry and archive distance only, but it is a second locked test rather than a preregistration made before any confirmation results were seen.

Synthetic second source-held-out replication on the same Overthrust model and receiver geometry. The quality rule only abstains from selected FMM estimates. Noise realizations share each source waveform and are clustered by source location; results do not establish field-data or cross-geology performance.

Frozen rule: **accept FMM estimate only if max_azimuth_gap_deg <= 168.387367228**

| Metric | Always accept FMM | Quality gate |
|---|---:|---:|
| Eligible four-to-five-pick cases | 13 | 0 |
| Coverage | 1.000 | 0.000 |
| Success precision | 0.462 | NA |
| Successful estimates retained | 6 | 0 |
| Median location error (m) | 600.103 | NA |

## By SNR

| SNR | Eligible | Accepted | Successes / eligible | Accepted precision |
|---:|---:|---:|---:|---:|
| clean | 0 | 0 | 0/0 | NA |
| 1 | 0 | 0 | 0/0 | NA |
| 2 | 2 | 0 | 1/2 | NA |
| 3 | 11 | 0 | 5/11 | NA |

## By source

Noise conditions from one source are clustered and are not independent source replicates.

| Source | Eligible | Accepted | FMM successes | Accepted successes |
|---|---:|---:|---:|---:|
| qgr2_01_sw_shallow | 1 | 0 | 1 | 0 |
| qgr2_02_sw_deep | 3 | 0 | 3 | 0 |
| qgr2_03_se_shallow | 0 | 0 | 0 | 0 |
| qgr2_04_se_deep | 1 | 0 | 0 | 0 |
| qgr2_05_nw_shallow | 1 | 0 | 1 | 0 |
| qgr2_06_nw_deep | 1 | 0 | 1 | 0 |
| qgr2_07_ne_shallow | 3 | 0 | 0 | 0 |
| qgr2_08_ne_deep | 3 | 0 | 0 | 0 |

The gate abstains on selected estimates; it cannot change the underlying FMM estimate. Interpret conditional precision together with coverage, retained successes, and the number of source locations contributing accepted estimates.

Total observations evaluated: **80**.
