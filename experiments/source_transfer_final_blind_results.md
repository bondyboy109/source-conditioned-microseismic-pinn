# Source-transfer final blind localization test

Twelve noise-free synthetic source locations on the same Overthrust velocity model and receiver layout. This tests new source positions, not new geology or field data. FMM uses first-arrival picks and the known heterogeneous velocity; learned models use full receiver traces plus the same FMM-arrival constraint.

Primary summary is computed per source by taking the median over three training seeds. Bootstrap intervals resample the 12 source locations, not the 36 seed-event rows.

| Method | Sources with estimate | Median source-level eloc (m) | Mean source-level eloc (m) | Sources <=500 m |
|---|---:|---:|---:|---:|
| source_to_target | 12 | 369.1 | 461.4 | 7/12 |
| target_to_target | 12 | 410.6 | 433.8 | 9/12 |
| target_only_1500 | 12 | 330.2 | 390.1 | 9/12 |
| FMM first-arrival baseline | 12 | 197.5 | 287.4 | 9/12 |

## Paired source-level differences

Negative error differences favor the first method.

| Comparison | Sources | Mean difference (m) | Median difference (m) | Source-bootstrap 95% CI for mean (m) |
|---|---:|---:|---:|---:|
| Source-to-target minus target-to-target | 12 | 27.7 | -8.0 | [-82.2, 141.1] |
| Source-to-target minus FMM | 12 | 174.0 | 131.1 | [-8.2, 368.4] |
| Target-to-target minus FMM | 12 | 146.4 | 251.5 | [6.4, 274.9] |

## Per-source results

| Source | FMM eloc (m) | Source-to-target median across seeds (m) | Target-to-target median across seeds (m) | Target-only median across seeds (m) |
|---|---:|---:|---:|---:|
| blind_01_sw_shallow | 152.1 | 127.0 | 411.4 | 191.2 |
| blind_02_sw_middle | 50.5 | 641.4 | 294.2 | 202.4 |
| blind_03_sw_deep | 131.4 | 844.2 | 409.8 | 611.0 |
| blind_04_nw_shallow | 843.9 | 813.4 | 781.2 | 718.4 |
| blind_05_nw_middle | 507.2 | 103.1 | 110.2 | 111.6 |
| blind_06_nw_deep | 609.9 | 1071.7 | 905.6 | 1088.9 |
| blind_07_se_shallow | 35.1 | 552.6 | 573.4 | 287.2 |
| blind_08_se_middle | 243.0 | 78.3 | 311.3 | 51.4 |
| blind_09_se_deep | 326.4 | 205.0 | 166.9 | 463.3 |
| blind_10_ne_shallow | 131.4 | 362.2 | 420.6 | 373.3 |
| blind_11_ne_middle | 85.9 | 374.7 | 383.6 | 134.1 |
| blind_12_ne_deep | 332.2 | 363.5 | 437.1 | 448.5 |

## Interpretation

The registered source-transfer criterion did not pass. Do not claim that source-domain pretraining plus transfer improves localization on this test.

FMM uses picked first arrivals and the known heterogeneous target velocity. The learned arms use full traces plus the same frozen FMM-arrival term. This is an operational comparison with different information inputs, not a same-input algorithm contest. All records are noise-free synthetic 5-Hz Overthrust data with one receiver geometry; they do not establish field or cross-geology performance.

Protocol SHA-256: `3696b99ef9849df9baaafa91811ab4fe7d5ee065ae1f453ebd8bd726f4d25352`. Observation count: 12; independent source count: 12; model seeds: [4101, 4102, 4103].
