# Source-depth failure diagnosis

This is a post-hoc summary of saved localization and noise records. It does not alter the locked method.

| Group | n | Median picks | Median abs depth error (m) | P90 abs depth error (m) | Median eloc (m) | Depth-boundary estimates |
|---|---:|---:|---:|---:|---:|---:|
| Overthrust hybrid | 7 | 10.0 | 178.5 | 828.5 | 394.5 | 0/7 |
| Layered hybrid | 4 | 7.5 | 55.4 | 303.7 | 163.8 | 1/4 |
| Layered FMM | 4 | 7.5 | 77.5 | 176.5 | 86.2 | 0/4 |

## Noise and pick counts

| SNR (dB) | Attempts | Hybrid available | Median picks | Pick-count range |
|---:|---:|---:|---:|---:|
| 20 | 24 | 18/24 | 8.0 | 4–14 |
| 10 | 24 | 15/24 | 7.0 | 4–11 |
| 5 | 24 | 8/24 | 5.0 | 3–9 |

## Event-level records

| Dataset | Event | Branch | Picks | True depth (m) | Estimated depth (m) | Δz (m) | eloc (m) | Boundary |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Overthrust confirmation | confirm_01_sw_shallow.npz | hybrid | 8 | 300.0 | 432.9 | 132.9 | 394.5 | no |
| Overthrust confirmation | confirm_02_se_deep.npz | hybrid | 7 | 1300.0 | 1121.5 | -178.5 | 471.3 | no |
| Overthrust confirmation | confirm_03_nw_deep.npz | hybrid | 8 | 1100.0 | 1015.9 | -84.1 | 226.9 | no |
| Overthrust confirmation | confirm_04_ne_shallow.npz | waveform_only | 4 | 350.0 | 100.0 | -250.0 | 14277.6 | no |
| Overthrust confirmation | confirm_05_west_deep.npz | hybrid | 14 | 1450.0 | 100.0 | -1350.0 | 1385.5 | no |
| Overthrust confirmation | confirm_06_east_shallow.npz | hybrid | 10 | 250.0 | 197.0 | -53.0 | 64.0 | no |
| Overthrust confirmation | confirm_07_north_mid.npz | hybrid | 15 | 800.0 | 480.0 | -320.0 | 394.5 | no |
| Overthrust confirmation | confirm_08_center_deep.npz | hybrid | 15 | 1250.0 | 769.2 | -480.8 | 606.0 | no |
| Layered post-hoc audit | layered_west_shallow | waveform_plus_fmm | 8 | 300.0 | 239.3 | -60.7 | 153.2 | no |
| Layered post-hoc audit | layered_west_shallow | fmm | 8 | 300.0 | 270.4 | -29.6 | 42.6 | no |
| Layered post-hoc audit | layered_center_deep | waveform_plus_fmm | 11 | 1300.0 | 1313.3 | 13.3 | 86.3 | no |
| Layered post-hoc audit | layered_center_deep | fmm | 11 | 1300.0 | 1258.0 | -42.0 | 51.4 | no |
| Layered post-hoc audit | layered_east_mid | waveform_plus_fmm | 7 | 800.0 | 392.2 | -407.8 | 420.9 | no |
| Layered post-hoc audit | layered_east_mid | fmm | 7 | 800.0 | 596.3 | -203.7 | 207.1 | no |
| Layered post-hoc audit | layered_north_deep | waveform_plus_fmm | 7 | 1450.0 | 1500.0 | 50.0 | 174.4 | yes |
| Layered post-hoc audit | layered_north_deep | fmm | 7 | 1450.0 | 1337.0 | -113.0 | 121.0 | no |

Saved experiments do not contain per-event per-receiver true arrival residuals or a receiver geometry rank/conditioning analysis. These summary associations cannot establish causation. Depth-boundary estimates and large depth errors motivate a development-only source-depth profile/observability audit; they do not justify changing the locked confirmation objective.
