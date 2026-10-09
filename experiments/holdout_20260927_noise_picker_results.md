# Noise sensitivity and arrival-picker comparison

Four independent holdout events were corrupted with Gaussian noise at
`SNR=20, 10, 5 dB`, with two deterministic realizations per SNR. The same
160-m FMM grid and `6 x 6 x 4` source candidates were used for both pickers.

| Picker | SNR | Mean location error (m) | Median location error (m) | Mean origin-time error (s) | Failures |
|:--|--:|--:|--:|--:|--:|
| 5%-of-peak threshold | 20 dB | 3130 | 3176 | 0.131 | 0/8 |
| Robust MAD trigger | 20 dB | **1128** | 1230 | **0.054** | 0/8 |
| 5%-of-peak threshold | 10 dB | 7171 | 7710 | 0.245 | 0/8 |
| Robust MAD trigger | 10 dB | **1556** | 1440 | **0.176** | 0/8 |
| 5%-of-peak threshold | 5 dB | 6604 | 6911 | 0.245 | 0/8 |
| Robust MAD trigger | 5 dB | **2903** | 2622 | **0.161** | 0/8 |

The robust picker estimates pre-event noise using the median absolute deviation,
smooths the absolute trace, and requires a sustained threshold crossing. It
substantially reduces the noise-induced failure of the original threshold rule,
especially at 20 and 10 dB SNR. The experiment uses synthetic Gaussian noise,
four events, and two realizations per SNR; it is a method-development result,
not a final uncertainty or field-data validation.
