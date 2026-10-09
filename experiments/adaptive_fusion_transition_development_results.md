# Adaptive-fusion transition SNR calibration

This SNR calibration uses events already seen during development and is not independent validation. It selects only noise levels for a separate locked-source experiment; it does not select a model, picker, fusion weight, or source-search setting.

| SNR (dB) | Trials | Median valid picks | Trials with 4-5 picks |
|---:|---:|---:|---:|
| 1 | 24 | 1.5 | 4.2% |
| 2 | 24 | 3.0 | 41.7% |
| 3 | 24 | 4.5 | 25.0% |
| 4 | 24 | 6.0 | 29.2% |
| 5 | 24 | 7.0 | 20.8% |
| 6 | 24 | 7.5 | 20.8% |
| 7 | 24 | 8.0 | 25.0% |
| 8 | 24 | 9.0 | 25.0% |
| 9 | 24 | 9.0 | 20.8% |

Selected center SNR: **2 dB**.
Locked confirmation SNRs: **[1, 2, 3] dB**.

These values were selected from pick counts only; no localization error was inspected.
