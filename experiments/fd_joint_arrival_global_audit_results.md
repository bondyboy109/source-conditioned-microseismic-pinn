# Post-hoc FMM arrival constraint on 75-node FD search

These three source records were previously inspected, including two boundary failures. The weight grid was fixed after the local development audit. This is a post-hoc diagnosis, not a second independent test; tuning or choosing a favorable weight from these events requires a genuinely new prospective evaluation. The 75-node grid is coarse, and FMM picks use a development-calibrated bias.

| Weight | Median eloc (m) | Median et (s) | Time-bound hits | Median waveform MSE |
|---:|---:|---:|---:|---:|
| 0 | 877.2 | 0.160 | 2 | 0.6268 |
| 0.1 | 852.9 | 0.104 | 1 | 0.6735 |
| 0.25 | 852.9 | 0.104 | 1 | 0.6735 |
| 1 | 852.9 | 0.098 | 1 | 0.6824 |

| Event | Weight | Estimate (x,y,z,t) | eloc (m) | et (s) | Bound | Waveform MSE | Arrival loss | Gain |
|---|---:|---|---:|---:|---|---:|---:|---:|
| fresh_west_mid | 0 | (5000.0, 11000.0, 800.0, 0.5) | 1700.4 | 0.280 | True | 0.7399 | 15.3993 | 0.2226 |
| fresh_west_mid | 0.1 | (2000.0, 11000.0, 1500.0, 0.05) | 2234.6 | 0.170 | True | 0.9999 | 5.9079 | 0.0083 |
| fresh_west_mid | 0.25 | (2000.0, 11000.0, 1500.0, 0.05) | 2234.6 | 0.170 | True | 0.9999 | 5.9079 | 0.0083 |
| fresh_west_mid | 1 | (2000.0, 11000.0, 1500.0, 0.05) | 2234.6 | 0.170 | True | 0.9999 | 5.9079 | 0.0083 |
| fresh_east_deep | 0 | (11000.0, 5000.0, 800.0, 0.5) | 877.2 | 0.160 | True | 0.6268 | 4.6984 | 0.3094 |
| fresh_east_deep | 0.1 | (11000.0, 5000.0, 1500.0, 0.322) | 852.9 | 0.018 | False | 0.6735 | 1.9288 | 0.6488 |
| fresh_east_deep | 0.25 | (11000.0, 5000.0, 1500.0, 0.322) | 852.9 | 0.018 | False | 0.6735 | 1.9288 | 0.6488 |
| fresh_east_deep | 1 | (11000.0, 5000.0, 1500.0, 0.326) | 852.9 | 0.014 | False | 0.6824 | 1.9106 | 0.6406 |
| fresh_center_shallow | 0 | (8000.0, 11000.0, 100.0, 0.274) | 447.3 | 0.104 | False | 0.3002 | 1.2930 | 0.1454 |
| fresh_center_shallow | 0.1 | (8000.0, 11000.0, 100.0, 0.274) | 447.3 | 0.104 | False | 0.3002 | 1.2930 | 0.1454 |
| fresh_center_shallow | 0.25 | (8000.0, 11000.0, 100.0, 0.274) | 447.3 | 0.104 | False | 0.3002 | 1.2930 | 0.1454 |
| fresh_center_shallow | 1 | (8000.0, 11000.0, 100.0, 0.268) | 447.3 | 0.098 | False | 0.3462 | 1.2059 | 0.1405 |

Each score evaluates the same FD template, discrete time shifts, and analytic nonnegative gain, adding alpha times mean squared corrected FMM pick residual divided by 0.10 s. The alpha=0 winner is checked against the earlier saved result. Waveform MSE is reported separately from the penalized total. The 75 shared FD templates took 777.7 s to synthesize.

At weights 0.1 and 0.25, the east-deep event moves off the time bound and its time error decreases, while the west-mid event remains on a bound and its location error increases. Its fitted gain of 0.0083 produces an almost-zero waveform (normalized MSE about 1), exposing a degenerate solution that arrival regularization alone does not prevent. The third event keeps the same location. The lower three-event median is therefore not evidence of a robust localization improvement. The 75-node grid, source-waveform scale model, pick validity, and FMM/FD mismatch need further diagnosis before freezing a new method.
