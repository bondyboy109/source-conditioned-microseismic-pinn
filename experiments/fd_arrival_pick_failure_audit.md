# Receiver-pick and FMM failure audit

Truth is used explicitly in this failure audit. The three events were inspected before; these diagnostics cannot select a weight or validate a new locator.

A fixed -0.2628 s development bias is removed from raw picks (a negative bias increases corrected pick times). Residual is predicted onset minus corrected pick; RMSE uses only receivers passing the original picker filter.

| Event | Valid/25 | True arrivals <= picker cutoff | Median pick-inferred origin (s) | Truth origin (s) | True RMSE (s) | Waveform winner RMSE (s) | Hybrid winner RMSE (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh_west_mid | 12/25 | 11 | 0.221 | 0.220 | 0.039 | 0.392 | 0.243 |
| fresh_east_deep | 9/25 | 9 | 0.353 | 0.340 | 0.034 | 0.217 | 0.139 |
| fresh_center_shallow | 11/25 | 10 | 0.186 | 0.170 | 0.032 | 0.114 | 0.114 |

The JSON records every receiver's coordinates, peak fraction, raw/corrected pick, and signed residual at the true and both selected sources. Counts for predicted arrivals before the picker cutoff describe geometric visibility; they do not independently validate the picker or FMM.
