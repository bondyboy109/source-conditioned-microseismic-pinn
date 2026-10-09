# FD-pair objective-ordering audit

The selected 0.05 branch and its paired restarted control score the same truth and previously recovered wrong-source candidates. Positive gaps favor truth.

| Branch | Wrong source wins waveform | Wrong source wins hybrid | Mean wrong-minus-truth waveform | Mean wrong-minus-truth hybrid |
|---|---:|---:|---:|---:|
| restarted_control | 9/12 | 9/12 | -0.1115 | -0.0275 |
| fd_pair_weight_0.05 | 9/12 | 9/12 | -0.1055 | -0.0215 |

This reuses four previously inspected development events and their existing recovered estimates. It diagnoses objective ordering only, does not rerun localization, and is not an independent test.
