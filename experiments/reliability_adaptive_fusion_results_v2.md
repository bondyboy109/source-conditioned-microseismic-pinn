# Reliability-adaptive fusion development audit

Post-hoc development audit on already inspected synthetic events. The adaptive rule is fixed for this audit but is not an independent blind confirmation. It reuses frozen FMM-seeded candidates and does not retrain or retune the waveform operator.

| Method | Mean eloc (m) | Median eloc (m) | <500 m |
|---|---:|---:|---:|
| waveform_only | 1406.9 | 1198.0 | 1/8 |
| fixed_hybrid | 437.3 | 397.6 | 4/8 |
| adaptive | 437.3 | 397.6 | 4/8 |
| pure_fmm | 244.9 | 240.9 | 8/8 |

| Event | Picks | Arrival RMSE (s) | Adaptive weight | Decision | Waveform (m) | Fixed hybrid (m) | Adaptive (m) | FMM (m) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| test_r04.npz | 9 | NA | 0.150 | adaptive_hybrid | 854.2 | 106.7 | 106.7 | 158.9 |
| test_r07.npz | 11 | NA | 0.217 | adaptive_hybrid | 1924.6 | 87.2 | 87.2 | 239.7 |
| test_r08.npz | 9 | NA | 0.150 | adaptive_hybrid | 3573.7 | 667.3 | 667.3 | 321.0 |
| test_center.npz | 10 | NA | 0.183 | adaptive_hybrid | 899.7 | 255.4 | 255.4 | 199.4 |
| test_west.npz | 9 | NA | 0.150 | adaptive_hybrid | 1534.8 | 539.8 | 539.8 | 275.4 |
| test_south.npz | 12 | NA | 0.250 | adaptive_hybrid | 71.8 | 71.8 | 71.8 | 242.1 |
| test_east.npz | 10 | NA | 0.183 | adaptive_hybrid | 922.0 | 634.2 | 634.2 | 163.7 |
| test_north.npz | 11 | NA | 0.217 | adaptive_hybrid | 1474.0 | 1136.1 | 1136.1 | 358.9 |

The rule was applied without source truth and without changing the frozen forward model or search.
