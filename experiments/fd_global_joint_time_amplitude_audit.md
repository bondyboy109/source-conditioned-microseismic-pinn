# Post-hoc global FD joint time/amplitude audit

Same 75 grid points and eight confirmation records as the registered FD unit-amplitude baseline. Each candidate jointly optimizes a discrete origin-time shift and a nonnegative scalar source gain. Source truth is used only to calculate errors after ranking.

| Event | Unit eloc (m) | Joint eloc (m) | Unit et (s) | Joint et (s) | Unit MSE | Joint MSE | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| confirm_01_west_shallow | 2332.0 | 2332.0 | 0.218 | 0.218 | 0.9161 | 0.8793 | 0.6444 |
| confirm_02_east_deep | 10753.1 | 2007.2 | 0.140 | 0.092 | 1.5813 | 0.9825 | 0.1169 |
| confirm_03_north_mid | 2215.0 | 1364.6 | 0.236 | 0.210 | 1.3868 | 0.6866 | 0.2584 |
| confirm_04_south_deep | 800.1 | 800.1 | 0.062 | 0.062 | 0.6142 | 0.5849 | 0.7903 |
| confirm_05_center_shallow | 670.8 | 489.9 | 0.072 | 0.136 | 0.3323 | 0.2244 | 0.1294 |
| confirm_06_southwest_deep | 1790.6 | 1614.4 | 0.182 | 0.170 | 1.7175 | 0.8958 | 0.2552 |
| confirm_07_northeast_mid | 1241.0 | 877.5 | 0.126 | 0.066 | 0.5648 | 0.3870 | 0.3992 |
| confirm_08_north_deep | 4577.9 | 1719.8 | 0.056 | 0.000 | 1.7159 | 0.9710 | 0.1394 |

Joint median eloc = 1489.5 m; median et = 0.114 s; median relative L2 = 0.8832.
Templates: 777.9 s; median per-event ranking: 0.741 s.

This objective differs from the original baseline. Lower waveform MSE does not guarantee lower location error; tiny fitted gains can favor nearly silent predictions. This post-hoc change does not establish an accuracy-matched comparison with a neural workflow.
