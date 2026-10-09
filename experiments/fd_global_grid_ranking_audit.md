# Post-hoc FD global-grid ranking audit

The nearest grid node is selected using the true source only for this diagnostic. It was never supplied to the global FD search.

| Event | Nearest-node error (m) | Winner error (m) | Nearest MSE | Winner MSE | Nearest selected? |
|---|---:|---:|---:|---:|:---:|
| confirm_01_west_shallow | 1395.0 | 2332.0 | 125.8771 | 0.9161 | no |
| confirm_02_east_deep | 1851.7 | 10753.1 | 2.1099 | 1.5813 | no |
| confirm_03_north_mid | 1364.6 | 2215.0 | 3.2675 | 1.3868 | no |
| confirm_04_south_deep | 666.4 | 800.1 | 3.2783 | 0.6142 | no |
| confirm_05_center_shallow | 489.9 | 670.8 | 35.3482 | 0.3323 | no |
| confirm_06_southwest_deep | 1614.4 | 1790.6 | 1.7829 | 1.7175 | no |
| confirm_07_northeast_mid | 877.5 | 1241.0 | 1.7753 | 0.5648 | no |
| confirm_08_north_deep | 1719.8 | 4577.9 | 2.0766 | 1.7159 | no |

0 of 8 winning nodes are nearest by distance. The nearest-node error is an oracle grid floor, not an achievable blind-search result. Both columns optimize the same discrete origin-time shift, and scores use the unchanged normalized waveform MSE.

Median nearest-node distance: 1379.8 m; median selected-node distance: 2002.8 m. The recomputed winning scores match the frozen baseline. In some events the nearest node lies on the 100 m depth plane and has a much larger waveform loss. This suggests examining depth sampling and boundary sensitivity, but this audit does not identify a unique cause or validate a finer inverse.
