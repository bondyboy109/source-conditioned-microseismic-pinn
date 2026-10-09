# Post-hoc FD shallow-source depth profiles

Two previously inspected events. The fixed x/y location is the nearest horizontal node of the original 75-point FD grid. Known truth is used only to choose the diagnostic events and interpret errors. The FD boundary, receivers, velocity, source and discrete origin-time optimization are unchanged.

| Event | Depth (m) | Location error (m) | Normalized MSE | Best origin (s) | Predicted/observed energy |
|---|---:|---:|---:|---:|---:|
| confirm_01_west_shallow | 100 | 1395.0 | 125.8771 | 0.050 | 124.8731 |
| confirm_01_west_shallow | 250 | 1369.5 | 23.5327 | 0.050 | 22.5915 |
| confirm_01_west_shallow | 400 | 1360.2 | 6.0520 | 0.498 | 6.2154 |
| confirm_01_west_shallow | 550 | 1367.3 | 3.9324 | 0.454 | 3.9070 |
| confirm_01_west_shallow | 800 | 1415.0 | 3.2985 | 0.366 | 2.8520 |
| confirm_01_west_shallow | 1100 | 1525.2 | 1.5593 | 0.280 | 0.8322 |
| confirm_01_west_shallow | 1500 | 1743.0 | 1.3191 | 0.182 | 0.4963 |
| confirm_05_center_shallow | 100 | 489.9 | 35.3482 | 0.276 | 46.3379 |
| confirm_05_center_shallow | 250 | 450.0 | 7.2897 | 0.242 | 12.7037 |
| confirm_05_center_shallow | 400 | 458.3 | 0.6949 | 0.196 | 2.5392 |
| confirm_05_center_shallow | 550 | 512.3 | 0.3280 | 0.150 | 1.1158 |
| confirm_05_center_shallow | 800 | 670.8 | 0.3323 | 0.068 | 0.8762 |
| confirm_05_center_shallow | 1100 | 916.5 | 1.1332 | 0.146 | 0.3344 |
| confirm_05_center_shallow | 1500 | 1280.6 | 1.0803 | 0.500 | 0.1205 |

The profile does not isolate the effect of the top boundary; doing so would require a matched observation and forward model with a different top-boundary condition. At 100 m, the shifted prediction energies are about 125 and 46 times the observed energy for events 01 and 05, respectively. The minimum losses along these fixed x/y profiles occur at 1500 and 550 m, not at the true 410 and 300 m depths. This does not identify a top-boundary effect: coarse horizontal position, source-to-receiver distance and origin-time trade-offs also change the fit. No row is a blind localization estimate.
