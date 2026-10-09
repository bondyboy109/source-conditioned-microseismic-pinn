# FD true-source forward consistency audit

Post-hoc diagnostic on the eight frozen records. The truths were not used in either global FD candidate grid or event ranking.

| Event | Direct truth relative L2 | Shift at truth time relative L2 | Best shift relative L2 | Best shift origin error (s) |
|---|---:|---:|---:|---:|
| confirm_01_west_shallow | 2.539e-06 | 1.178e-03 | 1.178e-03 | 7.153e-09 |
| confirm_02_east_deep | 1.637e-06 | 1.520e-06 | 1.520e-06 | 1.431e-08 |
| confirm_03_north_mid | 2.241e-06 | 4.026e-06 | 4.026e-06 | 5.364e-09 |
| confirm_04_south_deep | 1.561e-06 | 1.411e-06 | 1.411e-06 | 1.311e-08 |
| confirm_05_center_shallow | 1.718e-06 | 1.948e-02 | 1.948e-02 | 5.960e-10 |
| confirm_06_southwest_deep | 1.710e-06 | 1.623e-06 | 1.623e-06 | 2.384e-09 |
| confirm_07_northeast_mid | 2.213e-06 | 1.924e-06 | 1.924e-06 | 1.073e-08 |
| confirm_08_north_deep | 1.848e-06 | 1.855e-06 | 1.855e-06 | 1.431e-08 |

This tests solver agreement and the approximate discrete time-shift interface at known locations. It is not an estimate of location accuracy or an independent FD inverse benchmark.
