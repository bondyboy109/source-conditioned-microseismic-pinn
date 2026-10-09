# Coarse global FD waveform confirmation baseline

The FD branch searches the full registered source bounds on a fixed 3 x 3 x 2 spatial grid and optimizes origin time by discrete waveform shifting. It does not use FMM estimates. The spatial resolution is coarse and is reported explicitly as a baseline limitation.

| Event | FD location error (m) | FD origin error (s) | FD waveform relative L2 | Candidates | Runtime (s) | Peak CUDA MiB |
|---|---:|---:|---:|---:|---:|---:|
| confirm_01_west_shallow | 5425.7 | 0.2260 | 1.1332 | 18 | 189.6 | 4817.4 |
| confirm_02_east_deep | 4757.0 | 0.0920 | 1.3587 | 18 | 189.6 | 4817.4 |
| confirm_03_north_mid | 10821.6 | 0.2600 | 1.4738 | 18 | 189.5 | 4817.4 |
| confirm_04_south_deep | 2905.2 | 0.0220 | 1.2251 | 18 | 189.6 | 4817.4 |
| confirm_05_center_shallow | 1280.6 | 0.3600 | 1.0394 | 18 | 189.5 | 4817.4 |
| confirm_06_southwest_deep | 2721.5 | 0.1900 | 1.5619 | 18 | 189.6 | 4817.4 |
| confirm_07_northeast_mid | 15288.6 | 0.2300 | 1.1801 | 18 | 189.9 | 4817.4 |
| confirm_08_north_deep | 1719.8 | 0.0000 | 1.4410 | 18 | 189.7 | 4817.4 |
| **Median** | **3831.1** | **0.2080** | **1.2919** | -- | **189.6** | -- |

The eight repeated template syntheses totaled 1516.9 s. The template library is identical for all events and could instead be shared; one synthesis took about 189.6 s. Ranking time for a cached library was not measured, so these timings must not be used as a refined FD inversion cost or an accuracy-matched break-even comparison. Peak CUDA allocation was 4817.4 MiB.

The distance from each true source to its nearest grid node was 1395.0, 1851.7, 2966.2, 2905.2, 489.9, 2721.5, 2276.0, and 1719.8 m, respectively (median 2063.8 m). The waveform objective selected a non-nearest node in five events. This is a low-resolution global FD cost and objective pilot; it is not a high-resolution global FD optimum.
