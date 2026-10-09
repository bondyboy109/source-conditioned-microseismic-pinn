# Cached higher-resolution global FD waveform confirmation baseline

The FD branch searches the full registered source bounds on a fixed 5 x 5 x 3 spatial grid and optimizes origin time by discrete waveform shifting. It does not use FMM estimates. The 75 templates are generated once and reused for all events.

| Event | Grid floor (m) | FD location error (m) | FD origin error (s) | Relative L2 | Ranking (s) |
|---|---:|---:|---:|---:|---:|
| confirm_01_west_shallow | 1395.0 | 2332.0 | 0.2180 | 0.9571 | 0.5344 |
| confirm_02_east_deep | 1851.7 | 10753.1 | 0.1400 | 1.2575 | 0.5573 |
| confirm_03_north_mid | 1364.6 | 2215.0 | 0.2360 | 1.1776 | 0.5487 |
| confirm_04_south_deep | 666.4 | 800.1 | 0.0620 | 0.7837 | 0.5406 |
| confirm_05_center_shallow | 489.9 | 670.8 | 0.0720 | 0.5764 | 0.5432 |
| confirm_06_southwest_deep | 1614.4 | 1790.6 | 0.1820 | 1.3105 | 0.5408 |
| confirm_07_northeast_mid | 877.5 | 1241.0 | 0.1260 | 0.7515 | 0.5597 |
| confirm_08_north_deep | 1719.8 | 4577.9 | 0.0560 | 1.3099 | 0.5495 |
| **Median** | **1379.8** | **2002.8** | **0.1330** | **1.0674** | **0.5459** |

Template synthesis took 821.8 s for 75 candidates and reached 4817.5 MiB peak CUDA allocation. The templates were reused for all events; ranking time excludes FD synthesis.

This is a cached higher-resolution global FD pilot. It searches the full source bounds without FMM initialization on a 75-point spatial grid, but the grid remains coarser than a continuous global inverse and the waveform objective uses discrete origin-time shifts. The cached-template runtime and per-event ranking time are reported separately.
