# Fresh-source FD objective check

Three newly generated 5-Hz source positions; frozen 75-node grid, waveform window and origin-time bounds. Both objectives use identical templates and observations. Unit fits time only; joint fits time and one nonnegative amplitude per candidate.

| Event | Grid floor (m) | Unit eloc (m) | Joint eloc (m) | Unit et (s) | Joint et (s) | Unit MSE | Joint MSE | Joint gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh_west_mid | 1700.4 | 1869.1 | 1700.4 | 0.102 | 0.280 | 1.5988 | 0.7399 | 0.2226 |
| fresh_east_deep | 852.9 | 852.9 | 877.2 | 0.018 | 0.160 | 0.7692 | 0.6268 | 0.3094 |
| fresh_center_shallow | 447.3 | 571.1 | 447.3 | 0.096 | 0.104 | 0.4620 | 0.3002 | 0.1454 |

| Median | - | 852.9 | 877.2 | 0.096 | 0.160 | - | - | - |

FD template synthesis: 778.3 s; median unit/joint ranking: 0.598/0.756 s.

The prespecified median spatial-error comparison does not support an overall
improvement: joint fitting improves two of three individual locations but
raises the median from 852.9 to 877.2 m. All three origin-time errors increase
(median 0.096 to 0.160 s). The west-mid and east-deep joint estimates hit the
0.5 s upper time bound, indicating a time/depth/gain trade-off despite lower
waveform MSE. The grid floor is a spatial lower bound for this fixed candidate
set, not a measure of origin-time accuracy. Three events and one synthetic
geology are insufficient to claim a robust gain.

The three source positions are new but the velocity model, source pulse, receiver array, solver and search grid are shared with development. The joint objective was selected after inspecting the earlier eight events. This is prospective only with respect to source positions, not an independent geological or field test.
