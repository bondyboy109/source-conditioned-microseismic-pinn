# Independent eight-event FMM arrival baseline

The FMM baseline used the velocity model decimated to 160 m, a `6 x 6 x 4`
coarse spatial grid, and a `3 x 3 x 3` local refinement around the best coarse
candidate. Origin time was estimated by the median picked-arrival minus FMM
traveltime. All 25 receiver traces produced a threshold pick in every event.

| Event | Location error (m) | Origin-time error (s) | Search time (s) |
|:--|--:|--:|--:|
| 01 | 528 | 0.015 | 26.5 |
| 02 | 955 | 0.167 | 26.4 |
| 03 | 1090 | 0.156 | 26.3 |
| 04 | 1166 | 0.185 | 26.6 |
| 05 | 1167 | 0.070 | 26.4 |
| 06 | 1020 | 0.129 | 26.5 |
| 07 | 1328 | 0.158 | 26.6 |
| 08 | 1526 | 0.090 | 26.5 |
| **Mean** | **1097** | **0.121** | **26.5** |

This is a useful physics-based reference, but the first-arrival picks use a
5%-of-peak threshold. The true-source FMM-versus-pick RMSE is about `0.121 s`
on event 01, so the pick uncertainty limits the interpretation of the baseline.
