# FMM first-arrival localization baseline

This baseline uses numerical fast marching on the heterogeneous Overthrust
velocity model rather than the learned wavefield operator. For each spatial
candidate, the script solves the first-arrival Eikonal equation and evaluates
the travel time at all receivers. Origin time is estimated as the median of
`observed_pick - computed_traveltime`, clipped to `[0.05, 0.50] s`.

The velocity grid is decimated by four for the coarse search (`160 m` cell
spacing). The spatial candidate grid is `8 x 8 x 5` over
`x,y=2000--14000 m` and `z=100--1500 m`. This is a baseline, not the final
high-resolution inversion.

| Independent event | `eloc` (m) | `et` (s) | Arrival RMSE (s) | Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| 01 | 959 | 0.110 | 0.218 | 87.3 |
| 02 | 515 | 0.159 | 0.158 | 93.6 |
| 03 | 1483 | 0.257 | 0.243 | 88.1 |
| **Mean** | **986** | **0.175** | **0.206** | **89.7** |

The FMM spatial error is lower than the source-conditioned PINN diagnostic
mean (`1447 m`), supporting the use of numerical heterogeneous traveltimes as
a stronger localization baseline. Origin-time errors remain large because the
coarse spatial grid, limited 0--2 s windows, and first-arrival picks produce a
poorly conditioned origin-time estimate. A denser search and robust arrival
pick weighting are required before using this as a final benchmark.

Raw output: `outputs/fmm_independent_inversion.json`.

## Local-refinement control

Independent test 02 was rerun by adding a `3 x 3 x 3` spatial FMM search
around the coarse winner. The estimate and errors were unchanged (`eloc=515 m`,
`et=0.159 s`). This indicates that the dominant error is not simply the
coarse spatial grid; the first-arrival picking and limited event window also
affect the origin-time and source trade-off.
