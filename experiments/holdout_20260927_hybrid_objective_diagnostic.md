# Waveform plus FMM-arrival objective diagnostic

This diagnostic evaluated the frozen joint MoE on holdout event 01 over a
`6 x 6 x 4` spatial source grid. For each spatial candidate, origin time was
estimated from the median difference between picked receiver arrivals and FMM
traveltimes. The waveform loss and arrival residual were then compared without
retraining the network.

| Objective | Best candidate `(x,y,z,t)` | Location error | Arrival RMSE | Waveform loss |
|:--|:--|--:|--:|--:|
| Waveform only | `(2000, 14000, 567, 0.050)` | 7226 m | 1.015 s | 1.069 |
| FMM arrival only | `(4400, 6800, 100, 0.195)` | 528 m | 0.150 s | 1.191 |
| Hybrid, alpha=0.25 | `(4400, 6800, 100, 0.195)` | 528 m | 0.150 s | 1.191 |
| Hybrid, alpha=1.0 | `(4400, 6800, 100, 0.195)` | 528 m | 0.150 s | 1.191 |

The true source is `(4200, 7200, 380, 0.18)`. Adding the FMM arrival term
therefore moves the selected coarse candidate much closer to the true source
than the waveform-only objective. However, the true-source FMM arrival RMSE
against the 5%-of-peak threshold picks is `0.121 s`, which is large relative to
the origin-time error. These threshold picks are not independent physical
first-arrival labels and should not be treated as ground truth.

The result justifies a controlled next experiment: evaluate the hybrid
objective on all eight independent holdouts using a fixed FMM grid, fixed pick
rule, and predeclared weighting. It does not yet establish publication-level
localization accuracy or validate calibrated uncertainty.
