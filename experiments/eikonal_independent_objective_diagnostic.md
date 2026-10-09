# Post-hoc audit of failed eikonal-carrier localization

This diagnosis uses the eight events **already spent** by `eikonal_independent_protocol.json`. It is not a new independent test and does not modify a checkpoint or a registered search. Ground truth is used here to evaluate the frozen loss surface after localization. Each loss is the same receiver/time-stride-4, full-record-RMS-normalized waveform MSE used by the original local refinement.

| Event | FMM error (m) | Hybrid error (m) | Loss at true source/time | Best loss at FMM location* | Best loss at true location* | Loss at returned source | Lowest point on FMM-to-truth line |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 01 | 366.5 | 390.8 | 0.8885 | 1.2565 | 0.8587 | 0.5836 | 60% toward truth |
| 02 | 512.3 | 605.5 | 0.8569 | 1.0137 | 0.8569 | 0.7441 | truth |
| 03 | 193.6 | 952.2 | 1.1805 | 1.1314 | 1.1255 | 0.7380 | truth |
| 04 | 220.9 | 277.5 | 1.2243 | 1.4883 | 1.1567 | 0.9104 | truth |
| 05 | 195.3 | 152.9 | 0.7000 | 1.0555 | 0.6681 | 0.6146 | truth |
| 06 | 174.4 | 12.9 | 0.3709 | 0.6028 | 0.3709 | 0.3580 | truth |
| 07 | 249.8 | 628.5 | 1.0284 | 0.8280 | 1.0271 | 0.5168 | FMM |
| 08 | 490.4 | 173.5 | 0.5758 | 1.0446 | 0.5758 | 0.4545 | truth |

*Origin time profiled over 0.05-0.50 s at 0.01 s increments; the true-source/time column uses the actual recorded origin time. The 11-point line profile interpolates between FMM and the known true location for **diagnosis only**. It was not supplied to the inversion.

The returned location has **lower model waveform loss than the true location in all eight events**. For event 03 the loss falls from 1.1805 at the true source to 0.7380 at a location 952 m away. For event 07 even the time-profiled loss directly at the true location (1.0271) exceeds the loss at the FMM position (0.8280). The optimizer therefore follows the stated objective, but that objective is not reliably minimized at the true source. Adding more local iterations alone cannot repair this mismatch. Seven of eight line profiles favor the truth or a point toward it, yet the four-dimensional optimizer can move into a still lower, wrong-location basin off that line.

At the eight known sources, the frozen eikonal travel-time predictor differs from threshold-based FD first-arrival picks by an average per-event RMSE of 0.376 s. This is a descriptive comparison: the 5%-of-peak picker, a band-limited FD waveform and a geometric eikonal arrival do not share an exact arrival definition. The training and FD scripts both specify a 5 Hz Ricker pulse, a 50 m Gaussian source, and zero values on all six boundaries. No direct source-wavelet or boundary-setting discrepancy was found in this code check. A learned travel-time error, finite waveform-model capacity, sampling/weighting and numerical FD-PINN differences may still contribute; the present audit cannot isolate their causal shares.

For a more comparable traveltime target, the frozen carrier was also evaluated against FMM on the **same eight source-receiver pairs**, using the original baseline's 80 m velocity sampling and trilinear receiver interpolation. Eikonal minus FMM traveltimes have a mean bias of **+0.071 s**, a mean event RMSE of **0.086 s**, and a mean RMSE of **0.047 s after subtracting each event's constant bias**. The last value measures receiver-to-receiver moveout disagreement, which an origin-time adjustment cannot eliminate. The FMM source position is snapped to the nearest 80 m cell (up to 69.3 m here), so part of this discrepancy reflects the reference discretization. At 5 Hz, the remaining 0.047 s is roughly one quarter of a period; the comparison identifies a plausible source of phase error, but does not establish that it alone causes the observed localization failures.

The next **development** step is to quantify waveform and arrival errors separately on the 29 training sources and designated development sources, and to investigate forward-operator accuracy before introducing a new localization rule. Any changed operator or search rule requires newly registered, unseen events for a new confirmation; these eight events cannot be reused as untouched evidence.

Machine-readable 11-point profiles, best times, coordinates and per-event losses are in `eikonal_independent_objective_diagnostic.json`. Arrival-pick metrics are in `eikonal_independent_traveltime_diagnostic.json`; paired receiver traveltimes and FMM source-grid offsets are in `eikonal_independent_carrier_vs_fmm.json`.
