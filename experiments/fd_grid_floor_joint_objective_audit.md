# FD truth-versus-nearest-grid objective audit

Truth enters these comparisons explicitly; all three events were previously inspected. This isolates grid and objective effects and is not an independent localization result.

The same frozen corrected picks, FMM fields, FD solver, 0.1 weight, and origin-time bounds are used at each position. The truth position is included solely for this diagnostic.

| Event | Grid floor (m) | Truth waveform MSE | Truth arrival loss | Truth joint | Nearest waveform MSE | Nearest joint | Selected joint |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh_west_mid | 1700.4 | 0.00000 | 0.1559 | 0.0156 | 1.0000 | 1.8290 | 1.5907 |
| fresh_east_deep | 852.9 | 0.00000 | 0.1179 | 0.0118 | 0.6735 | 0.8664 | 0.8664 |
| fresh_center_shallow | 447.3 | 0.00001 | 0.1012 | 0.0101 | 0.3002 | 0.4295 | 0.4295 |

The JSON also records the optimized origin time, gain, and separate losses for both fixed positions and the previous global winners. The nearest-grid candidate was part of the original 75-node search; the truth position was not.

At the west-mid event, the true source at 0.220 s gives fitted gain 1.000 and essentially zero waveform error. Its nearest grid node is 1700.4 m away; with the arrival term included, that node chooses 0.234 s and gain 0, yielding waveform MSE 1.000 and joint loss 1.829. The selected but more distant node chooses the lower time bound, gain 0.0083, and joint loss 1.591. Thus the coarse candidate set excludes the well-fitting source, and the nonnegative unconstrained gain permits a nearly silent waveform to rank above the closest node. The FMM pick residual at the true source is small (0.039 s RMSE across 12 valid receivers), so a global pick bias alone does not explain this event's failure.

These numbers do not show that finer gridding alone will recover every event. A prospective test needs a prespecified spatial refinement and an amplitude-identifiability rule, followed by previously unseen source positions. Do not tune either using these three inspected records and then count them as confirmation.
