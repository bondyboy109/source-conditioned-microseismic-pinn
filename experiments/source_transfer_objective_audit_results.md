# Frozen transfer objective at truth and recovered source

At each frozen model, evaluate the same fine-sampled waveform MSE and target-FMM arrival objective at the known true source and the existing hybrid estimate. Positive estimate-minus-truth loss means the true source has a lower objective value; no search parameter or loss weight is changed.

| Arm | Target updates | Estimate objective below truth | Mean estimate minus truth hybrid loss | Mean waveform loss at truth | Mean arrival loss at truth |
|---|---:|---:|---:|---:|---:|
| source_to_target | 0 | 12/12 | -0.7648 | 1.8244 | 0.1106 |
| source_to_target | 500 | 10/12 | -0.0914 | 0.9846 | 0.1106 |
| source_to_target | 1500 | 9/12 | -0.0371 | 0.9368 | 0.1106 |
| target_to_target | 0 | 12/12 | -0.1414 | 1.1586 | 0.1106 |
| target_to_target | 500 | 10/12 | -0.0610 | 0.9207 | 0.1106 |
| target_to_target | 1500 | 10/12 | -0.0395 | 0.7719 | 0.1106 |

For the same twelve seed-event pairs at each stage, the difference below is
the loss at the recovered estimate minus the loss at the known true source.
The arrival contribution already includes the frozen weight of 0.25:

| Arm | Target updates | Waveform prefers wrong estimate | Mean waveform difference | Arrival prefers wrong estimate | Mean weighted arrival difference |
|---|---:|---:|---:|---:|---:|
| source_to_target | 0 | 12/12 | -0.7939 | 0/12 | +0.0291 |
| source_to_target | 500 | 11/12 | -0.1469 | 4/12 | +0.0555 |
| source_to_target | 1500 | 9/12 | -0.1211 | 0/12 | +0.0840 |
| target_to_target | 0 | 12/12 | -0.1947 | 1/12 | +0.0533 |
| target_to_target | 500 | 10/12 | -0.1125 | 2/12 | +0.0514 |
| target_to_target | 1500 | 10/12 | -0.1001 | 1/12 | +0.0606 |

At 1500 updates, the FMM arrival term favors the true location in every
source-to-target comparison, whereas the waveform term favors the wrong
estimate in nine. The existing hybrid objective therefore prefers the wrong
estimate in nine of twelve comparisons. For the four selected events, the
source-to-target waveform loss at truth (mean 0.9368) also exceeds the
target-to-target value (0.7719), even though its earlier **16-event full-trace**
relative L2 was lower. These are different metrics and event subsets; one
cannot substitute for the other. Lower full-trace L2 does not guarantee an
accurate source-dependent loss landscape.

Recomputed waveform loss at the existing estimates agrees with the frozen
localization records to within 2.4e-7. The comparison does not identify the
global minimum or establish that retuning the arrival weight would improve
new events. The selected development events have already been inspected.

Positive estimate-minus-truth loss indicates that the known source has a lower objective than the recovered estimate; it is a diagnostic of search, not proof that a practical search can find it. Negative values show that the frozen surrogate and arrival objective prefer a wrong location over truth. This audit evaluates only the existing estimates and truth, not a global minimum.

Uses four previously inspected development events. The four source_transfer_moe_test events are excluded. These scores diagnose the objective and cannot establish independent transfer benefit.
