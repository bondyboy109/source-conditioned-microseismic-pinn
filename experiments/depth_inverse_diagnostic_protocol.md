# Frozen depth-inverse development diagnostic

Use only the already inspected four `depth_coverage/dev_*.npz` FD events
and the frozen `outputs/depth_coverage_aug24_hybrid_1000/conditioned_5Hz.pth`.
Do not retrain, select a checkpoint, or generate new FD data in this audit.
All results are **development diagnostics** and cannot be called blind
confirmation or localization accuracy.

For all four events, report per-receiver contributions to unaltered waveform
error, observed energy, the `0--0.5, 0.5--1, 1--1.5, 1.5--2 s` error and
energy fractions, and diagnostic per-receiver best gain and shift for active
stations. No gain/shift correction enters the primary error or inverse loss.
For the original `train_09, z=940 m` gate failure, also compare the waveform
*difference* from the nearest depth-training source at `z=860 m`, with FD
and PINN responses both evaluated at their own source coordinates.

For each development source, scan the fixed `(x,y,t)` waveform inverse
objective versus depth at 20 m steps in `[max(100,z_true-160),
min(1500,z_true+160)]`, inserting the exact true depth. Keep 25 receivers
and every fourth 2 ms sample, and divide both predicted and observed
waveforms by the full unstrided observation RMS for every candidate.
Report the true-depth loss, scan minimum, and whether an off-truth depth
has lower misfit; this is **not** a location result because three true
coordinates are supplied.

For the two deeper events (`train_01, z=740 m` and `train_09, z=940 m`),
run a local four-parameter box-constrained Adam diagnostic. Limit x/y to
`truth +/- 160 m`, depth to `truth +/- 160 m`, and source time to
`truth +/- 0.04 s`, intersected with the project's source bounds. Start
once at `(x+80, y, z-80, t+0.02)` and once at
`(x-80, y+80, z+80, t-0.02)`; use 120 updates, normalized box fractions,
learning rate `0.015`, and the same unstrided-RMS waveform objective as
the scan. Report each start, final estimate, true-source misfit, final
misfit, coordinate error and any boundary hit. This search is deliberately
local and truth-centered; its purpose is to diagnose optimizer behavior,
not demonstrate blind localization. Do not use an FD waveform at the true
candidate as a model replacement.
