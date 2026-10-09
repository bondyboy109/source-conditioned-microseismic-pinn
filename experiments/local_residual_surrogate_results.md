# Local-FD residual-surrogate development audit

This experiment used the fixed two-neighbour local FD interpolator as a
baseline and trained a small source-dependent neural network to predict the
interpolation residual. The residual network received receiver coordinates,
time, source coordinates, source time, and the interpolated waveform value.
Training used leave-one-out interpolation targets from the 33 registered FD
records and 2,000 Adam updates.

The 16 evaluation records were opened in earlier experiments. They were not
used to tune the model, but they are still historical development data rather
than independent confirmation.

| Forward surrogate | Mean full-record L2 | Mean active-sample L2 |
|:--|--:|--:|
| Local FD interpolation | 1.2323 | 1.0585 |
| Residual surrogate | 1.1183 | 1.0324 |
| Frozen neural checkpoint | **1.0493** | **0.9614** |

The residual network improved the raw local interpolator, but remained 6.6%
worse than the frozen neural checkpoint and improved only 6 of 16 records. The
predefined gate (at least 5% improvement and at least 9 improved records) did
not pass. Therefore this model must not be used for a new localization test.

The result narrows the diagnosis: local FD information is useful, but a small
post-hoc correction cannot recover the missing global wavefield response. A
future positive experiment would need joint training of the source-conditioned
wavefield and the local baseline, or a richer mixture-of-experts with a learned
gate and an explicit rejection/uncertainty rule outside the training domain.
