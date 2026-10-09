# Receiver-loss weight sweep on development events

This development-only experiment trained four source-conditioned PINNs with
`receiver_loss_weight` equal to 0.03, 0.1, 0.3, and 1.0. All other settings
were held fixed: 5 Hz carrier, 16 training events, seed 42, 5,000 updates,
learning rate `1e-4`, receiver batch size 1024, and receiver-only training.
The three registered events at 250, 800, and 2000 m were evaluation-only.

| weight | 250 m raw L2 | 800 m raw L2 | 2000 m raw L2 | mean raw L2 | 250 m shift+gain | 800 m shift+gain | 2000 m shift+gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.03 | 0.7664 | 1.0196 | 0.9780 | 0.9213 | 0.6672 | 0.8616 | 0.8912 |
| 0.10 | 0.7810 | 1.0306 | 0.9961 | 0.9359 | 0.6683 | 0.8521 | 0.9047 |
| 0.30 | 0.7925 | 1.0208 | 0.9946 | 0.9360 | 0.6668 | 0.8468 | 0.9344 |
| 1.00 | 0.7703 | 0.9662 | 1.0186 | 0.9184 | 0.6599 | 0.8297 | 0.9320 |

The raw relative L2 errors remain large for all weights. The lowest
three-event mean raw error is 0.9184 at weight 1.0, but there is no monotonic
relationship and the events do not favor the same weight. The 1.0--2.0 s
relative L2 ranges from 0.94--0.96 at 250 m, 1.19--1.25 at 800 m, and
1.17--1.19 at 2000 m. The sweep does not remove the late waveform mismatch.
The shift-and-gain values are diagnostics only and cannot be used to claim
an improvement of the unadjusted forward model.

This is not a physics-loss ablation. With `--receiver-only`, the objective is
just `receiver_loss_weight * receiver_MSE`; multiplying the objective by a
positive constant does not change the Adam gradient direction and can have
only limited effects through optimizer state and stopping at a fixed update
count. Therefore these results do not establish an optimal balance between
data fit and PDE/IC/BC regularization. That question requires a hybrid sweep
with the physical terms enabled, fixed sampling, and the same initialization.

The JSON records contain the full event-level diagnostics:

* `receiver_weight_0.03_development.json`
* `receiver_weight_0.1_development.json`
* `receiver_weight_0.3_development.json`
* `receiver_weight_1_development.json`

The development set was used only for diagnosis and configuration planning;
it was not used to select a final test result.
