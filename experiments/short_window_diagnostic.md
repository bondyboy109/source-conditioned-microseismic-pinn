# Short-window source-conditioned diagnostic

This development experiment crops the existing six-face zero-Dirichlet FD
records to 0-2 s, keeping the same 5 Hz source, 40 m grid, 0.002 s time
step, stations, and source coordinates. A relative-source/travel-time
carrier was trained with receiver supervision only, seed 42, 5000 steps,
batch size 1024, and output scale `1e-5`. No cropped development event was
passed to training.

| Event | Distance from train_01 | Relative L2 over 0-2 s | Best global-gain L2 |
| --- | ---: | ---: | ---: |
| train_01 | 0 m | 0.439 | 0.438 |
| train_02 | training source | 0.387 | 0.375 |
| train_03 | training source | 0.495 | 0.465 |
| train_04 | training source | 0.408 | 0.392 |
| train_05 | training source | 0.489 | 0.458 |
| train_06 | training source | 0.349 | 0.337 |
| train_07 | training source | 0.533 | 0.524 |
| train_08 | training source | 0.540 | 0.526 |
| dev_distance_0250 | 250 m | **0.808** | 0.748 |
| dev_distance_0800 | 800 m | 1.279 | 0.995 |
| dev_distance_2000 | 2000 m | 1.178 | 1.000 |

The 250 m development source improves from 1.275 in the corresponding
9-second model to 0.808 in the first 2 s, beating the zero-wavefield baseline
of 1.0. The two farther development sources remain worse than zero without
post-hoc amplitude adjustment. This supports the conclusion that late
boundary-return energy was a major source of the 9-second mismatch, but it
also shows that a short window does not solve source generalization over
kilometre-scale changes. The amplitude-adjusted values are diagnostic only;
the original-amplitude L2 is the primary metric.

The cropped data are in `data/observations/window_0to2s/`; the checkpoint and
all audits are in `outputs/dev_relative_carrier_5hz_window0to2_train8_5000/`.
This remains a supervised representation diagnostic, not a validated PINN
forward model or localization result.

## Short-window leave-one-source-out and gate ablation

The same 0-2 s setup was evaluated in eight leave-one-source-out folds. Each
fold trained on seven cropped events for 5000 steps and evaluated only on the
omitted cropped event. The errors were `1.120, 1.022, 1.115, 1.068, 1.270,
1.118, 1.053, 1.004` (median `1.087`; zero baseline `1.0`). Thus, the
single 250 m development result is not representative: short windows alone
do not provide reliable source-to-source generalization.

As a registered architecture ablation, removing the `hard_initial` time gate
from the same short-window, eight-source training reduced the three distance
development errors to `0.795` (250 m), `1.197` (800 m), and `1.118` (2000 m),
compared with `0.808, 1.279, 1.178` with the gate. The gate contributes to
the error, but its removal does not make the model viable at kilometre-scale
source changes. The leave-one-out records are in
`outputs/dev_loo_relative_carrier_5hz_window0to2/`; the no-gate checkpoint and
audits are in `outputs/dev_relative_carrier_5hz_window0to2_no_gate_train8_5000/`.
