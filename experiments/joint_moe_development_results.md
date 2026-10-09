# Differentiable source-region mixture-of-experts audit

This experiment implements a genuine differentiable source-region mixture. A
shared frozen source-conditioned wavefield is followed by three trainable
residual experts. A trainable softmax gate receives `(x_s, y_s, z_s, t_s)` and
combines the three expert corrections. All experts and the gate are optimized
jointly on the 33 registered FD receiver records.

The model is differentiable with respect to all four source parameters. A
direct autograd check returned finite, nonzero source-gradient norm
`1.3693×10⁻⁴` for a random batch.

The 16 evaluation records were opened in earlier experiments, so this remains
a development diagnostic rather than independent validation.

| Forward surrogate | Mean full-record L2 | Mean active-sample L2 |
|:--|--:|--:|
| Frozen base model | 1.04933 | **0.96142** |
| Joint gated MoE | **1.04811** | 0.96197 |

The MoE improved the mean full-record L2 by only `0.12%` and improved 9 of 16
records. The mean gate usage was `[0.316, 0.293, 0.392]`; the largest expert
received less than 90% of the mass, so the gate did not collapse. However, the
predefined gate required at least 5% mean improvement, and it failed.

The implementation is structurally correct but the current data and training
protocol do not provide a meaningful accuracy gain. The model must not be used
for a new localization experiment. A publication-quality version would need
joint physics and receiver losses, more source-region coverage, independent
source tests, and a comparison against the frozen base before claiming a
computational advance.
