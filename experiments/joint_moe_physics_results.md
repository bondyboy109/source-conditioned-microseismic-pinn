# Physics-and-receiver joint mixture-of-experts audit

This experiment extends the differentiable source-region mixture-of-experts
model with four jointly evaluated terms: the acoustic PDE residual, the
initial-condition residual, the boundary-condition residual, and the receiver
waveform loss. The frozen source-conditioned base model is shared by three
trainable residual experts. A differentiable softmax gate receives
`(x_s, y_s, z_s, t_s)` and combines the expert corrections. A gate-balance
penalty was included to discourage collapse to one expert.

The run used one random seed, the 5 Hz source term, 1,000 optimization updates,
64 physics points, 32 initial-condition points, 8 boundary points per face,
and a receiver loss weight of 0.1. Training took 114.73 s on the development
GPU. The gate remained non-collapsed, with mean usage
`[0.365, 0.395, 0.240]`.

| Forward surrogate | Mean full-record L2 | Mean active-sample L2 |
|:--|--:|--:|
| Frozen base model | 1.04933 | 0.96142 |
| Physics-and-receiver joint MoE | **1.04228** | **0.95811** |

The joint MoE improved the mean full-record L2 by **0.67%** and the mean
active-sample L2 by **0.34%**. It improved 15 of 16 previously opened records,
but it did not meet the predefined 5% full-record improvement criterion. The
small numerical gain therefore does not establish a meaningful computational
advance, and the model must not yet be used to claim successful localization
or calibrated uncertainty quantification.

The 16 records were opened in earlier development experiments. They are not an
independent validation set, and this one-seed, one-frequency run is a
structural diagnostic. The next scientifically valid step is to improve and
evaluate source-region coverage with a newly held-out set, or to report this
result as a negative development result. New localization experiments should
wait until the forward surrogate passes a prespecified independent forward-error
criterion.
