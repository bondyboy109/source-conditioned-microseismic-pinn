# Uniform-medium physics calibration

This development experiment tests whether receiver-pretrained source-conditioned
weights can be fine-tuned with the acoustic PDE without destroying waveform
accuracy. It uses a separate constant `3500 m/s` model, 5 Hz source, 40 m grid,
25 receivers, and a 0-2 s window. The FD records and PINN both use zero
Dirichlet conditions on all six faces. The training source is `(3500, 4000,
500, 0.12)` and the untouched test source is `(3750, 4000, 500, 0.12)`.

| Stage | Train-source relative L2 | 250 m unseen-source relative L2 | Exact source-peak PDE residual | Active-neighbourhood relative residual RMSE |
| --- | ---: | ---: | ---: | ---: |
| Receiver-only pretraining, 5000 steps | 0.2977 | 0.3675 | -1.0886 | 1.012 |
| Hybrid PDE/IC/BC/receiver, 1000 steps | 0.0336 | 0.2129 | 14.223 | 7.741 |
| Exact-source correction, 500 steps | 0.0315 | 0.2429 | 0.0477 | 1.117 |

The 1 m distance core was then replaced by a checkpointed 10 m core, with the
same source sampling and optimization protocol:

| 10 m-core stage | Train-source relative L2 | 250 m unseen-source relative L2 | Exact source-peak PDE residual | Active-neighbourhood relative residual RMSE |
| --- | ---: | ---: | ---: | ---: |
| Receiver-only pretraining, 5000 steps | 0.2129 | not audited in this run | -0.9910 | 0.995 |
| Hybrid PDE/IC/BC/receiver, 1000 steps | 0.0392 | 0.2254 | 0.3818 | 0.724 |
| Exact-source correction, 500 steps | 0.0212 | 0.2231 | 0.3453 | 0.734 |

The first hybrid stage used 256 physics, 128 initial-condition, and 16 points
per boundary face per update. Half of the physics points were sampled near a
source. It greatly improved receiver waveforms but revealed that continuous
near-source jitter almost never sampled the exact source centre. The resulting
source-centre residual was therefore unconstrained and increased despite a
small batch-average PDE loss.

The correction stage placed 3.125% of physics points exactly at sampled source
locations, including samples at the Ricker peak, and reduced the learning rate
to `1e-6`. This lowered the exact source-peak residual from `14.223` to
`0.0477`; training-source L2 remained near `0.03`. The unseen-source L2 rose
from `0.213` to `0.243`, but stayed below both the receiver-only value `0.368`
and the zero-prediction baseline `1.0`. At 100 m offset the final residual is
`-0.1266`. A 185-point audit over 0-200 m offsets, six spatial directions,
and five source-time offsets gives an active-forcing relative residual RMSE of
`1.117`. This is slightly worse than the receiver-only value `1.012`, so the
correction fixes the audited source peak but does not yet satisfy the PDE over
the source neighbourhood.

The final development checkpoint is
`outputs/uniform_zero_hybrid_exact_5hz_500/conditioned_5Hz.pth`. Its recorded
training time is 22.2 s and peak allocated CUDA memory is 110,377,472 bytes on
the local RTX 4060 Laptop GPU. The zero-boundary FD records are
`data/observations/uniform_3500/zero_train.npz` and `zero_test.npz`, with
independent provenance JSON files.

The fixed-protocol core-radius sweep was:

| Distance core | Train-source relative L2 | 250 m unseen-source relative L2 | Source-peak residual | Active-neighbourhood relative residual RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 5 m | 0.0207 | 0.2239 | 0.283 | 0.860 |
| 10 m | 0.0212 | 0.2231 | 0.345 | 0.734 |
| 20 m | 0.0241 | 0.2167 | -0.179 | 0.554 |

The 20 m core is selected for the next development stage because it gives the
lowest fixed-neighbourhood residual and the lowest held-out waveform error in
this small, predeclared sweep. The selected checkpoint is
`outputs/uniform_zero_core20_hybrid_exact_5hz_500/conditioned_5Hz.pth`. This is
still a calibration result rather than Overthrust localization validation: the
next gate is to repeat the same receiver-only, hybrid, and neighbourhood audit
on the heterogeneous velocity model with untouched events kept out of tuning.
