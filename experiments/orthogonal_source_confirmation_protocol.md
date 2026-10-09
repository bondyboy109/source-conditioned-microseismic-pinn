# Fixed-checkpoint orthogonal source confirmation protocol

This protocol was written before generating any of the ten FD records below.
The selected 20-event checkpoint is
`outputs/local_density_aug20_scratch_hybrid_1000/conditioned_5Hz.pth`
(SHA-256 `16084c46b8d073da157eabed1cd54fcf6164acb45bbb80626ef3ca0e476867f3`).
The paired 16-event control is
`outputs/local_density_control16_scratch_hybrid_1000/conditioned_5Hz.pth`
(SHA-256 `15b5651ba9bc8274579d4eb02bc6ce1912832003b7b8d69a82fc5c28b76b9374`).
No retraining, calibration, gain fitting, checkpoint selection, or inversion
is permitted based on these observations.

Generate independent-of-training **synthetic FD** records with the original
Overthrust velocity, 5 Hz Ricker source, 40 m FD grid, dt = 0.002 s,
0--2 s recording, 25 unchanged receivers, and six zero-Dirichlet faces.
The locations below avoid all 20 receiver-training source coordinates:

| Event file stem | Source x,y,z (m), origin time (s) | Anchor, shift |
| --- | --- | --- |
| `a01_ym120` | 3500, 3880, 500, 0.12 | train_01, y -120 m |
| `a01_yp120` | 3500, 4120, 500, 0.12 | train_01, y +120 m |
| `a01_zm080` | 3500, 4000, 420, 0.12 | train_01, z -80 m |
| `a01_zp080` | 3500, 4000, 580, 0.12 | train_01, z +80 m |
| `a01_xp400` | 3900, 4000, 500, 0.12 | train_01, x +400 m |
| `a09_ym120` | 6500, 3380, 700, 0.20 | train_09, y -120 m |
| `a09_yp120` | 6500, 3620, 700, 0.20 | train_09, y +120 m |
| `a09_zm080` | 6500, 3500, 620, 0.20 | train_09, z -80 m |
| `a09_zp080` | 6500, 3500, 780, 0.20 | train_09, z +80 m |
| `a09_xp400` | 6900, 3500, 700, 0.20 | train_09, x +400 m |

Primary endpoint: report **every** raw original-amplitude receiver-waveform
relative L2 against FD, with per-anchor and overall arithmetic means, for
the frozen 20-event model and 16-event control. Report the worst event, not
only the mean. An across-direction local feasibility signal requires *all*
eight y/z events to have raw L2 below 0.5, both +400 m events to have raw L2
below 0.5, and all ten events to beat the zero-prediction baseline of 1.0.
This is deliberately stricter than the earlier x-direction development
criterion; passing would support only these two neighborhoods, not the
entire faulted model or localization.

Secondary endpoints: finite-displacement response relative L2 versus each
anchor's original FD record, peak-amplitude ratios, time-window error, the
number of records overlapping the training set (expected zero), FD runtime
and memory. No trained source estimates, confidence intervals, or claimed
computational advantage are produced by this protocol. Events generated
after extensive same-model development are a one-time synthetic confirmation,
not field validation; once evaluated they cannot be reused as untouched tests.
