# Reference-velocity sweep on development events

The travel-time carrier was trained from scratch four times with reference
velocities of 2800, 3500, 4500, and 5500 m/s. All runs used the same 16 FD
training events, seed 42, 5 Hz, 5000 receiver-only updates, learning rate
`1e-4`, batch size 1024, all 16 events per batch, receiver-loss weight 0.1,
and a 20 m distance core. The three development events were not used for
training or model selection. This sweep changes only the fixed velocity in
the differentiable carrier's straight-ray time hint.

| Reference velocity (m/s) | 250 m raw L2 | 800 m raw L2 | 2000 m raw L2 | Mean raw L2 |
| ---: | ---: | ---: | ---: | ---: |
| 2800 | 0.9608 | 1.3898 | 1.1848 | 1.1785 |
| 3500 | **0.7810** | **1.0306** | **0.9961** | **0.9359** |
| 4500 | 0.7919 | 1.3117 | 1.1608 | 1.0881 |
| 5500 | 0.8705 | 1.4075 | 1.1990 | 1.1590 |

The 3500 m/s setting is best on all three development events in this
predeclared scratch-training sweep. Its shift-and-gain diagnostic L2 values
are 0.6683, 0.8521, and 0.9047 at 250, 800, and 2000 m, respectively. The
late-coda mismatch remains substantial, and the 800 m and 2000 m raw errors
remain close to or above one. Therefore the reference velocity is a useful
carrier hyperparameter, but selecting it cannot be presented as a solution to
the source-conditioned forward-model generalization problem.

The scratch protocol uses a newly isolated receiver-batch RNG, whereas the
previously selected 3500 m/s checkpoint consumed the global RNG. The old
checkpoint must not be used as a paired point in this table; only the four
scratch runs share the same receiver mini-batches.

The checkpoint and raw audit files are stored in:

* `outputs/reference_speed_scratch_2800_receiver_5000/`
* `outputs/reference_speed_scratch_3500_receiver_5000/`
* `outputs/reference_speed_scratch_4500_receiver_5000/`
* `outputs/reference_speed_scratch_5500_receiver_5000/`
* `outputs/reference_speed_scratch_<speed>_receiver_dev.json`
