# Seeded joint-MoE evaluation on the independent holdout

The two independently trained joint MoE checkpoints were evaluated at the
known true source parameters on the eight-event holdout generated after the
development experiments. The holdout was not used for training or model
selection. This is a forward-operator test, not a source-localization test.

| model | mean full relative L2 | event-to-event std | relative change vs base | events improved |
|---|---:|---:|---:|---:|
| Frozen base | 1.132645 | -- | 0.00% | -- |
| Joint MoE, seed 7 | 1.133150 | 0.053629 | -0.045% | 0/8 |
| Joint MoE, seed 123 | 1.133467 | 0.053742 | -0.073% | 0/8 |
| Seed mean +/- std | 1.133309 +/- 0.000224 | -- | -0.059% +/- 0.020% | -- |

Both seeded MoE checkpoints are marginally worse than the frozen base on every
holdout event. Their mean gates remain close to uniform:

| seed | gate 1 | gate 2 | gate 3 |
|---:|---:|---:|---:|
| 7 | 0.33309 | 0.33291 | 0.33397 |
| 123 | 0.33371 | 0.33272 | 0.33358 |

The result rules out gate collapse as the cause of the missing improvement.
Instead, the current residual experts and training objective do not improve
the forward prediction under this independent evaluation. The earlier
single-checkpoint holdout result (5.57% improvement) is therefore sensitive to
the checkpoint/training realization and should not be presented as a robust
multi-seed gain.

The machine-readable result is
`experiments/joint_moe_physics_seeded_holdout.json`. The evaluation script is
`examples/evaluate_seeded_moe_holdout.py`.

## Consequence for the manuscript

The source-conditioned MoE can be reported as an implemented differentiable
interface whose gate remains stable across seeds. A positive accuracy claim
requires a revised training objective or architecture followed by a fresh
independent holdout evaluation. The current evidence supports no claim that
this joint MoE improves the forward operator, and it does not support source
localization or uncertainty-quantification claims.
