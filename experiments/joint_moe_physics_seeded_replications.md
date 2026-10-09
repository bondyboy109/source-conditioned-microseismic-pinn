# Seeded replication of the physics-and-receiver joint MoE

## Purpose

This experiment tests whether the previously observed improvement of the joint
physics-and-receiver mixture-of-experts (MoE) is reproducible across random
initializations. The base source-conditioned PINN is frozen. Two independent
seeds (7 and 123) train the three residual experts and the source-dependent
softmax gate for 1000 updates using the same 33-record training pool and the
same physics, receiver, and gate-balance loss terms.

The 16 records used for evaluation were already opened during development and
are therefore used only for seed-stability analysis. They are not an
independent test set. The independent eight-event holdout remains the correct
test set for the forward generalization result.

## Results

| model / seed | mean full relative L2 | change vs frozen base |
|---|---:|---:|
| Frozen source-conditioned base | 1.049326 | 0.00% |
| Joint MoE, seed 7 | 1.049819 | -0.047% |
| Joint MoE, seed 123 | 1.050167 | -0.080% |
| Joint MoE, seeds 7 and 123 (mean +/- std) | 1.049993 +/- 0.000246 | -0.064% +/- 0.017% |

The two MoE runs differ by only 0.000349 in mean relative L2, so the
optimization is numerically repeatable at this training budget. However, both
runs are marginally worse than the frozen base on the 16-record development
set. The earlier single-run 0.671% improvement should therefore be treated as
a development observation rather than a reproducible effect.

At update 1000, the mean gate vectors were:

| seed | gate 1 | gate 2 | gate 3 |
|---:|---:|---:|---:|
| 7 | 0.33290 | 0.33325 | 0.33384 |
| 123 | 0.33296 | 0.33312 | 0.33392 |

The gate did not collapse: all three experts retained approximately one-third
of the probability mass. This indicates that the lack of improvement is not
caused by a single expert taking over, but by the residual experts failing to
reduce the forward error at this training budget.

Training wall time was 85.87 s for seed 7 and 85.94 s for seed 123 on the
RTX 4060 Laptop GPU. The checkpoints are:

- `outputs/joint_moe_physics_replications/seed_7.pth`
- `outputs/joint_moe_physics_replications/seed_123.pth`

The machine-readable source record is
`experiments/joint_moe_physics_seeded_replications.json`.

## Interpretation and reporting decision

This replication does not support a claim that physics-and-receiver joint MoE
consistently improves the development forward error. The manuscript should
report the independent eight-event holdout result separately (where the
previous MoE checkpoint improved the mean full L2 by 5.57%) and label the
seeded experiment as a stability check. It should not pool the 16 development
records with the eight-event holdout or present the development result as an
unbiased generalization estimate.

The next useful experiment is to evaluate both seeded checkpoints on the
independent eight-event holdout. If the holdout gain is also absent, the MoE
should be described as a differentiable source-conditioned interface with a
small but not yet statistically established forward benefit, rather than as a
validated accuracy improvement.
