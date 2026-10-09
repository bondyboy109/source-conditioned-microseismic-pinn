# Receiver-priority MoE ablation

This development ablation increased the receiver-loss weight from `0.1` to
`1.0`, removed the mean-gate balance penalty, and trained for 3000 updates.
The purpose was to test whether the previous joint objective was suppressing
the source-dependent residual experts.

## Results

| evaluation | frozen base mean full L2 | receiver-priority MoE mean full L2 | relative change | events improved |
|---|---:|---:|---:|---:|
| 16-record development set | 1.049326 | 1.048003 | +0.126% | 12/16 |
| independent 8-event holdout | 1.132645 | 1.113601 | +1.681% | 8/8 |

The holdout error decreased, but the gate collapsed: the mean gate usage was
`[0.0271, 0.0098, 0.9632]`. Two experts received almost no probability mass.
The improvement therefore cannot be attributed to a learned mixture of source
regions; it is effectively a single residual expert with an unused pair of
experts. The registered gate-collapse criterion fails.

Training wall time was 345.5 s on the RTX 4060 Laptop GPU. The checkpoint is
`outputs/joint_moe_physics_receiver_priority/joint_moe_physics.pth`.

## Decision

This ablation is rejected as the final method despite the positive holdout
number. It demonstrates that removing the gate regularizer creates a shortcut
that improves the aggregate forward error by sacrificing the intended
multi-expert structure. A subsequent experiment must preserve non-collapsed
regional specialization, for example with source-prototype separation or
explicit region assignment, and must again be tested on the independent
holdout.

Machine-readable output:
`experiments/joint_moe_physics_receiver_priority_holdout_v2.json` and
`experiments/joint_moe_physics_receiver_priority_results.json`.
