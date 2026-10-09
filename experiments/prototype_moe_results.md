# Differentiable source-prototype gated MoE

This experiment replaces the unconstrained MLP gate with a differentiable
soft assignment to three learnable source-parameter prototypes. The prototypes
are initialized at three regions of the normalized source domain and optimized
together with the residual experts. The receiver-loss weight is `0.5`; the
physics, initial-condition, and boundary losses remain active. Training uses
one development seed (`2027`) for 1500 updates.

## Results

| evaluation | frozen base mean full L2 | prototype MoE mean full L2 | relative improvement | events improved | max mean gate |
|---|---:|---:|---:|---:|---:|
| 16-record development set | 1.049326 | 1.045308 | 0.383% | 15/16 | 0.725 |
| independent 8-event holdout | 1.132645 | 1.123828 | 0.779% | 7/8 | 0.659 |

The gate remains non-collapsed on both evaluations. The holdout result is
positive but below the pre-registered 5% improvement criterion. It should be
reported as a small forward-operator improvement rather than a decisive
accuracy gain. No localization or UQ conclusion follows from this
forward-only test.

The first attempted prototype run used physical-second values for the source
time prototypes even though the network input is normalized by the 9-s time
length. That run was interrupted before saving any output and is excluded.
The final run uses normalized time prototypes (`0.015`, `0.030`, and `0.045`).

Training wall time was 423.3 s on the RTX 4060 Laptop GPU. The checkpoint is
`outputs/prototype_moe_development/joint_moe_physics.pth`.

Machine-readable outputs:

- `experiments/prototype_moe_results.json`
- `experiments/prototype_moe_holdout_v2.json`

The implementation is in `mixture_of_experts.py` as
`PrototypeGatedResidualMixtureOfExperts`; the training and evaluation scripts
are `examples/run_prototype_moe_development.py` and
`examples/evaluate_prototype_moe_holdout.py`.
