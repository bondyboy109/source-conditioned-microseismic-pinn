# Seeded replication of the prototype-gated MoE

The prototype-gated residual MoE was trained twice with independent seeds
(`2027` and `2028`) using the same 1500-update protocol. Both checkpoints were
then evaluated at the true source parameters on the independent eight-event
holdout.

| model | holdout mean full relative L2 | event std | relative improvement vs base | events improved | mean gate |
|---|---:|---:|---:|---:|---|
| Frozen base | 1.132645 | -- | 0.000% | -- | -- |
| Prototype MoE, seed 2027 | 1.123828 | 0.053831 | 0.779% | 7/8 | [0.012, 0.659, 0.328] |
| Prototype MoE, seed 2028 | 1.122427 | 0.053947 | 0.902% | 7/8 | [0.029, 0.643, 0.328] |
| Two-seed mean +/- std | 1.123127 +/- 0.000990 | -- | 0.840% +/- 0.061% | -- | [0.021, 0.651, 0.328] |

The two seeds give a similar positive forward change, and neither gate
collapses. The result is therefore more reproducible than the earlier generic
MLP-gate MoE. Nevertheless, the mean improvement is only `0.84%`, below the
pre-registered 5% decision threshold. The method should be described as a
stable but modest forward-operator refinement, not as a decisive accuracy
advance.

This remains a forward-only test at known source parameters. It does not
validate source localization, uncertainty calibration, or superiority over FD
inversion. Those claims require a separate source-parameter search using this
operator and independent localization metrics.

Machine-readable outputs:

- `experiments/prototype_moe_holdout_v2.json`
- `experiments/prototype_moe_seed2028_holdout.json`
- `experiments/prototype_moe_seed2028_results.json`
