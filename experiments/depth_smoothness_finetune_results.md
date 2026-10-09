# Depth-smoothness fine-tuning: development result

This is a development-only ablation. It starts from the registered 33-source, 5 Hz checkpoint and fine-tunes two identical copies for 500 updates: `control` uses receiver supervision only, while `candidate` adds a midpoint source-depth smoothness regularizer. The three endpoint pairs were already inspected during model development, and all 12 depth-offset records below were already spent development data. They must not be reported as independent validation or as formal localization accuracy.

The run took 10.54 s on the configured CUDA device. The final receiver loss was 0.01083 for `control` and 0.04671 for `candidate`; the candidate smoothness loss was 0.00537. Thus, the regularizer was active, but it also increased the ordinary receiver fitting loss under this setting.

## Forward waveform error

Relative L2 is `||prediction - FD|| / ||FD||`. Active L2 uses FD samples whose absolute amplitude is at least 5% of that record's global peak.

| Development group | Records | Control full L2 | Candidate full L2 | Control active L2 | Candidate active L2 |
|:--|--:|--:|--:|--:|--:|
| ±80 m | 4 | 0.202 | 0.182 | 0.174 | 0.155 |
| ±160 m | 4 | 0.077 | 0.144 | 0.028 | 0.092 |
| ±280 m | 4 | 0.646 | 0.710 | 0.563 | 0.624 |
| All 12 | 12 | 0.308 | 0.345 | 0.255 | 0.290 |

The candidate improves the nearest ±80 m group by about 9.6% in full-record L2, but it is worse at ±160 m and ±280 m. Across all 12 records, full-record L2 increases by 12.0% and active L2 increases by 13.9%. The smoothness penalty therefore does not provide a reliable improvement in this configuration.

## Interpretation

The result does not support retaining this regularizer as a claimed solution to the source-conditioned forward-model problem. It can make nearby depth responses slightly smoother, but the trade-off is poorer receiver fitting and larger errors outside the closest group. The experiment also uses one seed, one velocity model, one frequency, previously inspected sources, and no source inversion. It should be retained as a negative development ablation or omitted from the main paper; it should not be used to support localization or uncertainty claims.

Inputs and provenance are recorded in `depth_smoothness_finetune_protocol.json` and `depth_smoothness_finetune_results.json`. The two saved checkpoints are `outputs/depth_smoothness_finetune_500/control.pth` and `outputs/depth_smoothness_finetune_500/candidate.pth`.
