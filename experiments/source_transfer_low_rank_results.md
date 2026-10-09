# Source-conditioned low-rank source-transfer result

| Arm | Median eloc (m) | Mean eloc (m) | Mean et (s) | Mean full relative L2 |
|---|---:|---:|---:|---:|
| source_to_target | 553.0 | 565.1 | 0.0256 | 0.7259 |
| target_to_target | 510.5 | 465.2 | 0.0302 | 0.7202 |
| target_only_1500 | 528.9 | 503.0 | 0.0159 | 0.6847 |

Three paired seeds and four held-out source positions compare source pretraining plus target transfer, target pretraining plus target continuation, and target-only training under the same low-rank operator. The staged source-to-target and target-to-target arms each use 1500 + 1500 updates, whereas target-only uses 1500 total updates. Only the staged arms match the total update budget; target-only controls target-stage exposure. The frozen protocol's description of a common update budget is therefore too broad. This is a pilot, not a statistical significance claim.
