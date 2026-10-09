# FD source-difference weight sweep

All branches restart from the same 500-update checkpoints and replay the same 1000 target batches and FD-pair sequence. Weight zero is the existing restarted control.

| Pretraining | Pair weight | Median training-pair delta L2 | Mean development full-trace L2 | Development change vs control |
|---|---:|---:|---:|---:|
| source_to_target | 0.000 | 0.846 | 0.816 | +0.000 |
| source_to_target | 0.025 | 0.838 | 0.817 | +0.001 |
| source_to_target | 0.050 | 0.832 | 0.817 | +0.001 |
| source_to_target | 0.100 | 0.825 | 0.820 | +0.005 |
| target_to_target | 0.000 | 0.805 | 0.851 | +0.000 |
| target_to_target | 0.025 | 0.802 | 0.854 | +0.002 |
| target_to_target | 0.050 | 0.800 | 0.855 | +0.004 |
| target_to_target | 0.100 | 0.796 | 0.858 | +0.007 |

Lower values are better. Selection from this table would be development tuning and requires a fresh locked localization set.

This sweep uses target training pairs and 16 previously inspected development events. It is suitable for choosing a future protocol only; it is not an independent test and does not support a localization claim.
