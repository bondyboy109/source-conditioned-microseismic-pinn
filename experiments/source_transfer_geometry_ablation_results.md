# Matched geometry-feature ablation

All four arms use identical events, seeds, 500+500 updates, sampling and optimizer settings.

| Arm | Outcomes | Mean development relative L2 | Median development relative L2 |
|---|---:|---:|---:|
| source_to_target_no_geometry | 48 | 1.1348 | 1.0965 |
| source_to_target_geometry | 48 | 1.1441 | 1.1175 |
| target_to_target_no_geometry | 48 | 1.1288 | 1.1118 |
| target_to_target_geometry | 48 | 1.1531 | 1.1496 |

Development-only matched-budget ablation. The four reserved source_transfer_moe_test events and eight source-conditioned confirmation events are never loaded.
