# Matched geometry-feature ablation

All four arms use identical events, seeds, 1500+1500 updates, sampling and optimizer settings.

| Arm | Outcomes | Mean development relative L2 | Median development relative L2 |
|---|---:|---:|---:|
| source_to_target_no_geometry | 48 | 0.8138 | 0.7534 |
| source_to_target_geometry | 48 | 0.7938 | 0.7512 |
| target_to_target_no_geometry | 48 | 0.8445 | 0.8358 |
| target_to_target_geometry | 48 | 0.8542 | 0.8730 |

Paired differences use the same seed and development event. A negative delta means
the second arm has lower relative L2.

| Comparison | Mean delta | Median delta | Second arm better |
|---|---:|---:|---:|
| source_to_target_no_geometry -> source_to_target_geometry | -0.0201 | -0.0120 | 32/48 |
| target_to_target_no_geometry -> target_to_target_geometry | +0.0097 | +0.0023 | 19/48 |
| source_to_target_no_geometry -> target_to_target_no_geometry | +0.0307 | -0.0133 | 27/48 |
| source_to_target_geometry -> target_to_target_geometry | +0.0605 | +0.0230 | 16/48 |

The geometric features therefore improve the source-transfer arm at the larger
update budget, but do not improve the target-only arm. The transfer arm remains
worse than the matched target-only reference when both use the geometric model.

Development-only matched-budget ablation. The four reserved source_transfer_moe_test events and eight source-conditioned confirmation events are never loaded.
