# Locked test: geometry-feature ablation

No test event was used during training or model selection.

| Arm | Outcomes | Median hybrid eloc (m) | Mean hybrid eloc (m) | Median hybrid et (s) | Mean full waveform relative L2 |
|---|---:|---:|---:|---:|---:|
| source_to_target_no_geometry | 12 | 545.1 | 529.8 | 0.0214 | 0.7692 |
| source_to_target_geometry | 12 | 576.8 | 585.2 | 0.0165 | 0.7729 |
| target_to_target_no_geometry | 12 | 490.6 | 450.3 | 0.0229 | 0.7506 |
| target_to_target_geometry | 12 | 493.0 | 481.8 | 0.0240 | 0.7439 |

Paired test-event differences use the same seed and event; a negative `eloc`
delta means the second arm is better.

| Comparison | Mean eloc delta (m) | Median eloc delta (m) | Second arm better |
|---|---:|---:|---:|
| source_to_target_no_geometry -> source_to_target_geometry | +55.4 | +10.8 | 4/12 |
| target_to_target_no_geometry -> target_to_target_geometry | +31.5 | +9.9 | 5/12 |
| source_to_target_no_geometry -> target_to_target_no_geometry | -79.5 | -86.9 | 8/12 |
| source_to_target_geometry -> target_to_target_geometry | -103.4 | -130.6 | 10/12 |

The geometric features do not improve locked-test localization. The source
transfer arm is worse after adding them, despite a small origin-time improvement.
The target-only reference remains better for spatial localization in both feature
settings.

Locked evaluation only. No training, tuning, or model selection uses these four reserved test events.
