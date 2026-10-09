# Geometry-feature source-transfer pilot

The model adds normalized source-receiver distance and a stable inverse-distance feature. All other data, FMM fields, optimizer settings and sampling remain registered.

| Arm | Outcomes | Mean development relative L2 | Median development relative L2 |
|---|---:|---:|---:|
| source_to_target | 48 | 1.1441 | 1.1175 |
| target_to_target | 48 | 1.1531 | 1.1496 |

Development-only pilot. Uses the existing 16 source events and 33 target events for training and the existing 16 development events for evaluation. The four reserved source_transfer_moe_test events and eight source-conditioned confirmation events are never loaded. The 500+500 update budget is a capacity diagnostic, not a final comparison.
