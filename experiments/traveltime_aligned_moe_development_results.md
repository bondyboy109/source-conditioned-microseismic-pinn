# FMM-traveltime-aligned MoE development result

This is a development-only waveform test. The registered eight-event ensemble confirmation was excluded from training and model selection.

| Metric | Frozen base | Aligned MoE |
|---|---:|---:|
| Mean full relative L2 | 1.0493 | 0.8126 |
| Median full relative L2 | 1.0383 | 0.7935 |
| Events below zero-prediction L2 | -- | 14/16 |
| Events improved | -- | 15/16 |

Mean gate use: `[0.47999468445777893, 0.2436460554599762, 0.27635934948921204]`.

## Predeclared gates

| Gate | Result |
|---|---|
| mean_l2_improves_by_ten_percent | PASS |
| at_least_twelve_events_improve | PASS |
| at_least_twelve_events_beat_zero_prediction | PASS |
| prototype_gate_does_not_collapse | PASS |
| all_source_gradient_components_present | PASS |

All gates passed: **True**.

Proceed to registered development localization without changing the model.

Limitations: One 5-Hz synthetic Overthrust model and one receiver geometry; receiver-supervised waveform surrogate with FMM physics in its coordinates, not a standalone PDE solution or independent validation.
