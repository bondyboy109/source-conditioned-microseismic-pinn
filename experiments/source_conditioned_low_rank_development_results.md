# Source-conditioned low-rank operator development result

The model was selected using the 16-event development set only. The independent confirmation set was excluded.

| Metric | Frozen base | Low-rank operator |
|---|---:|---:|
| Mean full relative L2 | 1.0493 | 0.7567 |
| Median full relative L2 | 1.0383 | 0.6826 |
| Events below zero-prediction L2 | -- | 14/16 |
| Events improved over frozen base | -- | 16/16 |

Median absolute source-loss gradient: `[1.0684292316436768, 1.237959623336792, 0.13599801063537598, 0.9155009984970093]`.

## Predeclared gates

| Gate | Result |
|---|---|
| mean_l2_improves_by_ten_percent | PASS |
| at_least_twelve_events_improve | PASS |
| at_least_twelve_events_beat_zero_prediction | PASS |
| all_source_gradient_components_present | PASS |

All gates passed: **True**.

Proceed to independent localization only if this result remains unchanged under the frozen confirmation protocol.

Limitations: One 5-Hz synthetic Overthrust model and one receiver geometry; the operator is a receiver-supervised waveform surrogate with FMM travel time, not an independent PDE solver.
