# FMM-aligned MoE localization development result

All 16 records were already used for forward development; this is not independent confirmation.

| Metric | Waveform only | Waveform + FMM arrival |
|---|---:|---:|
| Median location error (m) | 840.3 | 395.3 |
| Mean location error (m) | 1369.8 | 388.2 |
| Median origin-time error (s) | 0.0996 | 0.0171 |
| Events below 500 m | 3/16 | 11/16 |

Hybrid improved spatial error in 16/16 events.

## Predeclared gates

| Gate | Result |
|---|---|
| all_events_have_at_least_six_valid_picks | PASS |
| hybrid_median_location_below_750m | PASS |
| hybrid_median_origin_time_below_0_10s | PASS |
| hybrid_median_location_not_worse_than_waveform_only | PASS |
| hybrid_improves_at_least_nine_events | PASS |
| hybrid_refinement_moves_25m_in_at_least_twelve_events | PASS |

All gates passed: **True**.

Freeze the method and register a new prospective confirmation.

Limitations: These sixteen source positions already served as forward-model development data. The comparison is a development audit, not independent confirmation. It uses one noise-free 5-Hz synthetic Overthrust model, one receiver geometry, known velocity, and a previously developed picker-bias correction.
