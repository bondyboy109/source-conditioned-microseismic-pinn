# FMM-aligned MoE prospective confirmation

The frozen method was evaluated once on eight source positions absent from all earlier data.

| Metric | Waveform only | Hybrid | FMM baseline |
|---|---:|---:|---:|
| Median location error (m) | 2400.2 | 362.9 | 276.8 |
| Mean location error (m) | 3326.2 | 481.6 | 272.2 |
| Median origin-time error (s) | 0.0923 | 0.0352 | 0.0082 |
| Events below 500 m | 0/8 | 5/8 | 7/8 |

## Predeclared gates

| Gate | Result |
|---|---|
| all_events_have_at_least_six_valid_picks | PASS |
| hybrid_median_location_below_500m | PASS |
| hybrid_at_least_five_events_below_500m | PASS |
| hybrid_median_origin_time_below_0_05s | PASS |
| hybrid_at_least_six_origin_times_below_0_05s | PASS |
| hybrid_median_location_not_worse_than_waveform_only | PASS |
| hybrid_improves_over_waveform_in_at_least_six_events | PASS |

All gates passed: **True**.

Limitations: Unseen source positions share the same noise-free 5-Hz synthetic Overthrust velocity, receiver geometry, source mechanism, and FD solver as development. This is source-position confirmation, not validation on unseen geology, noise, or field data.
