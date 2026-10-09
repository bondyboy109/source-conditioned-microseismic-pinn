# Low-rank operator cross-geology audit

These independent-layered FD records were generated and evaluated in earlier studies of a different frozen model. This is a post-hoc zero-shot audit for the low-rank operator, not a prospective blind test.

| Method | n | Median eloc (m) | Mean eloc (m) | SD eloc (m) | Mean et (s) | <500 m |
|---|---:|---:|---:|---:|---:|---:|
| Waveform only | 4 | 1126.9 | 946.8 | 581.9 | 0.1654 | 1/4 |
| Waveform + FMM arrival | 4 | 163.8 | 208.7 | 146.4 | 0.0597 | 4/4 |
| FMM arrival baseline | 4 | 86.2 | 105.5 | 76.3 | 0.0388 | 4/4 |

## Event errors

| Event | Valid picks | Waveform eloc (m) | Hybrid eloc (m) | FMM eloc (m) | Hybrid et (s) |
|---|---:|---:|---:|---:|---:|
| layered_west_shallow | 8 | 1208.0 | 153.2 | 42.6 | 0.0517 |
| layered_center_deep | 11 | 106.0 | 86.3 | 51.4 | 0.0242 |
| layered_east_mid | 7 | 1045.9 | 420.9 | 207.1 | 0.1489 |
| layered_north_deep | 7 | 1427.4 | 174.4 | 121.0 | 0.0138 |

FMM fields were reused from cache. Construction time was not measured in this evaluation; the earlier 0.00 s entry represented cache reuse rather than solver cost. Fresh construction and branch-specific search timings are reported separately in `source_conditioned_low_rank_cross_geology_cost_results.md`.

Interpretation: this four-event post-hoc audit tests transfer of the new operator to a distinct synthetic geology. It is not a prospective blind test and cannot support a broad generalization claim.

Four synthetic events, one receiver geometry, one frozen Overthrust-trained operator, and an arrival-pick bias calibrated on Overthrust. Existing event records had prior use in other model audits.
