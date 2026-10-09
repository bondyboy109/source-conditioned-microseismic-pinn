# Five-seed source-conditioned ensemble audit

Five members used seeds 42, 7, 123, 2028, and 4096.  Every member used the
same 25 receiver records, architecture, 5000 receiver-supervised updates, and
1000 hybrid updates.  Only initialization and mini-batch sampling changed.
The unchanged global source search was applied to the eight already inspected
cross-region events, so this is a post-hoc stability audit rather than an
independent calibration experiment.

| Seed | Mean location error (m) | Median location error (m) | Mean origin-time error (s) | Median origin-time error (s) |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 517.4 | 311.1 | 0.0457 | 0.0154 |
| 7 | 563.5 | 437.1 | 0.0584 | 0.0189 |
| 123 | 1216.8 | 1048.2 | 0.0760 | 0.0235 |
| 2028 | 585.7 | 434.8 | 0.0560 | 0.0422 |
| 4096 | 591.6 | 692.6 | 0.0543 | 0.0222 |

The component-wise ensemble mean has a mean location error of 500.6 m, a
median location error of 543.7 m, a mean origin-time error of 0.0460 s, and a
median origin-time error of 0.0364 s.  Averaging members therefore does not
improve on seed 42 or the 191.8 m FMM median on these events.

| Component | Covered events | Empirical coverage | Mean interval width | Median interval width |
| --- | ---: | ---: | ---: | ---: |
| x | 8/8 | 100.0% | 1766.6 m | 1581.3 m |
| y | 7/8 | 87.5% | 1822.7 m | 1972.0 m |
| z | 6/8 | 75.0% | 829.2 m | 708.0 m |
| Origin time | 7/8 | 87.5% | 0.266 s | 0.280 s |

The nominal intervals are the member mean plus or minus 1.96 sample standard
deviations.  Their broad widths expose seed instability; they are not
confidence intervals or calibrated predictive intervals.  The ensemble does
not include observation noise, velocity uncertainty, receiver geometry,
source physics, or model-form uncertainty.

Five-member training required 356.47 s in total.  The 40 member-event
inversions required 151.49 s, or 3.787 s per member-event.  Full member
estimates, component errors, interval endpoints, checkpoint hashes, and costs
are stored in `source_conditioned_ensemble_results.json`; the frozen training
manifest is `source_conditioned_ensemble_members.json`.
