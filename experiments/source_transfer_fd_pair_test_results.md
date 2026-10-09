# Locked FD-pair continuation test

The four reserved source_transfer_moe_test events were evaluated once with identical search, picker, FMM fields and hybrid weight.

| Branch | Outcomes | Median hybrid eloc (m) | Mean hybrid eloc (m) | Median hybrid et (s) |
|---|---:|---:|---:|---:|
| restarted_control | 12 | 521.9 | 507.8 | 0.0208 |
| fd_pair_weight_0.05 | 12 | 538.2 | 511.7 | 0.0207 |

The selected branch improves hybrid location error in 8/12 paired seed-event outcomes. Negative paired differences favor the selected branch; all differences are stored in JSON.

The four source_transfer_moe_test records were reserved before this continuation was trained. This is one four-position synthetic test set, not four independent geologies; no retuning is permitted after this evaluation.
