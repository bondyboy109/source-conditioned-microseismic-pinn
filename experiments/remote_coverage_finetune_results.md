# Remote-source coverage fine-tuning and generalization audit

The selected midpoint-teacher checkpoint was fine-tuned for 500 updates using the four remote records from the preceding confirmation. This is a development experiment because those records had already been opened. The base checkpoint remained fixed as a paired control.

On the four remote records used for fine-tuning, the mean full-record relative L2 decreased from `1.026` to `0.174`. This is a training fit and does not measure generalization. On the four local confirmation records, the error changed from `0.238` to `0.255`; on the 12 previously used depth records, it changed from `0.292` to `0.300`.

To test transfer, the fine-tuned candidate and the frozen base were then evaluated on 16 previously opened records that were not included in the four-source fine-tuning set. The candidate was worse on 15 of 16 records:

| Metric | Frozen base | Remote-fine-tuned candidate |
|:--|--:|--:|
| Mean full L2 | 1.049 | 1.074 |
| Median full L2 | 1.038 | 1.074 |
| Mean active L2 | 0.961 | 0.966 |
| Events improved | — | 1 / 16 |

The experiment therefore shows overfitting to the four added remote records rather than transferable source coverage. Adding a few remote waveforms with the same receiver-only objective is not sufficient to repair the global forward operator. This candidate should not replace the midpoint-teacher checkpoint, and it should not be used for localization.

The raw fine-tuning result is in `remote_coverage_finetune_results.json`. The 16-event transfer audit is in `remote_coverage_generalization_audit_results.json`, with its frozen input manifest in `remote_coverage_generalization_audit_protocol.json`. Both analyses are development records; none of the 16 events remains untouched for a future claim.
