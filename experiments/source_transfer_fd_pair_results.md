# FD neighboring-source difference supervision pilot

The 500-update checkpoint and 1000 subsequent target batches match the existing restarted control. The new branch adds same-receiver/time FD waveform differences for observed training-source pairs.

| Pretraining | Continuation | Median training-pair delta relative L2 | Mean development full-trace relative L2 |
|---|---|---:|---:|
| source_to_target | restarted_control | 0.841 | 0.816 |
| source_to_target | fd_pair | 0.822 | 0.826 |
| target_to_target | restarted_control | 0.805 | 0.851 |
| target_to_target | fd_pair | 0.781 | 0.865 |

Lower relative L2 is better. The 31 pair errors are in-sample; the 16 development waveforms were previously inspected. This experiment does not measure localization performance.

All 31 pairs use the 33 target training events. This is a development-only pilot with previously inspected development waveforms; do not call these independent tests or claim improved localization without a fresh locked evaluation.
