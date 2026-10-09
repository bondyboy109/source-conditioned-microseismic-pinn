# Frozen paired receiver heads: archived depth-offset audit

This post-hoc development audit evaluates the two receiver-only 5 Hz heads from `fmm_receiver_head_paired_results.md` on **12 already inspected** FD records: offsets of -80, +80, -160, +160, -280 and +280 m around each of two *training* anchors. It reuses the original FD records and pretrained heads without retraining, selecting new models, or generating new observations. Each forward evaluation uses the known source; this is not source localization or independent validation. Both heads have the same frozen learned eikonal model and differ only in the learned versus 80 m reciprocal-FMM travel-time hint supplied to the trained waveform head.

Full and active L2 use the same definitions as the paired experiment. Change L2 measures disagreement between the predicted and FD waveform differences from each corresponding anchor. Values below are arithmetic means of four records per offset magnitude, with no optimized gain, shift or source parameters.

| Depth offset | Learned full L2 | FMM full L2 | Learned active L2 | FMM active L2 | Learned change L2 | FMM change L2 |
|:--|--:|--:|--:|--:|--:|--:|
| 80 m | 0.418 | 0.468 | 0.361 | 0.407 | 0.971 | 1.110 |
| 160 m | 0.659 | 0.746 | 0.591 | 0.678 | 0.861 | 0.963 |
| 280 m | 0.964 | 1.017 | 0.878 | 0.916 | 0.688 | 0.707 |
| All 12 | 0.680 | 0.744 | 0.610 | 0.667 | 0.840 | 0.927 |

FMM lowers full-record L2 for only 1/12 records and change L2 for 2/12. Both heads worsen in absolute waveform error as the depth offset grows. Although change L2 decreases at 280 m, its denominator is the FD *change* norm, which grows with displacement; that decrease does not mean the absolute waveform prediction improves. These findings agree with the previous 240 m directional audit: replacing the travel-time hint alone does not resolve the depth-dependent waveform problem in this development region.

The old depth sweep showed that a naive 5%-of-each-trace-peak arrival picker could trigger on neural output before the source origin. Applying the same picker here, among **142 usable FD station/event pairs**, 99 learned-head and 103 FMM-head predictions cross threshold before origin minus 0.10 s; 72 and 70 respectively cross at the first sample (t=0). The corresponding FD pre-origin signal is zero in the archived audit. Mean neural pre-origin RMS is 0.231 and 0.228 times the respective full FD-record RMS. Raw paired-pick RMSE averages 1.262 s and 1.292 s, but these numbers are dominated by noncausal background and must **not** be interpreted as physical first-arrival travel-time error or a reliable ranking of travel-time solvers.

Constructing 25 reciprocal FMM fields again took 36.91 s and cached 181.1 MiB; this is repeated setup work, not per-event inversion cost. One seed, one velocity model, two anchors, reused development observations, no PDE/IC/BC constraints in the trained heads, 80 m FMM spacing and up to 60 m receiver snapping limit this conclusion. No continuous source-coordinate gradient passes through the FMM lookup. The remaining development target is a causal, source-sensitive wavefield operator, evaluated first on waveform displacement response and only later in a preregistered localization test on untouched observations. These numbers do not revise the paper's localization claims.

Inputs and per-record metrics: `fmm_receiver_head_depth_audit_protocol.json` and `fmm_receiver_head_depth_audit_results.json`. Run from the project root with `python examples/audit_fmm_receiver_head_depth.py`; the script verifies input hashes and refuses to overwrite the existing result.
