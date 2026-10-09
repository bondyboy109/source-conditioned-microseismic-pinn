# Why the cross-region inversion returns wrong sources: development audit

This audit leaves the model weights, the global search protocol, and the
inversion loss unchanged. It examines four **previously inspected** events
using the fixed 25-source and 29-source 5 Hz checkpoints. Predictions are
evaluated at the known true source and at the respective search output using
the same 25 receivers, every fourth time sample, and the same full-record
observation RMS as the inverse objective. Receiver and time-window loss parts
sum to the actual normalized MSE. No truth is supplied to either search.

| Event | 25-source MSE: true / returned | 29-source MSE: true / returned | Closest original coarse-grid candidate rank (of 5808; 29-source model) |
| --- | ---: | ---: | ---: |
| region_04 | 0.423 / 0.778 | 0.235 / 0.880 | 1320 |
| region_07 | 0.600 / 0.884 | 0.149 / 0.877 | 5350 |
| region_08 | 1.591 / 0.625 | 0.672 / 0.887 | 17 |
| central_gap | 0.865 / 0.770 | 0.097 / 0.313 | 5806 |

The control model's waveform loss **actually favors the wrong returned
source** over truth for `region_08` and `central_gap`, consistent with its
large forward error there. The 29-source model reverses this ordering for
all four events: it fits the FD data **better at truth** than at the returned
source. Yet its unchanged 5-start search still misses those better locations.
Hence more FD supervision partially repairs the forward objective locally,
while the selection/optimization step remains a separate obstacle.

The initial grid is only 11 x 11 x 8 x 6 across the whole 4-D search box.
For the augmented model the closest grid candidate to `region_04` is offset
by (-100, +550, 0) m and +0.04 s; the candidate ranks 1320th by the coarse
loss. The offsets and ranks are (+40, -550, -80) m / +0.03 s, rank 5350 for
`region_07`; (+150, +210, 0) m / +0.04 s, rank 17 for `region_08`; and
(+150, -500, +50) m / -0.02 s, rank 5806 for `central_gap`. None is among
the five selected starts. At `central_gap`, MSE is 0.097 at the continuous
truth but 3.802 on its nearest **coarse-sampled** grid candidate (2.289
when evaluated with the refinement sampling). This sharp change means a
coarse nearest candidate is not a reliable stand-in for a good continuous
source, especially when origin time and position both change. Ranking this
one nearest point does not prove no other grid candidate reaches the correct
basin; the completed searches show that their selected five did not.

The error is not confined to a single outlier trace. For the 29-source model
at `region_07`, the returned-minus-true loss is +0.727: the 0.5--1.0 s window
accounts for +0.397, and receiver indices 14, 13 and 18 contribute +0.194,
+0.177 and +0.076. At `region_04`, the 0.5--1.5 s windows contribute +0.490
of the +0.646 gap. These are decompositions of the **existing** full-waveform
loss, not alternative weights. Since the wrong sources were selected on the
same observations, these component comparisons are descriptive and cannot
validate a new window or station weighting.

## Decision

Do not infer that the 29-source model fixes cross-region localization from
its lower forward L2. First develop a truth-independent candidate generator
and an initial score evaluated consistently with the refinement objective,
then test whether it can find the low-loss basin without inserting the true
source. An FMM first-arrival proposal is a diagnostic option but uses the
known velocity and must be costed and labeled as a hybrid baseline. Existing
eight cross-region events and four new development events cannot become
fresh confirmation after search changes; preregister separate FD sources
only when the development method is fixed. Full-domain and field accuracy
remain unestablished.

Machine-readable audit: `cross_region_objective_audit.json` and
`cross_region_coarse_ranking.json`. Reproduction scripts:
`examples/audit_cross_region_objective.py` and
`examples/audit_cross_region_coarse_ranking.py`.
