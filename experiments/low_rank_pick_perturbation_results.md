# Pick perturbation audit for the frozen low-rank method

Eight synthetic events have been used in prior arrival-picker development, so this is a post-hoc stress audit. It uses one Overthrust model and one receiver geometry. Perturbations are independent Gaussian timing errors applied to the same robust picks for all branches.

| Pick-error SD (samples) | Trials | Waveform median eloc (m) | Hybrid median eloc (m) | FMM median eloc (m) |
|---:|---:|---:|---:|---:|
| 0 | 24 | 918.0 | 546.5 | 175.2 |
| 1 | 24 | 918.0 | 544.9 | 175.2 |
| 2 | 24 | 918.0 | 544.5 | 198.7 |
| 5 | 24 | 918.0 | 533.5 | 205.7 |

Each row contains 24 trials (8 events x 3 seeds). The waveform-only branch
completed all 24 trials at every perturbation level. The hybrid and pure-FMM
branches completed 21/24 trials at every level; the remaining three trials
correspond to `fresh_08`, which has only five valid picks in the 2.0-s model
window and therefore fails the predeclared six-pick minimum. On the 21 paired
successful trials, hybrid improved the location error relative to waveform-only
in 18/21 trials at 0-sample perturbation and 18/21 at 5-sample perturbation.
Pure FMM remained more accurate than hybrid in 18/21 and 16/21 paired trials,
respectively.

Cached FMM field access: 0.06 s (the FMM field was constructed and cached before this run).

The same perturbed picks were supplied to all branches. This is a post-hoc stress audit and not a prospective blind test.
