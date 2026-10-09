# One-time forward confirmation of the low-weight midpoint-teacher model

This is the frozen forward-only confirmation registered in `midpoint_teacher_confirmation_protocol.json`. The selected checkpoint (`teacher_0.002_loadable.pth`) and the receiver-only control were fixed before generating eight new 5 Hz second-order FD records. No source inversion or hyperparameter adjustment was performed.

The four `local` events are within 250 m of a training source (nearest distance 138.6 m). The four `remote` events are at least 1.3 km from every training source. Each record has 25 receivers, 1001 samples over 2 s, 40 m FD spacing, CFL 0.534, and six zero-Dirichlet faces.

| Event | Stratum | Control full L2 | Selected full L2 | Selected improves? |
|:--|:--|--:|--:|:--:|
| local_01 | local | 0.270 | 0.224 | yes |
| local_02 | local | 0.318 | 0.215 | yes |
| local_03 | local | 0.354 | 0.327 | yes |
| local_04 | local | 0.210 | 0.188 | yes |
| remote_01 | remote | 1.048 | 1.071 | no |
| remote_02 | remote | 0.961 | 1.001 | no |
| remote_03 | remote | 1.138 | 1.142 | no |
| remote_04 | remote | 0.860 | 0.890 | no |

The selected model has mean full L2 `0.632`, median `0.609`, and improves exactly 4 of 8 events. Its local mean is `0.238`, while its remote mean is `1.026`. The control mean over all eight events is `0.645`. The local improvement is therefore reproducible on new nearby sources, but it does not transfer to remote source locations; all four remote records are at or above the zero-prediction relative-L2 baseline of 1.0 except one control case.

The predeclared forward gate fails for three reasons: not all selected-event errors are below 1.0, the remote mean exceeds 0.85, and the selected model does not improve at least five of eight events. The failed gate means this checkpoint must not be used for a new localization claim. The result supports a narrower conclusion: midpoint FD supervision improves local interpolation within the training-source neighbourhood, but the current source-conditioned operator still lacks reliable global source generalization.

The eight FD files are now spent confirmation data and must not be reused as untouched tests after further model changes. Full hashes and per-event metrics are in `midpoint_teacher_confirmation_results.json`; the frozen design and thresholds are in `midpoint_teacher_confirmation_protocol.json`.
