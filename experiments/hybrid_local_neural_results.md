# Fixed local-interpolation/neural hybrid audit

This is a post-hoc development audit on 16 records that were already opened in
earlier experiments. It does not constitute independent validation and it does
not support a localization or uncertainty-quantification claim.

The local branch uses the configuration fixed by the earlier local scan:
two nearest FD training records, inverse-distance-squared weights, a 1000 m
horizontal scale, and a 50 m depth scale. The hybrid selects the local branch
only when its nearest scaled source distance is at most 2.0; otherwise it uses
the frozen neural checkpoint. No parameters were selected from these 16 target
records.

| Forward surrogate | Mean full-record L2 | Mean active-sample L2 |
|:--|--:|--:|
| Local FD interpolation | 1.2323 | 1.0585 |
| Frozen neural checkpoint | **1.0493** | **0.9614** |
| Fixed hybrid | 1.1123 | 0.9891 |

The hybrid selected the local branch for 5 of 16 records and improved over the
neural branch for 0 of 16 records. The local interpolation branch is useful only
inside the narrow source neighbourhood where the archived FD records bracket a
target. Applying it as a fixed fallback to the broader evaluation set made the
mean error worse than the neural model.

This result closes the simple fallback route. The current model still lacks a
forward operator accurate enough for a new localization experiment. The next
structural change must model source-region dependence explicitly, for example
with separate local experts or a neural residual correction around a local FD
interpolant, followed by a newly registered forward test only after a predefined
error gate is passed.

Full per-event metrics, neighbours, hashes, and the fixed protocol are in
`hybrid_local_neural_results.json` and `hybrid_local_neural_protocol.json`.
