# Prototype-gated MoE localization smoke test

The prototype-gated checkpoint was used in the same bounded coarse-to-fine
source search as the earlier generic MoE diagnostic, on holdout event 01 only.
This test checks differentiability and objective geometry; it is not a
publication-level localization evaluation.

| quantity | result |
|---|---:|
| true source `(x, y, z, ts)` | `(4200 m, 7200 m, 380 m, 0.180 s)` |
| estimated source | `(4874 m, 13710 m, 306 m, 0.363 s)` |
| location error | `6545 m` |
| origin-time error | `0.183 s` |
| objective at selected estimate | `1.0929` |
| objective at true source | `1.2648` |
| source-gradient finite | yes |

The operator provides finite gradients, but the waveform objective is still
lower away from the true source and the search converges near the upper `y`
and time bounds. The approximately 0.84% forward holdout improvement of the
prototype MoE therefore does not yet produce a reliable localization method.
The manuscript must not report this smoke test as localization accuracy.
