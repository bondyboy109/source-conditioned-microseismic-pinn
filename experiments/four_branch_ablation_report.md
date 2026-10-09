# Four-branch localization ablation

This report records the locked candidate-pool comparison used to separate the
contributions of waveform fitting and heterogeneous traveltime alignment. All
three branches use the same eight finite-difference events, receiver geometry,
candidate pool, frozen source-conditioned MoE, and search budget.

The archived result is
`experiments/refined_fmm_moe_diagnostic_results.json`.

| Branch | Mean `eloc` (m) | Median `eloc` (m) | Mean `et` (s) |
|---|---:|---:|---:|
| Waveform-only MoE | 4477.1 | 3599.5 | 0.1092 |
| Waveform + FMM arrival residual | 246.7 | 227.4 | 0.0184 |
| Refined FMM arrival baseline | 176.6 | 172.8 | 0.0142 |

The hybrid objective improved the location error relative to waveform-only for
all eight events, but selected the same candidate as refined FMM for only one
event and was better than refined FMM for three events. The result supports the
following computational claim: heterogeneous FMM traveltime alignment and an
arrival residual repair the spatially ill-conditioned waveform-only learned
interface. It does not support a claim that the learned surrogate is more
accurate than FMM.

## Scope

This is a fixed-rule diagnostic on previously inspected development events. It
is not an independent generalization test, and it does not isolate an
average-velocity shift branch. A future four-branch confirmation should add
that branch before any claim about replacing FMM is made.
