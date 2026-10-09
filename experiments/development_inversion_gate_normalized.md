# Development-only waveform inversion gate

The previously selected 3500 m/s receiver-trained checkpoint was frozen.
We evaluated only the three development events (`dev_distance_0250`,
`dev_distance_0800`, `dev_distance_2000`) with the four-dimensional waveform
inversion in `invert_source_conditioned.py`. The independent test events were
not consulted. The first pass revealed an inverse-objective scale mismatch:
training divided each event's waveforms by its full-record RMS, whereas the
inversion used physical amplitudes directly. This produced MSE around 1e-13,
effectively zero local updates, and misleading gradient thresholds.

The inversion now divides both observed and predicted amplitudes by the
observation's *full-record* RMS. The same fixed scale is used for every
candidate, including the true source. In a 4 x 4 x 3 x 3 coarse search plus
100 local steps, all four source-gradient components become finite and
nonzero, but location errors remain 3172, 2841, and 2026 m. A larger
6 x 6 x 4 x 4 grid with 200 local steps gives:

| Development offset | Recovered location error (m) | Origin-time error (s) | Recovered normalized MSE | True-source normalized MSE |
| ---: | ---: | ---: | ---: | ---: |
| 250 m | 770.4 | 0.2221 | 0.9400 | 0.5747 |
| 800 m | 546.7 | 0.0793 | 0.7240 | 1.2034 |
| 2000 m | 1187.1 | 0.0700 | 1.1274 | 1.1240 |

Even with a repaired objective and finer coarse grid, these errors fail the
localization gate. In the 250 m case the true-source misfit is **lower** than
the recovered misfit, indicating search/optimization failure. In the 800 m
case a wrong source attains a lower waveform misfit than the true source,
indicating forward-model misspecification or ambiguity. The 2000 m case is
similarly ambiguous. Reporting a percentage localization accuracy or
calibrated uncertainty from this forward model would be unsupported.

These are diagnostic, development-set results. The coarse grid is still
sparse, and the local optimizer's bounded logistic parameterization can
saturate at grid boundaries. No independent confirmation or multi-seed
inversion is claimed. Raw results are in
`outputs/development_inversion_normalized_fine_01.json` through `_03.json`.
