# Search audit for source-conditioned waveform inversion

The inversion objective is now centralized in `waveform_mse`: predicted and
observed amplitudes are divided once by the full-record observation RMS. A
unit test checks the resulting objective and source gradient. The local
optimizer now uses projected Adam directly in the bounded normalized source
box, avoiding logit saturation at coarse-grid boundaries. The coarse search
can refine multiple lowest-misfit starts (`--restarts`).

Using the frozen 3500 m/s receiver-trained checkpoint, the three development
events were searched on a `6 x 6 x 4 x 4` grid, followed by 200 projected
Adam steps from each of the four best grid candidates. The true sources were
not used to initialize or select candidates.

| Event | Final location error (m) | Origin-time error (s) | Final normalized MSE | True-source normalized MSE |
| ---: | ---: | ---: | ---: | ---: |
| 250 m | 788.8 | 0.1313 | 0.6921 | 0.5747 |
| 800 m | 546.7 | 0.0793 | 0.7240 | 1.2034 |
| 2000 m | 1139.7 | 0.0866 | 0.9791 | 1.1240 |

Compared with the single-start run, multi-start refinement does not remove the
kilometre-scale error. For the 250 m event, the true-source misfit remains
lower than the selected estimate, so the optimizer still fails to recover the
best known candidate. For 800 and 2000 m, a non-true source has a lower model
misfit than the true source, which indicates forward-model mismatch or
waveform non-uniqueness rather than a simple grid-search failure.

These results close the searcher audit but do not pass the localization gate.
The method should not report the earlier 91.642% accuracy or calibrated UQ
from this checkpoint. Raw outputs are in
`outputs/development_inversion_restarts_01.json` through `_03.json`.
