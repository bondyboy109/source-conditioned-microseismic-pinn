# Fixed-source amplitude and shape development audit

All predictions use the true source position and origin time, with no time shift. Full waveforms at identical receiver/time samples share the reference event RMS. Nonnegative gains are fitted afterward for attribution only.

| Path | Pair weight | Outcomes | Mean raw L2 | Mean global-gain L2 | Mean receiver-gain oracle L2 | Median predicted/reference energy |
|---|---:|---:|---:|---:|---:|---:|
| source_to_target | 0.00 | 48 | 0.8157 | 0.7777 | 0.7363 | 0.5651 |
| source_to_target | 0.05 | 48 | 0.8166 | 0.7782 | 0.7346 | 0.5479 |
| target_to_target | 0.00 | 48 | 0.8513 | 0.7887 | 0.7416 | 0.5963 |
| target_to_target | 0.05 | 48 | 0.8555 | 0.7893 | 0.7413 | 0.5532 |

Energy ratios use squared waveform norms, not RMS or L2-norm ratios. The receiver-gain oracle can absorb station-specific amplitude distortions; its remaining residual still contains waveform-shape and timing error. This audit alone cannot distinguish those two errors or attribute them to a specific architectural cause.

Previously inspected development events only. Gains are fitted against known reference waveforms at the true source. This is an oracle diagnostic, not blind localization, improved forward prediction or independent confirmation. No test or confirmation event is loaded.

Each group contains 3 seeds x 16 previously inspected development positions, not 48 independent geologies. No weights, models or localization settings change.
