# Four-source cross-region coverage pilot: development only

This pilot follows `cross_region_coverage_development_protocol.json`,
registered before generating the four added training and four new development
FD events. It responds to failures already observed in the original eight
cross-region events. None of the new event coordinates had appeared in the
archived data or the fixed 25-source checkpoint. The control checkpoint SHA-256
was verified; the augmented checkpoint lists precisely the original 25 plus
four added training records. Both models use the same architecture, seed 42,
two-stage update counts and loss settings, but the 29-source model uses more
FD supervision and proportionately more receiver samples per update.

| New development event | Control 25 raw waveform L2 | Augmented 29 raw waveform L2 |
| --- | ---: | ---: |
| dev_r04 | 0.8103 | 0.4899 |
| dev_r07 | 0.7842 | 0.5291 |
| dev_r08 | 1.3009 | 0.6854 |
| dev_center | 0.9398 | 0.5194 |
| **Mean** | **0.9588** | **0.5559** |

All four improve; the arithmetic mean falls by 42.0%, and all augmented
errors are below 0.75. Thus all three *development forward-error* gates pass.
These new sources are close to newly added training sources selected in
response to previous failures. This establishes local improvement under
additional FD supervision, not full-domain generalization or a better
localization method.

To check whether the forward gain helps the stated localization goal, we ran
the **original global 5-start search unchanged** on the already-inspected
eight cross-region FD records, using the fixed new 29-source checkpoint. This
is post-hoc development analysis, never an independent repeat of the original
test or a replacement for its failed gates.

| Previously inspected event | Control 25 location error (m) | Augmented 29 location error (m) |
| --- | ---: | ---: |
| region_02 | 27.4 | 23.6 |
| region_03 | 52.9 | 79.5 |
| region_04 | 485.5 | 579.4 |
| region_05 | 96.6 | 125.2 |
| region_06 | 136.7 | 87.4 |
| region_07 | 1855.2 | 1391.5 |
| region_08 | 721.6 | 1804.1 |
| central_gap | 763.8 | 395.5 |
| **Median** | **311.1** | **260.3** |

Both models place only five of these eight within 500 m; the mean location
error actually rises from 517.4 to 560.8 m. In particular, `region_08`
becomes much worse despite improvement on the nearby new development waveform.
This prevents interpreting lower forward L2 near extra training sources as
reliable cross-region inversion. Training-source coverage alone does not
resolve the observed non-identifiability or optimization/forward-model
interaction; the experiment does not isolate these causes.

Generating four additional FD *training* waveforms cost 34.50 s; the 29-source
model took 27.47 s for 5000 receiver-only updates and 47.83 s for 1000 hybrid
updates on the existing GPU (75.30 s combined). Four development FD records
and evaluation have further costs, not included in that training total. The
gain is therefore not a demonstrated end-to-end acceleration.

Do not move to a one-time independent localization test claiming success on
the basis of these forward metrics. Investigate why the inversion objective
prefers incorrect positions, define a location-relevant development gate and
only then preregister fresh sources for a final confirmation. Existing eight
cross-region events and these four development events have been inspected.
The synthetic one-velocity, one-seed, 5 Hz setting still has no noise/field/UQ
validation.

Raw results: `cross_region_coverage_development_results.json`,
`cross_region_coverage_localization_development.json`, and
`cross_region_coverage_{control25,aug29}_forward.json`. The development runner
is `examples/run_cross_region_coverage_development.py` and the localization
audit is `examples/audit_coverage_localization_development.py`.
