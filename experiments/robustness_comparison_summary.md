# Robustness comparison summary

The repository already contains frozen stress tests under additive noise,
receiver removal, and an independently generated layered--undulating model.
The same hybrid, waveform-only, and FMM branches were evaluated without
retuning the checkpoint.

## Noise and receiver loss

The complete stress result is
`experiments/traveltime_aligned_moe_stress_results.json`. It contains 120
trials across 20, 10, and 5 dB noise and 20% and 40% receiver removal. At 20
and 10 dB, the hybrid pipeline had no failures. At 5 dB, 11 of 24 trials
failed because too few usable picks remained. Receiver removal caused 3 of 24
failures at 20% removal and 7 of 24 at 40% removal.

The hybrid branch remained substantially better than waveform-only under the
tested noise conditions, while FMM was generally more accurate in the clean
and moderate-noise cases. At 5 dB all branches shared failures caused by pick
availability; this is a data-quality limitation rather than evidence that the
hybrid is noise-proof.

## Independent geology

On four events from an independently generated layered--undulating model, the
median location errors were:

| Branch | Median `eloc` (m) | Mean `eloc` (m) |
|---|---:|---:|
| Waveform-only | 1034.1 | 938.5 |
| Hybrid | 226.3 | 282.1 |
| FMM | 86.2 | 105.5 |

The hybrid therefore retained a substantial advantage over waveform-only under
this model mismatch, but it did not surpass FMM. These results support
robustness of the repaired interface, not universal generalization.
