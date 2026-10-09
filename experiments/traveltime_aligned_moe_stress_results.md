# Frozen MoE noise and missing-receiver stress test

The checkpoint, FMM fields, arrival correction, hybrid weight, and search settings were frozen.

| Condition | Trials | Median picks | Waveform median (m) | Hybrid median (m) | FMM median (m) | Hybrid failures |
|---|---:|---:|---:|---:|---:|---:|
| noise:5.0 | 24 | 6.0 | 2401.8 | 710.9 | 549.2 | 11 |
| noise:10.0 | 24 | 8.0 | 2403.7 | 636.6 | 511.5 | 0 |
| noise:20.0 | 24 | 9.0 | 2390.4 | 547.4 | 597.7 | 0 |
| missing_receivers:20.0 | 24 | 7.5 | 1294.6 | 364.0 | 143.8 | 3 |
| missing_receivers:40.0 | 24 | 6.0 | 781.2 | 302.5 | 194.9 | 7 |

Runtime: 1051.4 s.

Limitations: This is a perturbation stress test of the same eight prospective source positions, velocity model, source mechanism, receiver geometry, and FD solver. The perturbations were specified after the clean confirmation, so the results test robustness but are not a second independent confirmation.
