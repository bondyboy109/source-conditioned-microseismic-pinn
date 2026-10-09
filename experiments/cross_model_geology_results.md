# Cross-geology zero-shot audit

The frozen Overthrust-trained MoE was evaluated without retraining on an independent layered model.

| Metric | Waveform only | Hybrid | FMM baseline |
|---|---:|---:|---:|
| Median location error (m) | 1034.1 | 226.3 | 86.2 |
| Mean location error (m) | 938.5 | 282.1 | 105.5 |
| Median origin-time error (s) | 0.0927 | 0.0247 | 0.0327 |
| Events below 500 m | 1 | 3 | 4 |

FMM field setup: 38.89 s.

Zero-shot transfer audit only; no retraining and no claim of cross-geology generalization.

The Overthrust-trained waveform surrogate is evaluated on a distinct synthetic geology; the FMM fields use the independent model, while picker bias and search settings remain frozen from Overthrust development.
