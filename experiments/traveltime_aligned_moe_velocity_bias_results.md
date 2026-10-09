# Frozen localization under uniform velocity bias

FD observations use the original Overthrust velocity. Both neural travel-time features and the FMM baseline use the biased inversion velocity.

| Inversion velocity | Trials | Waveform median (m) | Hybrid median (m) | Hybrid median time (s) | FMM median (m) |
|---|---:|---:|---:|---:|---:|
| 90% of truth | 8 | 1560.0 | 442.6 | 0.1171 | 254.9 |
| 95% of truth | 8 | 1171.2 | 328.8 | 0.0494 | 212.3 |
| 105% of truth | 8 | 1503.1 | 361.1 | 0.0139 | 230.1 |
| 110% of truth | 8 | 1107.1 | 289.9 | 0.0496 | 378.8 |

Runtime: 346.6 s.

Limitations: The experiment applies a spatially uniform multiplicative velocity bias to the same Overthrust model and eight clean prospective events. It measures bulk calibration sensitivity, not transfer to independent geology, local structural error, anisotropy, attenuation, or field data.
