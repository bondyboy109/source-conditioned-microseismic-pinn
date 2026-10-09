# SNR=10 dB robust-picker hybrid refinement

At `SNR=10 dB`, two noisy realizations were generated for each of four
independent holdout events. A robust MAD onset picker supplied arrivals to the
same FMM coarse search used previously. A fixed local neighborhood was then
scored by the FMM arrival residual, the MoE waveform loss, and their fixed
hybrid combination.

| Selection | Mean location error (m) | Mean origin-time error (s) |
|:--|--:|--:|
| FMM coarse result | **1556** | **0.176** |
| Hybrid FMM + MoE waveform | **1556** | **0.176** |
| MoE waveform-only local choice | 3394 | 0.204 |

The hybrid criterion selected the same candidates as the FMM baseline in all
eight noisy realizations. The MoE waveform-only choice was worse. This confirms
that the current neural waveform surrogate is not yet a reliable noisy-data
refinement term, even though it improved the clean forward holdout error.

The result is a diagnostic, not a final robustness claim: one SNR, two
realizations per event, one checkpoint, and a coarse local grid were used.
