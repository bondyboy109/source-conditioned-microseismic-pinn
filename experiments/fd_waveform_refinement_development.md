# FMM-initialized local FD waveform refinement: development pilot

This pilot uses already-inspected events. It measures feasibility and cost; it is not an independent or global FD inversion.

| Event | FMM error (m) | Local FD error (m) | FMM origin error (s) | Local FD origin error (s) | FD candidates | FD runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| spatial_01 | 225.2 | 103.5 | 0.0004 | 0.0020 | 27 | 278.3 |
| spatial_02 | 247.0 | 101.1 | 0.0256 | 0.0020 | 27 | 278.2 |
| **Mean** | **236.1** | **102.3** | **0.0130** | **0.0020** | -- | **556.5** |

The FD branch evaluates physical forward solves around the frozen FMM estimate and selects origin time by a zero-padded discrete delay. It fits no waveform gain. The timing excludes the preceding FMM field setup and search.

Already-inspected events; development cost and feasibility pilot only. FMM supplies the local center. This is not global FD inversion, an independent test, or evidence of superiority.
