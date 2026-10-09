# Independent five-member ensemble confirmation

This experiment was executed once under the protocol in
`source_conditioned_ensemble_confirmation_protocol.json`. The protocol fixed
the eight sources, five checkpoint hashes, velocity-model hash, inversion
settings, FMM settings, interval definition, and acceptance gates before the
FD observations were generated. No model or threshold was changed after the
results were inspected.

## Event results

| Event | Ensemble-mean location error (m) | Origin-time error (s) | Ensemble waveform relative L2 | FMM location error (m) |
|---|---:|---:|---:|---:|
| SW shallow | 585.3 | 0.0534 | 0.986 | 495.0 |
| SE deep | 644.3 | 0.0097 | 0.888 | 336.4 |
| NW deep | 312.8 | 0.0033 | 0.971 | 361.6 |
| NE shallow | 523.3 | 0.1399 | 0.920 | Failed: 4 usable picks |
| West deep | 1561.7 | 0.2038 | 1.093 | 169.7 |
| East shallow | 1013.8 | 0.1709 | 0.920 | 160.7 |
| North mid-depth | 1705.3 | 0.0521 | 0.917 | 526.0 |
| Center deep | 862.4 | 0.0090 | 0.540 | 92.2 |

The ensemble-mean location error was 901.1 m on average and 753.3 m at the
median. One of eight events was below 500 m. The mean and median origin-time
errors were 0.0803 s and 0.0527 s; three of eight events were below 0.05 s.
For the seven events with enough first-arrival picks, FMM obtained mean and
median location errors of 305.9 m and 336.4 m. On those same seven events, the
PINN ensemble median was 1013.8 m.

## Nominal interval audit

The intervals are member mean plus or minus 1.96 sample standard deviations.
They describe variation over five training seeds and are not calibrated
confidence intervals.

| Component | Covered events | Empirical coverage | Mean width | Median width |
|---|---:|---:|---:|---:|
| x | 7/8 | 87.5% | 2510.0 m | 2128.8 m |
| y | 6/8 | 75.0% | 1782.7 m | 1952.3 m |
| z | 5/8 | 62.5% | 1143.5 m | 1070.2 m |
| Origin time | 6/8 | 75.0% | 0.312 s | 0.287 s |

## Predeclared gates

| Gate | Result |
|---|---|
| At least 6/8 ensemble locations below 500 m | Failed (1/8) |
| At least 6/8 origin-time errors below 0.05 s | Failed (3/8) |
| Ensemble median location error below 500 m | Failed (753.3 m) |
| Ensemble median location error below FMM median | Failed: one FMM event was unavailable; on the matched seven, 1013.8 m vs 336.4 m |
| Every component covers at least 6/8 events | Failed (z covers 5/8) |
| Mean x interval width below 2000 m | Failed (2510.0 m) |
| Mean y interval width below 2000 m | Passed (1782.7 m) |
| Mean z interval width below 1000 m | Failed (1143.5 m) |
| Mean origin-time interval width below 0.30 s | Failed (0.312 s) |
| Every FMM event has at least six usable picks | Failed (one event has four) |

The experiment therefore did not pass the joint predeclared criterion.

## Measured cost

- Eight FD observations: 68.44 s total.
- Forty member-event inversions: 145.20 s total, or 3.630 s per member-event.
- FMM receiver-field setup: 38.93 s.
- Eight FMM searches: 0.557 s total.

These results apply to one noise-free 5-Hz synthetic Overthrust model and one
25-receiver geometry. The ensemble varies training randomness only and does
not represent velocity, acquisition, source-mechanism, or model-form
uncertainty.
