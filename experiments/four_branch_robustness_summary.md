# Four-branch robustness summary

The main clean confirmation and independent-geology results are summarized
below. The constant-velocity branch uses the separately trained `vavg=5000`
control and is reported as a diagnostic because its search and training
protocol are not identical to the registered hybrid protocol.

| Dataset | Waveform-only median (m) | Constant-velocity median (m) | Hybrid median (m) | FMM median (m) |
|---|---:|---:|---:|---:|
| Clean Overthrust confirmation (8) | 2400.2 | not rerun under the registered clean protocol | 362.9 | 276.8 |
| Independent layered model (4) | 1034.1 | 4984.2 | 226.3 | 86.2 |

The constant-velocity stress diagnostic used the same 15 perturbation
conditions as the registered stress test. On successful trials its median
location errors were approximately 6.4--6.6 km at 20 and 10 dB, and
5.2--8.9 km at 5 dB. Pick failures occurred in 11 of 24 trials at 5 dB.
With 20% receiver removal, 3 of 24 trials failed and successful-trial medians
were 5.5--5.6 km. With 40% removal, 7 of 24 trials failed and successful-trial
medians were 7.5--9.6 km.

These results provide a strong physical control for replacing a constant-speed
shift with heterogeneous FMM traveltimes. They should not be used to claim
that the hybrid learned surrogate is more accurate than FMM: on the clean and
cross-geology sets, FMM remains the more accurate branch.
