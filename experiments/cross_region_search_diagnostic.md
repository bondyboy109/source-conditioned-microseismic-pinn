# Cross-region PINN search diagnostic (development data)

This diagnostic uses the already-inspected `region_04` and `region_07`
observations and the same frozen checkpoint as the registered cross-region
comparison. It is **not a new validation**. The original search settings and
model weights were unchanged. We compared the original five-start search
with 20 starts on the original grid and 20 starts on a denser 17 x 17 x 11 x 9
grid. Receiver and time subsampling, refinement steps and optimizer settings
were held fixed. SHA-256 checks verified the checkpoint and observations.

| Event | Search | Location error (m) | Origin-time error (s) | Final normalized MSE | Search time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| region_04 | Original, 5 starts | 485.5 | 0.0512 | 0.7777 | 3.81 |
| region_04 | Original grid, 20 starts | 485.5 | 0.0512 | 0.7777 | 13.84 |
| region_04 | Dense grid, 20 starts | 522.6 | 0.0441 | 0.7793 | 15.05 |
| region_07 | Original, 5 starts | 1855.2 | 0.2535 | 0.8841 | 3.71 |
| region_07 | Original grid, 20 starts | 1855.2 | 0.2535 | 0.8841 | 12.78 |
| region_07 | Dense grid, 20 starts | 1373.2 | 0.0600 | 0.7681 | 15.12 |

Adding starts alone selected the same solution in both events and did not
improve their objective. The denser search found a lower-MSE solution for
`region_07`, reducing location error by 482 m, but its remaining 1.37 km
error is still unacceptable for precise localization. For `region_04`, the
dense-grid result had a slightly higher final loss and 37 m worse position
error. Thus coarse-grid resolution contributes to the `region_07` failure,
but it does not explain the full cross-region error.

As a diagnostic only, we also started local optimization at each known true
source **after both truth-independent searches had finished**. These are
oracle-initialized local-basin checks, not inversions and not valid location
estimates. For `region_04`, normalized MSE decreased from 0.4233 at truth to
0.3950 while the result moved 71.2 m away, mostly in depth. For `region_07`,
MSE decreased from 0.6001 to 0.4909 while the result moved 139.9 m away,
also mostly in depth. The optimizer therefore improves its waveform objective
by moving away from the generating source. This is direct evidence of
forward-model/objective mismatch; it cannot be fixed by a denser global search
alone. `region_08` also had raw waveform relative L2 1.2627 at truth in the
registered comparison, independently indicating poor cross-region forward
accuracy for at least part of the domain.

## Interpretation

Two effects are present: a denser initial grid can help one event avoid a
poor basin, but the fixed PINN waveform operator also prefers non-true source
parameters on both tested events. The current comparison does not quantify
their contributions across the full domain, and it does not establish a
general accuracy rate from two selected diagnostic cases. All eight
cross-region records have already been inspected, so any model/search changes
must be assessed on newly generated, predeclared events.

Selected coarse starts and all refined endpoints are in
`cross_region_search_diagnostic.json`. Reproduce from the project root with:

```powershell
& 'E:\soft\Anaconda3\envs\torch_env\python.exe' examples\diagnose_cross_region_search.py
```
