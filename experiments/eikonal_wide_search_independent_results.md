# Frozen 29-source PINN: independent wide-search evaluation

The eight source positions and the input hashes were recorded in
`eikonal_wide_search_independent_protocol.json` before generating any of these
FD observations. Preflight compared the registered positions with 172 previous
observation files and the 29 checkpoint training entries; none was within
100 m of a registered test position. These are new events in the **same**
Overthrust velocity model, FD simulator, receiver geometry and 5 Hz setup.
The wider search was developed on different, already inspected observations.

Both inversions used the same new FD waveforms. FMM used its previously fixed
first-arrival picker, reciprocal receiver fields and search settings. The
frozen PINN used the FMM location as its search center, the fixed seven offsets
per spatial axis, 46 origin times, 20 selected starts and 250 Adam steps. The
inversion code saved all FMM and PINN estimates in
`eikonal_wide_search_independent_estimates.json` **before** opening the FD
source truth for error calculations. One event did not meet the fixed six-pick
FMM threshold; it remains a failed event, and the dependent PINN search was
not run for it.

| Event | Valid FMM picks | FMM location error (m) | PINN location error (m) | PINN origin-time error (s) | PINN raw waveform L2 at truth |
|:--|--:|--:|--:|--:|--:|
| 01 | 8 | 769.6 | 436.5 | 0.1157 | 1.261 |
| 02 | 8 | 232.6 | 542.3 | 0.0325 | 0.866 |
| 03 | 8 | 339.4 | 955.0 | 0.2146 | 1.039 |
| 04 | 5 | failed | failed | failed | 1.124 |
| 05 | 14 | 110.5 | 1169.0 | 0.1587 | 1.135 |
| 06 | 10 | 134.7 | 814.1 | 0.1427 | 1.030 |
| 07 | 15 | 192.7 | 340.8 | 0.0718 | 1.030 |
| 08 | 12 | 410.7 | 289.0 | 0.0173 | 1.236 |
| **Median of 7 successful searches** | | **232.6** | **542.3** | | |

Only 3/8 PINN locations were within 500 m, 2/8 origin times were within
0.05 s, and PINN improved on FMM for 2/8 events. All five predeclared checks
failed, including the requirement that all eight events have at least six
usable FMM picks. The median raw relative waveform L2 at the true source was
1.082; it is a forward-model diagnostic, not a location error. Per-event
estimates, component errors and failure statuses are in the machine-readable
results file. Median per-event search times among successful events were
0.076 s for FMM and 27.13 s for the PINN search. The PINN additionally
requires FMM initialization: receiver-field setup took 37.10 s and occupied
181.1 MiB. FD synthesis took a median of 8.59 s per event and is excluded
from inverse search time; neural training cost is also excluded.

The independent result does **not** establish a localization gain or runtime
advantage over the FMM baseline for this frozen model. Larger candidate
coverage fixed a previously inspected development event but did not
generalize reliably to these new positions. In particular, event 04 shows
that the hybrid inherits FMM picker failures. Do not tune the search on these
eight events and subsequently describe a rerun on them as independent.
This synthetic test does not establish field robustness, noise tolerance,
uncertainty calibration or results at frequencies above 5 Hz. The raw records
are in `data/observations/eikonal_wide_search_independent/`; reproduce the
preflight and run with `examples/confirm_eikonal_wide_search_independent.py`.
