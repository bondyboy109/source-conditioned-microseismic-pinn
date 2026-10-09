# Frozen eikonal-carrier independent test

This is the first evaluation on these eight FD events. The source coordinates, checkpoint hashes, search settings and success thresholds were fixed in `eikonal_independent_protocol.json` before generating the new records. None of the eight spatial source positions coincides with the 140 older FD observations. This test uses the same Overthrust velocity and numerical source setup as development, so it is independent of the evaluated events, not of the underlying geology or simulator.

## Results

All locations are in metres; origin-time errors are in seconds. Raw L2 compares the neural prediction at the **true source** with the full 2 s FD receiver record, without fitted gain or time shift. PINN searches do not receive the true source. The eikonal hybrid receives the FMM estimated position as its initialization; FMM and the straight-carrier control each use their own fixed global search.

| Event | FMM `eloc` | FMM `et` | Straight `eloc` | Straight `et` | Straight L2 | Eikonal hybrid `eloc` | Hybrid `et` | Eikonal L2 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 01 | 366.5 | 0.1200 | 471.6 | 0.0352 | 0.997 | 390.8 | 0.0413 | 0.943 |
| 02 | 512.3 | 0.1060 | 1111.0 | 0.1662 | 0.937 | 605.5 | 0.0027 | 0.927 |
| 03 | 193.6 | 0.1083 | 2015.1 | 0.0508 | 1.023 | 952.2 | 0.1090 | 1.086 |
| 04 | 220.9 | 0.0899 | 1449.8 | 0.0354 | 1.150 | 277.5 | 0.0900 | 1.104 |
| 05 | 195.3 | 0.0919 | 2899.2 | 0.1400 | 1.045 | 152.9 | 0.0099 | 0.837 |
| 06 | 174.4 | 0.1136 | 3697.0 | 0.1817 | 0.659 | 12.9 | 0.0034 | 0.609 |
| 07 | 249.8 | 0.0489 | 835.7 | 0.1080 | 0.775 | 628.5 | 0.1182 | 1.016 |
| 08 | 490.4 | 0.0624 | 964.1 | 0.0824 | 0.782 | 173.5 | 0.0205 | 0.759 |
| **Median location** | **235.4** | | **1280.4** | | | **334.2** | | |

The eikonal hybrid has five of eight locations below 500 m, five of eight origin times within 0.05 s, and improves on FMM in only three of eight events. Three of eight true-source eikonal predictions have raw relative L2 below 0.85. Mean raw L2 changes only from 0.921 for the straight carrier to 0.910 for the eikonal carrier. All 8 FMM events have at least six usable first-arrival picks (7 to 12). Thus **five of six preregistered checks failed**; only the FMM pick count passed.

## Cost and interpretation

On the tested GPU, the 25 reciprocal FMM receiver fields took 38.36 s to set up and occupied 181.1 MiB; their per-event search took 0.055-0.095 s. Loading the frozen eikonal and straight models took 0.089 s and 0.010 s, respectively. The straight model's global waveform inversion took 3.97-5.10 s per event. Eikonal hybrid waveform proposal scoring took about 0.96 s and local refinement about 10.67 s **per event**, plus the FMM setup and per-event FMM search. FD synthesis for each test record took about 9 s and is a data-generation cost, not part of the inverse runtime. These times do not include the one-time training cost of either neural operator.

The hybrid is substantially better than the straight-carrier global search by median location error but is slower and less accurate by median than its FMM initializer. In particular, events 03 and 07 move far from useful FMM starting positions. This independent test does **not** support a claim that the eikonal-carrier PINN improves localization over FMM or delivers a computational speedup. The raw waveform L2 remains high for most events; low training loss or favorable development cases cannot replace this held-out assessment. Do not retune on these eight events and call a second evaluation independent.

The machine-readable per-event source estimates, component errors, pick counts and timings are in `eikonal_independent_results.json`; the FMM estimates and exact field-setup settings are in `eikonal_independent_fmm.json`. The one-time protocol is `eikonal_independent_protocol.json`.
