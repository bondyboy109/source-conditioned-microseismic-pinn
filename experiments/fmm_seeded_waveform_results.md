# FMM-seeded waveform search: development and one-time confirmation

The proposal generator uses the registered reciprocal-FMM first-arrival
estimate for each event as a **spatial center**. It evaluates 27 nearby
spatial positions (offsets -320/0/+320 m on each axis) and 46 origin times
from 0.05 to 0.50 s, yielding 1242 candidates. The scorer uses all 25
receivers, every fourth time sample, and the same full-record RMS-normalized
MSE as local waveform refinement. It retains up to 12 spatially distinct
starts and runs the unchanged 250-step Adam refinement. The FMM origin-time
estimate is not used to narrow the time scan. Candidate generation, scoring
and refinement do not receive the event truth.

This is a **known-velocity FMM + trained PINN hybrid**, not a standalone
PINN result. The FMM estimates depend on picked first arrivals from the FD
records and 25 receiver-centered traveltime fields calculated from the true
heterogeneous velocity. The original global PINN search and standalone FMM
use their previously fixed settings for the comparisons below.

## Already-inspected development events

The old eight cross-region events had guided model development and search
diagnostics. On those seen records, the hybrid had a median location error
of 58.3 m versus 260.3 m for the 29-source standalone PINN search and
191.8 m for FMM. All eight hybrid errors were below 500 m. This motivated
one-time confirmation but **cannot** be reported as independent evidence.
Its 8 searches added 4.71 s of waveform scoring and 61.81 s of refinement,
on top of 36.88 s of FMM setup and 0.67 s of FMM per-event searches in total.

## New registered events: failed confirmation

The eight new FD source coordinates and six gates were written in
`fmm_seeded_independent_protocol.json` **before** generating their records.
No event source or file overlaps the checkpoint's 29 training events or any
earlier archived event. The model, velocity and search settings were frozen.
The wavefield L2 below is the unadjusted PINN prediction **at the true source**,
compared with FD receiver traces; it is not a search loss or a tuned metric.

| New event | FMM error (m) | PINN-only error (m) | Hybrid error (m) | Hybrid origin-time error (s) | PINN truth-source waveform L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| test_r04 | 158.9 | 854.2 | 106.7 | 0.0154 | 0.801 |
| test_r07 | 239.7 | 1924.6 | 87.2 | 0.0009 | 0.706 |
| test_r08 | 321.0 | 3573.7 | 667.3 | 0.1460 | 0.992 |
| test_center | 199.4 | 899.7 | 255.4 | 0.0622 | 0.813 |
| test_west | 275.4 | 1534.8 | 539.8 | 0.0389 | 1.049 |
| test_south | 242.1 | 71.8 | 71.8 | 0.0028 | 0.613 |
| test_east | 163.7 | 922.0 | 634.2 | 0.0918 | 0.793 |
| test_north | 358.9 | 1474.0 | 1136.1 | 0.0574 | 1.096 |
| **Median** | **240.9** | **1198.0** | **397.6** | -- | -- |

The FMM valid-pick gate passed (every event had at least six usable first
arrivals). **All five hybrid gates failed**: only 4/8 position errors were
below 500 m (required at least six); 4/8 origin-time errors were below
0.05 s (required six); its median was 397.6 m (required below 200 m and
below the FMM median of 240.9 m); and only 2/8 true-source forward errors
were below 0.75 (required six). The hybrid improved relative to standalone
PINN on several events but exceeded the standalone FMM error on five of
eight. These tests have now been inspected and cannot be re-used as untouched
confirmation for future search or model changes.

The new FMM receiver-field setup took 35.88 s with a 181.1 MiB field cache;
its eight event searches totaled 0.62 s. Hybrid waveform candidate scoring
added 4.59 s and PINN refinement added 58.00 s for eight events: 62.59 s
extra beyond the FMM costs, or about 99.09 s inclusive of FMM setup and
search for this batch. Standalone PINN-only searches took 30.47 s, excluding
its training. Generating the eight FD *test data* took 68.92 s and is shared
data production, not a search cost. The 29-source checkpoint additionally
required 75.30 s of measured training and 34.50 s for four added FD training
events, beyond the earlier 25-source dataset costs. There is **no demonstrated
speed or location-accuracy advantage** over standalone FMM in the new batch.

This test is still a single-seed, one-velocity-model, same-solver 5 Hz
synthetic evaluation: no noise, field data, multiple geology models or
uncertainty coverage. Full-waveform PINN and first-arrival FMM have different
information inputs. The next model-development issue is forward accuracy
outside the newly supervised neighborhoods; optimizing a surrogate with
truth-source L2 at or above 1 cannot establish reliable waveform localization.
Changing scores or gates on these events would be development, and any
subsequent accuracy claim would require further freshly registered sources.

Raw, auditable records:
`fmm_seeded_waveform_development_results.json`,
`fmm_seeded_independent_protocol.json`,
`fmm_seeded_independent_fmm.json`, and
`fmm_seeded_independent_results.json`. Scripts:
`examples/run_fmm_seeded_waveform_development.py` and
`examples/confirm_fmm_seeded_waveform.py`.
