# Eikonal-carrier development audit

The fixed 29-source model uses a 3500 m/s straight-ray travel-time carrier.
This development model keeps the same 29 training records, seed, network
width, loss weights and two-stage update counts, but replaces that carrier
with the frozen 5000-step Overthrust source-conditioned eikonal network
(`outputs/eikonal_overthrust_5000/eikonal.pth`). The eight `test_*` records
were already opened during the failed FMM-seeded confirmation, so this is a
post-hoc development audit, not an independent validation.

## Forward-model comparison

| Event | Straight-carrier L2 | Eikonal-carrier L2 | Straight shift/gain L2 | Eikonal shift/gain L2 |
| --- | ---: | ---: | ---: | ---: |
| test_center | 0.8133 | 0.6662 | 0.7449 | 0.6270 |
| test_east | 0.7925 | 0.6910 | 0.7701 | 0.6896 |
| test_north | 1.0956 | 1.0358 | 0.9876 | 0.9932 |
| test_r04 | 0.8006 | 0.8735 | 0.7920 | 0.8325 |
| test_r07 | 0.7059 | 0.6507 | 0.6885 | 0.5913 |
| test_r08 | 0.9916 | 1.0122 | 0.8491 | 0.8545 |
| test_south | 0.6134 | 0.6142 | 0.6054 | 0.5929 |
| test_west | 1.0491 | 0.9229 | 0.9515 | 0.8993 |
| **Mean** | **0.8577** | **0.8083** | **0.7986** | **0.7600** |

The eikonal carrier improves the raw L2 on 5/8 events and lowers the mean by
5.76%. It improves the shift/gain diagnostic on 5/8 events and lowers its
mean by 4.83%. The result is mixed: `test_r04` and `test_r08` become worse,
and `test_north` remains above 1.0. A heterogeneous first-arrival carrier
therefore helps some cross-region waveforms but does not remove the full-wave
forward error, including amplitude and late-time waveform differences.

## FMM-seeded localization development

Using the identical FMM-centered 1242-candidate search and 12-start refinement
from the previous development audit gives:

| Event | FMM location error (m) | Eikonal hybrid location error (m) | Eikonal origin-time error (s) |
| --- | ---: | ---: | ---: |
| region_02 | 196.8 | 21.5 | 0.0013 |
| region_03 | 222.0 | 52.1 | 0.0050 |
| region_04 | 186.9 | 24.5 | 0.0005 |
| region_05 | 159.5 | 70.1 | 0.0021 |
| region_06 | 154.6 | 115.1 | 0.0017 |
| region_07 | 82.0 | 18.0 | 0.0006 |
| region_08 | 314.6 | 417.8 | 0.0506 |
| central_gap | 261.5 | 4.6 | 0.0010 |
| **Median** | **191.8** | **38.3** | -- |

This post-hoc development set gives 8/8 location errors below 500 m and 7/8
hybrid errors below the FMM location error. It is encouraging but cannot be
used as a new accuracy claim because the events guided the preceding model
and search development. `region_08` remains the difficult case and is also
one of the events whose forward L2 did not improve.

The two-stage eikonal-carrier training took 30.97 s for receiver pretraining
and 77.97 s for hybrid training on the existing RTX 4060 Laptop GPU. These
are additional development costs; they exclude the 28.52 s used to train the
eikonal travel-time network and all FD data-generation costs. The experiment
does not establish an end-to-end speed advantage.

## Decision

The eikonal carrier is the first modification that improves both forward
waveform statistics and the post-hoc localization diagnostic, so it is a
reasonable candidate for a new independent confirmation. It is not yet a
publishable result: the forward gain is modest and mixed, and the localization
set is already consumed. Freeze this checkpoint and search protocol, register
fresh source coordinates, generate new FD records, and repeat the comparison
without opening their source truth before the searches finish. Report all
failures and compare against FMM and the straight-carrier control.

Raw outputs: `eikonal_coverage_development_results.json`,
`eikonal_coverage_control_forward.json`,
`eikonal_coverage_augmented_forward.json`, and
`eikonal_seeded_localization_development.json`. Reproduction scripts:
`examples/run_eikonal_coverage_development.py` and
`examples/audit_eikonal_seeded_localization.py`.
