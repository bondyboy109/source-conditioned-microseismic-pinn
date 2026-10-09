# Source-conditioned low-rank localization result

This is a one-time locked-event evaluation for this frozen operator. The synthetic records have been examined in studies of other models; this is not a field-data validation.

| Objective | n / attempted | Median eloc (m) | Mean eloc (m) | Median et (s) | Mean et (s) | <500 m among available |
|---|---:|---:|---:|---:|---:|---:|
| Waveform only | 8/8 | 743.2 | 2513.9 | 0.1095 | 0.1166 | 2/8 |
| Waveform + FMM arrival | 7/8 | 394.5 | 506.1 | 0.0229 | 0.0293 | 5/7 |

FMM arrival constraint improved spatial error in 6/7 paired available events. The hybrid was unavailable on 1/8 attempted events; its error summaries exclude that event. Its overall yield below 500 m is 5/8 attempts. See `source_conditioned_low_rank_summary_zh.md` for the comparison on identical available events.

## Event results

| Event | eloc waveform (m) | eloc hybrid (m) | et hybrid (s) | Runtime (s) |
|---|---:|---:|---:|---:|
| confirm_01_sw_shallow.npz | 605.9 | 394.5 | 0.0158 | 10.2 |
| confirm_02_se_deep.npz | 871.7 | 471.3 | 0.0577 | 11.3 |
| confirm_03_nw_deep.npz | 658.3 | 226.9 | 0.0416 | 11.6 |
| confirm_04_ne_shallow.npz | 14277.6 | NA | NA | 4.5 |
| confirm_05_west_deep.npz | 2069.0 | 1385.5 | 0.0073 | 8.3 |
| confirm_06_east_shallow.npz | 320.0 | 64.0 | 0.0217 | 8.3 |
| confirm_07_north_mid.npz | 828.0 | 394.5 | 0.0383 | 8.3 |
| confirm_08_center_deep.npz | 480.9 | 606.0 | 0.0229 | 8.3 |

The eight FD records are synthetic and have been used in earlier studies of other frozen models. This comparison is locked with respect to the present operator, but it is not a prospectively collected field-data test.
