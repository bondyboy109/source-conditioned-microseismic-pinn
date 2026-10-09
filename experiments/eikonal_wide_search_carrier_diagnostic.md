# Learned carrier and frozen waveform-head audit

These are **post-hoc diagnostics** on 29 source-training records and eight
already-spent new-source FD records. The checkpoint, true-source coordinates,
original receiver observations and velocity grid are unchanged. They are not
new independent tests or source inversion results.

First, compare the frozen model's learned eikonal travel time against the
pre-existing baseline's reciprocal FMM fields at 80 m grid spacing. FMM uses
one field for each of 25 receivers and interpolates travel times at the true
source; the field's receiver position is rounded to its nearest cell
(20--60 m in this array). The same FD-valid station masks from the previous
waveform diagnostic are used for both groups.

| Median across events | 29 trained sources | 8 spent new sources |
|:--|--:|--:|
| Distance to nearest trained source | 0 m | 1736 m |
| Learned carrier vs FMM, RMSE on valid stations | 0.071 s | 0.088 s |
| Learned carrier vs FMM, demeaned RMSE on valid stations | 0.042 s | 0.038 s |
| True-source full-waveform relative L2 | 0.145 | 1.082 |

The carrier-to-FMM discrepancy is of the same order in both groups; the
waveform discrepancy is not. The carrier error is **not negligible** at 5 Hz
(one period is 0.20 s), and an approximate FMM field is not an exact FD
waveform solution. The comparison therefore does not validate the carrier
absolutely or prove the waveform head is the only cause of poor transfer.
It does show that a large new carrier-vs-FMM degradation is not observed
alongside the much larger new-source waveform error.

Next, keep the frozen waveform-head weights and known true sources fixed, but
replace the learned travel time in its phase and features with the computed
FMM time. The diagnostic implementation checks its **unmodified** output
against the actual model forward pass at every receiver and checks its raw L2
against the previously saved forward result before reporting a swap. No
weights are retrained and no candidate source is inverted.

| True-source relative waveform L2, median | Learned carrier | FMM substitution |
|:--|--:|--:|
| Training records | 0.145 | 1.186 |
| Eight spent new-source records | 1.082 | 1.132 |

The swap improves only 3/8 spent events and severely worsens the fitted
training examples. This is expected for a waveform head trained with the
learned carrier: the phase and learned amplitude coefficients depend on that
specific time representation. **These numbers do not compare a retrained
FMM-conditioned operator with the existing PINN.** They specifically reject
plugging FMM times into this frozen waveform head as an immediate remedy.

The next model-development experiment must jointly train a source-dependent
waveform operator with its chosen travel-time interface and compare its
waveform amplitude, phase and source-displacement response on existing
development data *before* inversion. Prior pilots with increased source
count/MLP width, straight-path velocity samples and onset gates already
failed to solve global transfer; repeating them unchanged is unlikely to
help. A new localization or generalization claim requires newly generated,
untouched test sources after the architecture and search have been fixed.

Full event-level times, FMM grid details, input hashes and runtimes:
`eikonal_wide_search_carrier_vs_fmm.json` and
`eikonal_wide_search_fmm_carrier_swap.json`. Reproduce with
`examples/audit_wide_search_carrier_vs_fmm.py` followed by
`examples/audit_fmm_carrier_waveform_swap.py`; both refuse to overwrite
existing results.
