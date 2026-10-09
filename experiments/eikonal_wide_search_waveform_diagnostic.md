# Why the independent wide-search localization failed

This is a **post-hoc diagnostic**, not a new confirmation experiment. It uses
the same frozen 29-source 5 Hz checkpoint, all 29 training FD records and all
eight already-spent independent FD records. All predictions are evaluated at
the *known true source*. No reported localization estimate was changed. Input
hashes and the previous independent result are checked by
`examples/diagnose_eikonal_wide_search_waveforms.py`; the full per-receiver
data are in `eikonal_wide_search_waveform_diagnostic.json`.

| Median across events | 29 training records | 8 spent test records |
|:--|--:|--:|
| Raw receiver waveform relative L2 | 0.145 | 1.082 |
| L2 after best event-wide scalar gain (diagnostic only) | 0.145 | 0.994 |
| Peak-amplitude ratio, informative receivers | 0.957 | 0.538 |
| Absolute zero-lag waveform cosine, informative receivers | 0.988 | 0.283 |
| Absolute cosine after up to 0.2 s per-receiver shift (diagnostic only) | 0.989 | 0.538 |
| Median absolute first-threshold-crossing difference | 0.276 s | 1.200 s |
| PINN picks in first 0.05 s / informative FD receiver-event pairs | 79/311 | 59/80 |

The first-threshold-crossing difference is **not** a measured propagation-time
error: the frozen model has no source-dependent onset gate and many test
predictions have a small but persistent pre-arrival oscillation, which makes
the 5%-of-trace picker fire at or near t=0. Training traces also contain this
artifact, despite their low global L2. At `test_05`, 14 of 14 otherwise valid
FD receivers have PINN first picks by 0.016 s, whereas their FD picks range
from approximately 0.67 to 1.73 s. The per-receiver shifts and fitted gains
above use the FD trace and are diagnostics only; they must never be carried
over to an inversion as if the truth were known.

Amplitude is also wrong on new sources, but a single fitted amplitude factor
only reduces their median L2 from 1.082 to 0.994. Zero-lag alignment drops
from 0.988 on training records to 0.283 on the spent test records. This is a
**source-generalization failure of the waveform operator**, involving shape
and timing as well as amplitude. Merely adding more spatial candidates or
adjusting a global gain cannot establish an accurate forward model. Previous
development tests of a smooth onset gate removed false early picks but did
not improve active-sample waveform L2 or source-dependent depth response
(`trained_onset_gate_development_results.md`). Those earlier tests do not
constitute confirmation on these eight new records.

For `test_04`, the fixed FMM pick rule accepts **5/25 receivers**, one fewer
than its preregistered minimum. The remaining 20 break down into 12 traces
with no threshold crossing in 2 s, seven with a peak below 1% of the event
maximum and one with a first pick after 1.85 s (the fixed 0.15 s end margin).
The source is near the array's corner at (12.9, 13.7) km. Its limited
2-second aperture is a separate failure of the FMM initializer; lowering the
six-pick threshold would change the original protocol and would not fix the
PINN waveform discrepancy. A longer-record experiment would require matching
FD/PINN timing, training coverage and a newly fixed evaluation design.

Visual checks at the true source: `outputs/eikonal_wide_search_waveform_diagnostic/training_04.png`,
`test_04.png` and `test_05.png` in that same directory. The figures overlay
four highest-amplitude FD station traces with the PINN prediction and show
the signed residual. The plotted 1e-6 scale is the numerical wavefield unit,
not a calibrated pressure measurement. The vertical dotted line marks the
1.85 s FMM picker cutoff. The training overlay follows FD closely, while
both new-event overlays have visible arrival, phase and amplitude differences.

The next model iteration should be developed using training and previously
spent development data, with active-window waveform L2 and per-receiver
alignment evaluated **before** localization. Freeze the new operator, the
observation window and the search rule, then generate new, untouched events
for any subsequent performance claim. The eight events in this report are
now diagnostic data and cannot be reused as independent evidence.
