# Source-depth response sweep (development)

The registered protocol selected two trained centers (a10 at 1000 m depth, a14 at 1200 m depth) and **twelve new** FD sources at vertical offsets of +/-80, +/-160 and +/-280 m, keeping x, y and origin time fixed. Before generating the records, the script checked that none of the new source locations coincided with 160 earlier FD observations. Both 29-source, 5 Hz checkpoints and the velocity volume remained frozen. Previously inspected +/-240 m records are shown only as context; they are not part of the new group or its means. This is a development study on one velocity model, not independent validation of localization.

The primary metric is true-source receiver-waveform L2, `||PINN-FD|| / ||FD||`, over 25 stations and 0-2 s, without fitted gain or time shift. The finite response metrics compare the change in the waveform after moving the source with the change in the FD waveform from the same training center.

| Offset magnitude | Eikonal L2 (4 new events) | Straight L2 (4 new events) | Eikonal change L2 | Eikonal change cosine | Eikonal/FD change norm |
|:--|--:|--:|--:|--:|--:|
| 80 m | 0.392 | 0.441 | 0.933 | 0.483 | 0.871 |
| 160 m | 0.620 | 0.674 | 0.814 | 0.601 | 0.854 |
| 280 m | 0.930 | 0.921 | 0.658 | 0.744 | 0.783 |

| Center | 80 m: eikonal / straight L2 | 160 m: eikonal / straight L2 | 280 m: eikonal / straight L2 | Old 240 m: eikonal L2 |
|:--|--:|--:|--:|--:|
| a10 | 0.395 / 0.429 | 0.679 / 0.733 | **1.118 / 1.103** | 0.976 |
| a14 | 0.389 / 0.452 | 0.561 / 0.615 | 0.742 / 0.740 | 0.697 |

In this paired grid, eikonal has lower raw L2 than straight in 10/12 events, particularly at 80 and 160 m. The advantage disappears at 280 m (mean eikonal 0.930 versus straight 0.921), and a10 at 280 m has mean L2 above the zero-prediction benchmark of 1. The lower *change-relative* L2 at 280 m does not contradict the rising raw L2: FD waveform differences relative to each center also grow (mean change/anchor norm: 0.397, 0.767, 1.245 for 80, 160, 280 m). These are different denominators. Increasing offset and depth-dependent FD responses cannot be separated as independent causal effects with two centers and one velocity model.

## Threshold-pick failure

The protocol also fixed a 5%-of-each-trace-peak, three-consecutive-sample arrival picker on stations with a usable FD pick. The unmodified machine-readable results contain its raw mean errors and RMSE, but **these cannot be interpreted as propagation-time accuracy**. An additional post-hoc audit found that, among 142 usable FD station/event pairs, eikonal predicts 65 threshold crossings *before* origin time minus 0.10 s; 39 occur at t=0. FD has zero such early crossings. Straight predicts 71 early crossings, 41 at t=0. The eikonal model's mean RMS output before origin minus 0.10 s is 0.126 times the full FD-record RMS, whereas FD receiver amplitudes in those early windows are zero. For example, on a10 -80 m one FD trace crosses threshold at about 1.874 s but the eikonal prediction crosses at 0 s. The fixed rule thus measures low-level network background as a false first arrival for many stations. The original pick results are retained for reproducibility; no adjusted threshold is reported as a replacement result. Diagnose pre-arrival output or impose a validated causal wavefield constraint before using waveform threshold picks as a travel-time metric.

The next operator development target is depth-dependent waveform response and pre-arrival suppression. A corrected architecture/training objective needs its own development evaluation and newly registered, unseen events for independent confirmation. The 12 new events here and the 240 m references are already consumed as development data. They provide no new location accuracy or computational speedup claim.

Inputs and full per-event measurements: `eikonal_depth_sweep_protocol.json` and `eikonal_depth_sweep_results.json`. Raw early-pick counts and original-result hash: `eikonal_depth_sweep_early_pick_audit.json`. Scripts: `examples/run_eikonal_depth_sweep.py` and `examples/audit_eikonal_depth_early_picks.py`. The 12 FD records took approximately 108.1 seconds total to generate on the current GPU; this is not inverse runtime.
