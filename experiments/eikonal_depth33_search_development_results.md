# Regional candidate-ranking and wider-search diagnostic

This is a post-hoc **development** experiment on the same eight inspected 5 Hz FD regional events. Both the old 29-source eikonal carrier and the new 33-source depth-coverage carrier were frozen. For each event, both models used the same archived FMM center, 0.05--0.50 s origin-time grid, normalized waveform objective and bounded 250-step Adam refinement. We scored the original spatial offsets `[-320, 0, 320]` m and a fixed enlarged grid `[-640, -320, -160, 0, 160, 320, 640]` m per axis. The enlarged search selected the 20 best spatially distinct starts with a 160 m minimum separation. For each model, source truth was read after its search only to measure error and retrospectively rank the nearest grid site; it did not enter the second model's search. All events and parameters were fixed in `eikonal_depth33_search_development_protocol.json` before this run.

| Event | Old 29-source, old search (m) | Old 29-source, wider search (m) | New 33-source, old search (m) | New 33-source, wider search (m) |
|:--|--:|--:|--:|--:|
| region_02 | 21.5 | 21.5 | 42.9 | 42.8 |
| region_03 | 52.1 | 52.1 | 67.5 | 67.5 |
| region_04 | 24.5 | 24.5 | 26.1 | 26.1 |
| region_05 | 70.1 | 70.1 | 133.3 | 133.3 |
| region_06 | 115.1 | 115.2 | 103.9 | 103.9 |
| region_07 | 18.0 | 18.0 | 22.5 | 22.5 |
| region_08 | 417.8 | **14.9** | 739.6 | **997.1** |
| central_gap | 4.6 | 4.6 | 21.0 | 21.0 |
| **Median** | **38.3** | **23.0** | **55.2** | **55.2** |

In `region_08`, the old model's nearest true-source spatial grid point ranked 18/27 under the original grid and 10/294 under the wider grid. It entered the wider search's 20 starts (98.6 m from truth); the selected estimate is 14.9 m from truth with normalized MSE 0.651. The new model's nearest point ranked 13/27 originally and 37/294 on the wider grid; **none** of the wider search's 20 selected starts was closer than 518.5 m to truth. Its selected estimate is 997.1 m away, with origin-time error 0.270 s and MSE 0.834. This is a candidate-ranking and basin-selection failure for the new model on this event; increasing candidate density alone did not fix it. At true x/y the new model still prefers the true 700 m depth, and its exact-truth loss is lower than the returned winner (`eikonal_depth33_region08_depth_profile.json`), so the 100 m boundary selected by the previous search was not evidence of a globally preferred shallow source.

The widened grid evaluates 13,524 candidates for `region_08` after bound clipping, versus 1,242 originally; across the eight events it costs about 28.8 s per event for the old model and 28.5 s for the new model, including 20 refinements, on the local GPU. For `region_08`, coarse waveform scoring increased from about 0.9 s to 10.0 s per model; this excludes the 36.9 s FMM receiver-field setup and the FD simulation cost. The median improvement for the old model and the single severe regression for the new model are **not** independent success/failure rates: both models and this search have now been examined on these records. Previous independent tests cannot be retroactively reclassified under the new search.

The defensible outcome is a search diagnosis, not a publishable localization result. Local depth supervision improves the inspected depth waveforms, while new regional data are needed to assess whether a frozen search and operator can localize reliably. Keep both checkpoints and all negative results; do not replace the old model with the 33-source model based on the local waveform gain alone. Full per-event rankings, nearest selected starts, MSE values, estimates and times are recorded in `eikonal_depth33_search_development_results.json`; reproduce with `examples/audit_eikonal_depth33_search.py`.
