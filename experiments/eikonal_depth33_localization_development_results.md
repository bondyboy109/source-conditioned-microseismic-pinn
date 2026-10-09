# Cross-region localization after local depth coverage (development)

The candidate adds four inspected +/-160 m depth receiver records around a10 and a14 to the original 29-source eikonal-carrier training set. The 33-source weights and the 29-source checkpoint were frozen before this comparison. We reused eight **already-inspected** regional FD records, the old FMM spatial centers, 1242 waveform candidates and twelve local refinements per event. A second 33-source checkpoint differs only by a fixed, source-dependent, approximate onset gate; every tensor in its state dictionary equals the ungated checkpoint. All runs use the same 5 Hz source wavelet, observations and velocity model. These are development results, not a new independent test.

| Event | FMM error (m) | Old 29-source (m) | New 33-source (m) | New 33-source with gate (m) |
|:--|--:|--:|--:|--:|
| region_02 | 196.8 | 21.5 | 42.9 | 42.4 |
| region_03 | 222.0 | 52.1 | 67.5 | 65.3 |
| region_04 | 186.9 | 24.5 | 26.1 | 26.3 |
| region_05 | 159.5 | 70.1 | 133.3 | 135.1 |
| region_06 | 154.6 | 115.1 | 103.9 | 107.4 |
| region_07 | 82.0 | 18.0 | 22.5 | 22.7 |
| region_08 | 314.6 | 417.8 | **739.6** | **740.3** |
| central_gap | 261.5 | 4.6 | 21.0 | 20.8 |
| **Median** | **191.8** | **38.3** | **55.2** | **53.8** |

The new model beats the old localization error on only one of eight regional events (`region_06`). In `region_08`, the new model places the source at the lower inversion bound of 100 m depth although the true depth is 700 m. It performs worse than both the original operator and FMM there. The onset gate does not fix this search failure.

The true-source receiver-waveform comparison gives useful context: mean raw relative L2 across these eight regions *improves* from 0.808 to 0.562, with seven of eight events better. But `region_08` alone worsens from 0.614 to 0.842. Lower **mean** forward error is thus not sufficient to preserve localization across regions. At the local depth offsets, eight of eight development waveform and depth-response metrics improved (see `eikonal_depth_coverage33_development_results.md`); that local gain does not generalize reliably to region-specific location searches.

A post-hoc depth/time scan at `region_08`'s **true** x/y isolates part of the failure. At every depth from 100 to 1500 m in 40 m steps, it chooses the best origin time on the original 0.05--0.50 s grid using the same normalized waveform objective. All three operators prefer the correct **700 m** depth, with the correct 0.37 s origin time. The original, new and gated losses at the exact true source are 0.694, 0.709 and 0.698, whereas their reported inverse winners have *higher* losses of 0.910, 0.910 and 0.877. At true x/y, the new model's best loss at 100 m depth is 0.953 versus 0.709 at 700 m. This disproves a simple explanation that the model's depth objective is minimized at 100 m everywhere. The FMM-centered candidate grid and local refinements missed a better basin around the true x/y; the worsened true-source waveform fit in `region_08` is a separate observed limitation. The true x/y scan is a diagnostic that uses known truth and cannot be counted as a working localization method.

The 33-source model is **not selected as a replacement** for the 29-source model, nor does this run authorize a positive localization or cost claim in the manuscript. New independent FD events should remain unopened until an operator and a truth-independent search protocol address the regional boundary failure. The next development target is to inspect candidate rankings around `region_08`'s FMM center and evaluate a fixed broader/multistart search on consumed development records before freezing another candidate.

Machine-readable localization, per-event estimates, forward checks and truth-centered diagnostic: `eikonal_depth33_localization_development_results.json`, `eikonal_depth33_raw_localization_development.json`, `eikonal_depth33_gated_localization_development.json`, `eikonal_depth33_regional_forward_development.json` and `eikonal_depth33_region08_depth_profile.json`. Registered settings: `eikonal_depth33_localization_development_protocol.json`, `eikonal_depth33_region08_depth_profile_protocol.json`. Scripts: `examples/run_eikonal_depth33_localization.py`, `examples/audit_eikonal_seeded_localization.py`, `examples/derive_onset_gate_checkpoint.py`, `examples/audit_eikonal_depth33_region08_profile.py`.
