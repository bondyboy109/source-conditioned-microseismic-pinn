# Directional source-response development grid

The coordinates and 12 FD runs were fixed in `eikonal_directional_grid_protocol.json` before generation. Two **training** events, `train_10` at (10000, 6500, 1000) m and `train_14` at (11000, 10000, 1200) m, are the centers. Each new event moves just one coordinate by -240 or +240 m and retains the corresponding origin time. Before generating data, the script verified that none of the 12 positions appeared in the 148 existing observation records. This is a **development experiment**, not an independent localization test or evidence of generalization across other velocity models.

Both frozen 29-source checkpoints were evaluated without retraining: a source-conditioned eikonal travel-time carrier (`ce744ba5...`) and its matched straight-time carrier (`5f3a27f...`). At the known source, raw waveform L2 means `||PINN-FD||_2 / ||FD||_2` over the full 25-receiver, 2-second record. Finite response error compares the *change in predicted waveforms from the anchor* with the corresponding FD change, normalized by the FD-change norm. Neither metric fits a time shift or amplitude factor.

| Anchor | Offset | Eikonal raw L2 | Straight raw L2 | Eikonal change L2 | Eikonal change cosine |
|:--|:--|--:|--:|--:|--:|
| 10 | x -240 m | 0.355 | 0.373 | 0.249 | 0.969 |
| 10 | x +240 m | 0.315 | 0.330 | 0.235 | 0.972 |
| 10 | y -240 m | 0.364 | 0.387 | 0.277 | 0.961 |
| 10 | y +240 m | 0.389 | 0.393 | 0.287 | 0.960 |
| 10 | z -240 m | **1.092** | 1.078 | **0.849** | **0.547** |
| 10 | z +240 m | 0.861 | 0.914 | 0.799 | 0.640 |
| 14 | x -240 m | 0.270 | 0.246 | 0.332 | 0.946 |
| 14 | x +240 m | 0.224 | 0.237 | 0.285 | 0.959 |
| 14 | y -240 m | 0.340 | 0.350 | 0.216 | 0.980 |
| 14 | y +240 m | 0.296 | 0.305 | 0.223 | 0.975 |
| 14 | z -240 m | 0.616 | 0.626 | 0.585 | 0.824 |
| 14 | z +240 m | 0.779 | 0.779 | 0.607 | 0.799 |

| Axis (four events each) | Eikonal raw L2 | Straight raw L2 | Eikonal change L2 | Straight change L2 | Eikonal change cosine | Eikonal/FD change norm |
|:--|--:|--:|--:|--:|--:|--:|
| x | 0.291 | 0.297 | 0.275 | 0.286 | 0.962 | 1.005 |
| y | 0.347 | 0.359 | 0.251 | 0.265 | 0.969 | 1.012 |
| z | **0.837** | **0.849** | **0.710** | **0.713** | **0.703** | **0.809** |

The eikonal carrier reduces raw L2 in 9/12 paired events, but its overall mean changes only from 0.502 (straight) to 0.492 (eikonal). Both operators have a marked depth-response weakness at these two anchors. A 240 m vertical displacement produces mean eikonal L2 0.837, compared with 0.291 and 0.347 for equally sized x and y displacements. The z-axis FD waveform change is about 1.10 times the anchor waveform norm, comparable to the x-axis value of 1.09; the high z error is not explained merely by a larger FD-change norm. The model's finite z response is both weaker and less aligned with FD. Its 0.809 average amplitude ratio and 0.703 average cosine contrast with x/y ratios near 1 and cosines above 0.96.

This is a narrow, paired diagnosis. Two anchors, one displacement magnitude, one 5 Hz model and no repetition cannot establish a universal anisotropy law or statistical superiority for the eikonal carrier. The anchor records are training data, so the small error at zero displacement is expected; the new offsets are development data and must not later be described as untouched tests. The next development experiment should hold the same anchors and vary **depth displacement** over several registered magnitudes while tracking FD/PINN waveform-change and first-arrival errors. Spatial extrapolation across the full 16 km model remains unresolved; the previous eight-event independent failure still stands.

The 12 FD simulations took 108.5 s in total on the current GPU (data-generation time, not inverse cost). Per-event source coordinates, hashes, metrics, nearest-training distances and FD times are in `eikonal_directional_grid_results.json`. Reproduce or preflight with `examples/run_eikonal_directional_grid.py` (`--preflight` verifies only inputs, and the completed report is protected against overwrite).
