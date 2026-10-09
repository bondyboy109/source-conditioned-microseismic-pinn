# Post-hoc source-conditioned localization diagnostic

This run applies the source-conditioned PINN inversion interface to the three records registered as `blind_01`–`blind_03` in `5hz_protocol.json`. The checkpoint was fixed before this run (`registered_5hz_train8/conditioned_5Hz.pth`, 5 Hz, 5000 updates). Each event used a 12 × 12 × 8 × 8 coarse grid followed by 300 Adam updates from the single best coarse candidate.

The records had already been opened in the earlier forward audit, and the predeclared forward feasibility gate had already failed. Consequently, this is a post-hoc diagnostic of the inversion behavior, not independent localization confirmation.

| Event | True source `(x,y,z,ts)` | Estimated source `(x,y,z,ts)` | `eloc` (m) | `et` (s) | Final normalized MSE | True-source MSE |
|:--|:--|:--|--:|--:|--:|--:|
| blind_01 | (6000, 6000, 350, 0.24) | (7424, 2030, 409, 0.131) | 4218 | 0.109 | 1.191 | 1.359 |
| blind_02 | (10500, 9500, 1000, 0.33) | (7059, 4462, 582, 0.160) | 6115 | 0.170 | 1.035 | 1.141 |
| blind_03 | (7000, 11000, 1400, 0.16) | (7313, 3105, 522, 0.172) | 7950 | 0.012 | 1.090 | 1.162 |

Across the three events, the mean spatial error is 6094 m and the median is 6115 m. The mean origin-time error is 0.097 s and the median is 0.109 s. Each event has a nonzero gradient in all four source coordinates, so the revised interface is differentiable with respect to `(xs, ys, zs, ts)`; differentiability alone did not produce a physically correct inverse.

The optimized source has a lower waveform misfit than the true source for all three events, by factors of 1.07–1.14 in true-MSE/final-MSE. This is a forward-model bias diagnostic: the inversion is fitting errors in the learned operator rather than recovering the FD source. The final losses remain near or above the zero-prediction baseline of 1.0, consistent with the earlier forward audit (`relative L2` values 1.127, 1.054 and 1.128).

The result does not support a localization-accuracy claim. It does establish the next technical priority: improve source-conditioned forward waveform fidelity on newly generated development events before running another localization test. In particular, the optimizer and grid should not be tuned against these three records; a future confirmation must use new FD events generated after the forward gate passes.

Full provenance and hashes are in `registered_blind_localization_diagnostic_protocol.json` and `registered_blind_localization_diagnostic.json`. Raw inversion records are under `outputs/blind_localization_registered_5hz/`.
