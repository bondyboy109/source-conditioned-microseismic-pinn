# FD amplitude attribution at fixed origin time

The source is the saved two-event depth profile; no FD waveforms were regenerated. At each depth, the time shift remains the unit-amplitude optimum from that profile. Only a nonnegative scalar gain is fit afterward. The original localization baseline and its objective remain unchanged.

| Event | Depth (m) | Fixed origin (s) | Unit MSE | Energy ratio | Best gain | Gain-adjusted MSE |
|---|---:|---:|---:|---:|---:|---:|
| confirm_01_west_shallow | 100 | 0.050 | 125.8771 | 124.8731 | 0.0000 | 1.0000 |
| confirm_01_west_shallow | 250 | 0.050 | 23.5327 | 22.5915 | 0.0013 | 1.0000 |
| confirm_01_west_shallow | 400 | 0.498 | 6.0520 | 6.2154 | 0.0936 | 0.9456 |
| confirm_01_west_shallow | 550 | 0.454 | 3.9324 | 3.9070 | 0.1247 | 0.9392 |
| confirm_01_west_shallow | 800 | 0.366 | 3.2985 | 2.8520 | 0.0970 | 0.9731 |
| confirm_01_west_shallow | 1100 | 0.280 | 1.5593 | 0.8322 | 0.1640 | 0.9776 |
| confirm_01_west_shallow | 1500 | 0.182 | 1.3191 | 0.4963 | 0.1785 | 0.9842 |
| confirm_05_center_shallow | 100 | 0.276 | 35.3482 | 46.3379 | 0.1294 | 0.2244 |
| confirm_05_center_shallow | 250 | 0.242 | 7.2897 | 12.7037 | 0.2524 | 0.1904 |
| confirm_05_center_shallow | 400 | 0.196 | 0.6949 | 2.5392 | 0.5601 | 0.2035 |
| confirm_05_center_shallow | 550 | 0.150 | 0.3280 | 1.1158 | 0.8011 | 0.2839 |
| confirm_05_center_shallow | 800 | 0.068 | 0.3323 | 0.8762 | 0.8810 | 0.3199 |
| confirm_05_center_shallow | 1100 | 0.146 | 1.1332 | 0.3344 | 0.3008 | 0.9697 |
| confirm_05_center_shallow | 1500 | 0.500 | 1.0803 | 0.1205 | 0.1667 | 0.9966 |

A large decrease after fitting gain indicates an amplitude contribution at the frozen shift, but the remaining loss includes timing and waveform-shape mismatch. For event 01 at 100 m, the best nonnegative gain is zero and the adjusted loss of one equals a zero-waveform prediction; this is not a successful waveform fit. For event 05 at 100 m, gain 0.1294 reduces the loss from 35.3482 to 0.2244, showing a substantial amplitude contribution but leaving nonzero residual mismatch. The adjusted minimum over depths is not the result of a joint time-and-amplitude search; doing that requires rescoring every time shift. These inspected events and truth-selected horizontal nodes cannot support a new blind localization claim.
