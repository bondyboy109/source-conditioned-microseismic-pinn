# FD joint origin-time and amplitude depth diagnostic

Two already inspected events at truth-selected coarse horizontal nodes. Each depth uses the original FD template and discrete time bounds. At every shift, an independent nonnegative scalar gain minimizes normalized MSE; the best shift and gain are selected jointly. Original unit-gain results are unchanged.

| Event | Depth (m) | Unit MSE | Fixed-time gain MSE | Joint MSE | Joint origin (s) | Joint gain |
|---|---:|---:|---:|---:|---:|---:|
| confirm_01_west_shallow | 100 | 125.8771 | 1.0000 | 0.9995 | 0.398 | 0.0019 |
| confirm_01_west_shallow | 250 | 23.5327 | 1.0000 | 0.9986 | 0.362 | 0.0079 |
| confirm_01_west_shallow | 400 | 6.0520 | 0.9456 | 0.9456 | 0.498 | 0.0936 |
| confirm_01_west_shallow | 550 | 3.9324 | 0.9392 | 0.9392 | 0.454 | 0.1247 |
| confirm_01_west_shallow | 800 | 3.2985 | 0.9731 | 0.9731 | 0.366 | 0.0970 |
| confirm_01_west_shallow | 1100 | 1.5593 | 0.9776 | 0.9776 | 0.280 | 0.1640 |
| confirm_01_west_shallow | 1500 | 1.3191 | 0.9842 | 0.9841 | 0.180 | 0.1790 |
| confirm_05_center_shallow | 100 | 35.3482 | 0.2244 | 0.2244 | 0.276 | 0.1294 |
| confirm_05_center_shallow | 250 | 7.2897 | 0.1904 | 0.1904 | 0.242 | 0.2524 |
| confirm_05_center_shallow | 400 | 0.6949 | 0.2035 | 0.2035 | 0.196 | 0.5601 |
| confirm_05_center_shallow | 550 | 0.3280 | 0.2839 | 0.2839 | 0.150 | 0.8011 |
| confirm_05_center_shallow | 800 | 0.3323 | 0.3199 | 0.3199 | 0.068 | 0.8810 |
| confirm_05_center_shallow | 1100 | 1.1332 | 0.9697 | 0.9697 | 0.146 | 0.3008 |
| confirm_05_center_shallow | 1500 | 1.0803 | 0.9966 | 0.9948 | 0.050 | 0.1691 |

Gain zero gives MSE one (zero waveform); such a candidate is not a meaningful successful fit. This diagnostic changes the objective and uses inspected events. It is neither a blind localization result nor an accuracy-matched comparison with the neural workflow. A global 75-node gain-profiled search would require rescoring every node using the same rule.
