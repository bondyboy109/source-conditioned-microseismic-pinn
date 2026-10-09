# FMM-seeded MoE waveform plus arrival refinement

The FMM estimates were used only to define a fixed local `3 x 3 x 3` spatial
neighborhood. Each neighborhood candidate was scored by the MoE waveform loss,
the FMM arrival residual, and the fixed hybrid objective
`waveform_loss + 0.25 * (arrival_RMSE / 0.10 s)^2`.

| Selection rule | Mean location error (m) | Median location error (m) | Mean origin-time error (s) |
|:--|--:|--:|--:|
| MoE waveform only | 2560 | 2208 | 0.169 |
| FMM arrival only | 1097 | 1128 | 0.121 |
| Hybrid (`alpha=0.25`) | **1049** | **1014** | **0.099** |

The hybrid criterion improves the FMM reference by about `4.4%` in mean
location error and `18.2%` in mean origin-time error. It improves the local
waveform-only choice substantially, but the experiment is still a diagnostic:
the local neighborhood is centered by FMM, the arrival labels use a threshold
pick, and only one MoE checkpoint and eight synthetic events were used. This
does not support a final localization-accuracy claim. The next validation
should pre-register the same protocol, add repeated seeds and noise levels,
and compare against the original waveform-only inversion on the same events.
