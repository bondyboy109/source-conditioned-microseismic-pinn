# Independent holdout forward audit: joint MoE

Four new FD receiver records were generated after the MoE development run and
were not used for training or model selection. The sources cover a shallow
interior location, a mid-depth interior location, a deep boundary-near
location, and a late-origin-time boundary-near location.

| Event | Source `(x, y, z, t)` | Base full L2 | MoE full L2 | Change |
|:--|:--|--:|--:|--:|
| event_01 | `(4200, 7200, 380, 0.18)` | 1.1626 | 1.0855 | -6.64% |
| event_02 | `(8800, 5200, 900, 0.28)` | 1.0965 | 1.0555 | -3.74% |
| event_03 | `(11800, 13200, 1280, 0.36)` | 1.2336 | 1.1231 | -8.96% |
| event_04 | `(13200, 6200, 700, 0.44)` | 1.0603 | 1.0272 | -3.12% |
| **Mean** | — | **1.1382** | **1.0728** | **-5.75%** |

The MoE improved all four records. Mean active-sample L2 decreased from
`1.0592` to `1.0273` (3.02%). The mean gate usage was
`[0.365, 0.395, 0.239]`, so the gate did not collapse to one expert.

This is an exploratory forward holdout, not a localization experiment. It uses
one checkpoint and four synthetic events; it does not establish robustness over
source positions, random seeds, noise levels, or receiver layouts. The result
is sufficient to justify expanding the independent holdout set, but it is not
enough to claim successful event localization or calibrated uncertainty.
