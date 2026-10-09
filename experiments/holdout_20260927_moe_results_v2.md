# Eight-event independent holdout forward audit

Eight FD receiver records were generated after the MoE development run and
were excluded from training and model selection. They span interior and
boundary-near source positions, depths from 380 to 1400 m, and origin times
from 0.12 to 0.44 s.

| Event | Source `(x,y,z,t)` | Base full L2 | MoE full L2 | Improvement |
|:--|:--|--:|--:|--:|
| 01 | `(4200,7200,380,0.18)` | 1.1626 | 1.0855 | 6.64% |
| 02 | `(8800,5200,900,0.28)` | 1.0965 | 1.0555 | 3.74% |
| 03 | `(11800,13200,1280,0.36)` | 1.2336 | 1.1231 | 8.95% |
| 04 | `(13200,6200,700,0.44)` | 1.0603 | 1.0272 | 3.12% |
| 05 | `(3000,11000,600,0.12)` | 1.1053 | 1.0571 | 4.37% |
| 06 | `(7000,14000,1100,0.24)` | 1.1594 | 1.0839 | 6.51% |
| 07 | `(10000,3000,450,0.32)` | 1.1401 | 1.0854 | 4.79% |
| 08 | `(14000,10000,1400,0.40)` | 1.1034 | 1.0391 | 5.83% |
| **Mean** | — | **1.1326** | **1.0696** | **5.57%** |

The MoE improved all eight records. Mean active-sample L2 decreased from
`1.0543` to `1.0241` (2.86%). Mean gate usage was
`[0.366, 0.400, 0.234]`, indicating no gate collapse.

This is a forward-only independent holdout audit. It uses one trained
checkpoint and eight synthetic records, so it does not establish successful
source localization, robustness over random seeds, noise sensitivity, or
calibrated uncertainty. The result supports proceeding to a carefully bounded
localization smoke test, while a publication claim still requires more held-out
events and repeated seeds.
