# MoE localization smoke test

After the eight-event independent forward audit, one held-out event was used
for a bounded localization smoke test. The test used a 6 x 6 x 4 x 4 coarse
source grid, retained the three lowest-misfit starts, and refined each for 80
Adam updates. Receiver and time subsampling was used only to keep this smoke
test inexpensive.

| Quantity | Value |
|:--|:--|
| True source `(x,y,z,t)` | `(4200, 7200, 380, 0.18)` |
| Recovered source `(x,y,z,t)` | `(4289.7, 13970.6, 100, 0.50)` |
| Location error | `6777.0 m` |
| Origin-time error | `0.320 s` |
| Selected coarse loss | `1.0433` |
| Selected refined loss | `1.0380` |
| Loss at true source | `1.1482` |

The model has a finite, nonzero gradient with respect to all four source
parameters, so the source-conditioned interface is differentiable. However,
the best objective value is attained far from the true source and at the upper
origin-time bound. This single smoke test therefore does **not** demonstrate
successful localization. It shows that the improved forward surrogate has not
yet removed the inverse ambiguity. A publication result would require a
pre-registered multi-event inversion protocol, independent source tests, and
comparison with FMM/FD baselines.
