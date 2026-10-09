# Spatially partitioned local-expert development audit

This diagnostic trained three local experts by fine-tuning the current best
source-conditioned checkpoint on spatially partitioned subsets of the 33
already registered FD training records. A target source was assigned to the
nearest expert centre in scaled `(x, y, z)` space. The models used 500
receiver-only updates, a learning rate of `2e-6`, and the same 5 Hz eikonal
carrier as the base checkpoint.

The 16 evaluation records were opened in earlier experiments, so this is not
independent validation. It is a development test of whether spatial partitioning
alone fixes the forward operator.

| Forward surrogate | Mean full-record L2 | Mean active-sample L2 |
|:--|--:|--:|
| Global base checkpoint | **1.0493** | **0.9614** |
| Nearest local expert | 1.0530 | 0.9631 |

Only 2 of 16 records improved. The three experts required 15.9 s of GPU
training in total, but the small computational cost did not produce a forward
accuracy gain. The expert assignments and per-event metrics are recorded in
`local_expert_development_results.json`.

This closes the simple “partition and fine-tune” route. The failure is useful:
the dominant error is not only spatial averaging between nearby sources. A
successful next model must learn a correction with a shared representation and
explicit source-region dependence, for example a residual network whose input
is the local FD interpolant plus source coordinates, or a jointly trained
mixture-of-experts with a smooth gate. No new localization test should be run
until such a model passes a new forward-error gate.
