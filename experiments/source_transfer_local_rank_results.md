# Local source-ranking development test

Both branches restart Adam at the frozen 500-target-update checkpoint and use the same 1000 target batches. Only local_rank adds a hinge loss that rewards lower waveform MSE at the known training source than at independently sampled nearby false sources. The true source time is held fixed for negative candidates. The previously frozen original 1500-update checkpoints are not used as controls because their Adam target phase was uninterrupted.

The proposed auxiliary loss was ineffective: all six paired control and ranking models have exactly equal parameters (maximum absolute difference = 0). The logged ranking hinge was zero at both recorded milestones, 500 and 1000 updates, for every seed and training path. Identical models would give identical localization predictions, so evaluation was skipped.

| Training path | Seeds | Max parameter difference |
|---|---:|---:|
| source_to_target | 4101, 4102, 4103 | 0 |
| target_to_target | 4101, 4102, 4103 | 0 |

The batch-averaged contrast between randomly perturbed training sources is already satisfied. This does not address the hard wrong-source minima observed on development events. Changing the margin using those events would amount to tuning on the diagnostic set. A physically supported next experiment would add FD records from neighboring *training* sources and check source-dependent waveform behavior before another independent localization evaluation.

Only target training events provide gradient updates. Four previously inspected development events provide diagnostics; the four source_transfer_moe_test events are not used. This discriminative auxiliary loss does not supply physical waveforms at negative sources and cannot by itself validate the forward operator.
