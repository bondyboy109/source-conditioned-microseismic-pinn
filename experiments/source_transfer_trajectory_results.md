# Source transfer trajectory, development only

The original four-event test set was excluded. The same target FMM fields, arrival picker, bias and localization settings were used at each stage.

| Arm | Target updates | Mean waveform relative L2 (16 events x 3 seeds) | Median hybrid location error (4 events x 3 seeds) | Median origin-time error (s) |
|---|---:|---:|---:|---:|
| source_to_target | 0 | 1.2782 | 788.7 m | 0.0161 |
| source_to_target | 500 | 0.9126 | 228.7 m | 0.0060 |
| source_to_target | 1500 | 0.8138 | 448.9 m | 0.0143 |
| target_to_target | 0 | 1.0576 | 579.8 m | 0.0210 |
| target_to_target | 500 | 0.9267 | 324.6 m | 0.0151 |
| target_to_target | 1500 | 0.8445 | 205.5 m | 0.0141 |

The pooled location medians do not preserve seed-event pairing. With each
source-to-target result compared against target-to-target for the same seed,
event, and target-update count, positive differences mean worse localization
after source-domain pretraining:

| Target updates | Mean paired location difference (m) | Median paired difference (m) | Source transfer worse | Mean paired waveform L2 difference |
|---:|---:|---:|---:|---:|
| 0 | +237.5 | +275.8 | 10/12 | +0.2206 |
| 500 | +114.1 | +28.3 | 8/12 | -0.0141 |
| 1500 | +103.5 | +53.0 | 10/12 | -0.0307 |

The 500-update pooled location median favors source transfer, but its paired
mean, paired median, and event count do not. Target-domain fine-tuning closes
the initial waveform gap; the slightly lower waveform L2 at 500 and 1500
updates does not translate into a paired localization benefit. Localization
also varies non-monotonically with updates, so the 500-update stage should not
be selected based on these development events and presented as an independent
confirmation. These are 12 paired seed-event outcomes over four positions in
one geology, not 12 independent geological settings.

Only previously used development events are evaluated. Do not use the four source_transfer_moe_test events for model selection; any newly chosen method needs a new prospective test set.

The 1500-update checkpoints are the original frozen pilot models; the 0 and 500-update checkpoints are reconstructed with the same seeds and sampling sequence. The target optimizer is restarted at the domain switch.
