# Source coverage expansion: development result

Eight additional 5 Hz sources (`train_09`-`train_16`) were generated with
the original six-zero-face FD solver and cropped to 0-2 s. The expanded
model used all 16 training sources, the same receiver-only relative-carrier
architecture without the initial-time gate, seed 42, 5000 steps, batch size
1024, output scale `1e-5`, and receiver weight 0.1 as the eight-source
comparison. The additional source locations and fixed settings are recorded
in `experiments/coverage_expansion_5hz.json`.

| Development event | Nearest of 16 training sources | Eight-source relative L2 | Sixteen-source relative L2 |
| --- | ---: | ---: | ---: |
| dev_distance_0250 | 250 m | 0.795 | 0.895 |
| dev_distance_0800 | 800 m | 1.197 | 1.377 |
| dev_distance_2000 | 1136 m | 1.118 | 1.079 |
| heldout_01 | 522 m | 0.997 | 0.986 |

The eight-source comparison and sixteen-source run took approximately
18.0 s and 20.6 s, respectively, for 5000 supervised GPU steps. Each
new full 9 s FD training event took about 46-49 s to generate, excluding
other experiment preparation. Checkpoint provenance confirms that none of
the four evaluation events entered the sixteen-source training set.

Doubling source count does not consistently reduce development error:
the nearest 250 m and 800 m events get worse, and only the two other
events improve slightly. Three of four sixteen-source results are close
to or worse than the zero-prediction baseline of 1.0. The `heldout_01`
source lies only 522 m from an added training event and has already been
used repeatedly for model development; its 0.986 score is not an
independent test result. No previously evaluated `blind_*` observation
was used in this experiment.

This model uses FD receiver waveforms for supervision and no PDE loss.
These measurements cannot establish PINN forward accuracy, localization,
or a computational advantage over FD. Source coverage alone is not a
verified fix for cross-source waveform generalization. The checkpoints,
training summary, and individual audits are in
`outputs/dev_relative_carrier_5hz_window0to2_no_gate_train16_5000/`.
