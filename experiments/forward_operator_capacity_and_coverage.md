# Forward-operator capacity and source-coverage controls

These controls test whether cross-source failure can be fixed by adding more
training sources or increasing the trainable MLP capacity. The three
`independent_test_*` events were generated before these runs and were never
used for training or model selection.

## Results on the same independent test set

| Model | Training sources | Trainable network | Test 01 L2 | Test 02 L2 | Test 03 L2 |
| --- | ---: | --- | ---: | ---: | ---: |
| Baseline hybrid | 16 | 128 hidden units, 5 layers | 0.907 | 1.005 | 1.720 |
| Coverage expansion | 24 | 128 hidden units, 5 layers | 0.951 | 1.124 | 3.852 |
| Capacity expansion | 24 | 256 hidden units, 6 layers | 1.080 | 1.102 | 4.085 |

The 24-source and wider-network controls do not improve independent
cross-source waveform prediction. Increasing source coverage actually worsens
the edge test, and increasing MLP capacity does not recover the missing
heterogeneous-medium physics. These controls reject two simple explanations:
the failure is not solved by source-count expansion alone and is not solved by
ordinary width/depth expansion.

The next forward-model implementation must expose velocity-field information
to the operator or use a numerically computed heterogeneous traveltime/path
representation. A source-conditioned MLP with a frozen first-arrival carrier
is insufficient for the observed curved, multipath, and amplitude-varying
waveforms.

The training script now accepts `--hidden-dim` and `--num-layers`; these values
are stored in each checkpoint for reproducibility. All 24 project tests pass.
