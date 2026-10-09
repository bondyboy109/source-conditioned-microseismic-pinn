# Trained onset gate: negative development result

The fixed protocol in `trained_onset_gate_development_protocol.json` held the 29 FD training events, 5 Hz eikonal carrier, seed 42, model size and two-stage update counts constant. The intervention was a smooth source-dependent onset gate **inside** the forward network during training: 5000 receiver-only steps and 1000 physics/receiver steps. The gate uses straight source-to-receiver distance, the registered maximum model speed of 6166.213 m/s, a 200 m support allowance, 0.20 s lead and 0.10 s smooth rise. It is differentiable with respect to source parameters and twice differentiable in the PDE input, but it is an approximate taper rather than an exact acoustic causal solution or an eikonal travel time.

The pretraining and hybrid runs took 33.33 s and 98.08 s, respectively, on the local GPU. All comparisons below use the same twelve **already-used development events**, the same true source and the same unshifted 5 Hz, 25-receiver, 0-2 s FD waveforms. No source inversion was performed.

| Mean across 12 events (counts pooled) | Original eikonal model | Original plus postprocessing gate | Newly trained gate |
|:--|--:|--:|--:|
| Full-record relative waveform L2 | 0.647 | 0.640 | **0.655** |
| FD-active-sample relative L2 | 0.582 | 0.582 | **0.592** |
| Finite depth-response relative L2 | 0.802 | 0.794 | **0.812** |
| Finite depth-response cosine | 0.609 | 0.614 | **0.600** |
| False picks before origin minus 0.10 s | 65 | 0 | **0** |
| Paired picks | 142 | 142 | **142** |
| Mean per-event pick RMSE (s) | 1.151 | 0.493 | **0.561** |

The trained gate improves active-sample L2 on 4/12 events and worsens it on 8/12. Its mean active L2 is 1.7% higher than the unchanged baseline. The depth-response L2 is also higher, and both the full-record L2 and picked arrival RMSE are worse than those obtained by simply applying the fixed postprocessing gate. Removing noncausal background therefore does **not** repair the source-dependent waveform operator in this experiment. This model fails the registered requirement to improve active waveforms and preserve depth response; it is not selected for independent localization. These event results cannot be presented as held-out confirmation or as a new localization accuracy result.

The complete per-event values, checked training hashes, run durations and checkpoint hashes are in `trained_onset_gate_development_results.json`. Reproduce with `examples/run_trained_onset_gate_development.py` followed by `examples/audit_trained_onset_gate_development.py`. Both original and gated checkpoints are retained, and the original model remains the reference.
