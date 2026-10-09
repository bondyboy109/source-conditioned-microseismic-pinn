# Fixed onset taper on the paired receiver heads (development only)

This post-hoc diagnostic applies the **previously fixed** smooth earliest-signal taper from `eikonal_causal_gate_development_protocol.json` to both frozen heads from the paired learned/FMM training run. The same 12 previously inspected depth-offset FD events and two training anchors are used at their known sources. Neither the gate parameters nor model weights were fitted to these records, and this is not a source-inversion or independent validation experiment.

The fixed taper uses the maximum registered velocity (6166.213 m/s), a 200 m source support, 0.20 s lead and 0.10 s rise. It uses Euclidean source-receiver distance as an early onset bound; because the synthetic source has nonzero tails, it is **not** an exact physical causality or heterogeneous traveltime condition. Applying the gate to FD records for comparison removes mean energy fraction 1.14e-23 (maximum 1.30e-22), effectively none of the *sampled* reference signal in these twelve events.

| Mean of 12 events, except pooled counts | Learned head before | Learned after | FMM head before | FMM after |
|:--|--:|--:|--:|--:|
| Full-record waveform relative L2 | 0.680 | 0.670 | 0.744 | 0.733 |
| FD-active-sample waveform relative L2 | 0.610 | 0.610 | 0.667 | 0.667 |
| Change-from-anchor relative L2 | 0.840 | 0.828 | 0.927 | 0.917 |
| Picks before origin minus 0.10 s | 99 | 0 | 103 | 0 |
| Paired threshold picks | 142 | 142 | 142 | 142 |
| Threshold-pick RMSE (s) | 1.262 | 0.490 | 1.292 | 0.495 |

The gate removes pre-origin prediction artifacts and slightly improves the full-record metric by suppressing output where FD has no sampled signal. **It does not change FD-active waveform L2**, and both depth-response errors remain high. The reduced threshold-pick RMSE still does not establish accurate propagation-time prediction: the picker depends on the model's trace peak and the early taper, and approximately 0.49 s of error remains. FMM-supplied carrier time also remains worse than the learned carrier on these depth records. The next proposed operator must improve source-dependent waveforms and their finite depth response, with a physically justified source condition; neither a postprocessing gate nor a carrier swap meets that requirement.

This uses already-spent development data, one velocity model and receiver-supervised heads without PDE/IC/BC training. The FMM travel-time lookup is not differentiable with respect to continuous source coordinates. These results cannot update the paper's localization or uncertainty claims. The repeated 25-field FMM setup is a diagnostic cost, not per-event inverse runtime.

Fixed inputs and per-event metrics are in `fmm_receiver_head_fixed_gate_protocol.json` and `fmm_receiver_head_fixed_gate_results.json`. Run with `python examples/audit_fmm_receiver_head_fixed_gate.py`; hashes are checked and existing results cannot be overwritten.
