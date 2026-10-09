# Overthrust source-conditioned forward calibration

This development run uses the 16 registered training events (`train_01` to
`train_16`) and evaluates only the three predeclared development events at
250, 800, and 2000 m source displacement. The previously inspected
`heldout_01` and `blind_01`--`blind_03` files are excluded from all tuning.
The records use the 0--2 s window and the independent second-order FD solver.

The heterogeneous velocity model requires absolute source coordinates in the
carrier. The translation-invariant carrier used for the uniform calibration is
therefore not used here. The Overthrust model uses the absolute-source
travel-time carrier with a 20 m smooth distance core and a 3500 m/s reference
velocity.

| Training protocol | train_01 L2 | train_08 L2 | train_16 L2 | dev 250 m | dev 800 m | dev 2000 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-event receiver batches, 5000 steps | 0.480 | 0.373 | 0.497 | 0.857 | 1.252 | 1.028 |
| 16-event joint receiver batches, 5000 steps | 0.116 | 0.102 | 0.117 | 0.760 | 1.001 | 0.934 |
| Joint batches + 1000-step physics/IC/BC fine-tuning | 0.094 | 0.087 | 0.095 | 0.782 | 1.244 | 1.294 |

Joint batches materially improve the fitted training events and reduce the
250 m development error. They also make the 2 km development error slightly
better than the zero-wavefield baseline, but the 800 m result is effectively
at the baseline. Adding the current PDE loss improves the fixed source-01
neighbourhood audit (active-forcing relative residual RMSE `0.570`, source-peak
residual `-0.293`) but degrades cross-source waveform generalization.

The current best forward checkpoint is
`outputs/overthrust_core20_train16_jointbatch_receiver_5hz_5000/conditioned_5Hz.pth`.
It has 67,074 trainable parameters and was trained on the RTX 4060 Laptop GPU.
The hybrid checkpoint is retained for the physics diagnostic at
`outputs/overthrust_core20_train16_jointbatch_hybrid_5hz_1000/conditioned_5Hz.pth`.

These results do not yet pass the forward-model gate for localization: one of
the three development events remains at or above the zero-prediction baseline,
and the heterogeneous-model PDE fine-tuning is not compatible with stable
cross-source generalization. The next controlled experiment is a reference
velocity sweep using the same 16-event joint-batch protocol and fixed
development set; blind events remain untouched.
