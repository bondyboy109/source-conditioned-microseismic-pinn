# Overthrust traveltime-interface comparison

All rows use the same 16 registered training events, 5 Hz, 0--2 s records,
joint receiver batches containing all 16 events, 67,074 trainable wavefield
parameters, and the same three development events. `blind_01`--`blind_03` and
`heldout_01` were not used.

| Traveltime interface | dev 250 m L2 | dev 800 m L2 | dev 2000 m L2 | Notes |
| --- | ---: | ---: | ---: | --- |
| Uniform 3500 m/s carrier | 0.760 | 1.001 | 0.934 | Best current baseline |
| Uniform 4000 m/s carrier | 0.688 | 1.068 | 1.011 | Improves only the nearest development source |
| Uniform 4500 m/s carrier | 0.816 | 1.353 | 1.268 | Worse at medium and long offsets |
| Straight-line velocity-integrated carrier | 0.926 | 1.264 | 1.026 | Uses 16-point path quadrature |
| Frozen source-conditioned eikonal carrier | 0.801 | 1.038 | 0.996 | Eikonal residual RMSE 9.20% |

The eikonal network is a factorized model with a 20 m smooth source core. In a
constant 3500 m/s calibration its analytic traveltime relative L2 is
`2.01e-5`, with mean absolute error 0.040 ms. On Overthrust, an independent
20,000-point audit gives relative eikonal residual RMSE `0.09197`, MAE
`0.07526`, and 95th percentile `0.17114`.

The frozen eikonal interface preserves source conditioning and differentiability
but does not recover the full waveform generalization required here. The
remaining error is consistent with unresolved curved and multipath arrivals,
which are not represented by the first-order eikonal residual alone. The
current forward gate therefore remains closed: no source localization or UQ
claim should be made from these checkpoints.

Relevant checkpoints:

- `outputs/overthrust_core20_train16_jointbatch_receiver_5hz_5000/conditioned_5Hz.pth`
- `outputs/overthrust_ray16_corrected_core20_train16_jointbatch_receiver_5hz_5000/conditioned_5Hz.pth`
- `outputs/overthrust_eikonal_carrier_train16_jointbatch_receiver_5hz_5000/conditioned_5Hz.pth`
- `outputs/eikonal_overthrust_5000/eikonal.pth`
