# Velocity transfer provenance audit

## Archived results and their limits

The archived `legacy/velocity_model.py` calls `train_source(train_loader,
val_loader)` and `train_target(train_loader, val_loader)` with **the same loader
objects** returned by one `load_data` call. No independent source-domain
velocity file or source checkpoint is supplied. The code therefore documents
continued fitting on a single Overthrust sample, not cross-model transfer.
The archived no-transfer script separately specifies 60 epochs and a
`2e-4` learning rate, versus up to 50 + 40 epochs and `2e-5` in the second
stage of the two-stage script. The reported 262.77 versus 203.77 m/s RMSE
cannot be attributed solely to initialization from these scripts.

The active `data/velocity/vp_model.npy` is elementwise identical to the
archived `E:/论文代码/代码/3.1/model/vp_model.npy` (shape `401 x 401 x 94`).
Against that archive's `vp_true.npy`, the full-grid RMSE is 204.16 m/s.
The source-conditioned wavefield code queries this **exported array**; it
does not load the pretrained velocity MLP or propagate derivatives to it.

## Independent-source feasibility pilot

`examples/audit_velocity_transfer.py` defines an independent analytic
one-dimensional source model:
`v(z) = 3000 + 1700 z + 300 tanh(8(z-0.55))` m/s, with normalized depth
`z in [0,1]`. The source model does not use Overthrust values. All arms use
the same 20,000 target samples, minibatches, normalization, 250 target steps,
and a spatially disjoint validation/test slab. Three fixed seeds are run.

| Initialization | Additional pretraining steps | Spatial-test RMSE mean (m/s) | Individual seed RMSEs (m/s) |
| --- | ---: | ---: | --- |
| Random | 0 | 762.8 | 749.3, 724.8, 814.4 |
| Independent layered source | 250 | 432.0 | 431.8, 432.1, 432.2 |
| Continued target fitting | 250 | 422.6 | 421.0, 426.3, 420.5 |

This short optimization pilot shows that pretraining helps relative to 250
target updates from scratch. **It does not show a source-domain advantage**:
target-only continuation has slightly lower error, and the pretrained arms
use twice as many total updates as the scratch arm. The pilot uses a smaller
64-wide, 3-hidden-layer network and a spatial slab test; it does not reproduce
the archived 256-wide, four-layer run or establish an effect on source
localization. Additional matched-total-compute comparisons and independent
geological source volumes are needed before claiming transferable geology.

Raw reproducible output: `outputs/velocity_transfer_independent_source_pilot.json`.

## Five-seed matched-update follow-up

The follow-up keeps the same 20,000 target samples, spatially disjoint
validation/test slabs, architecture (three hidden layers, width 64), and final
250 target minibatches per seed. Seeds are 17, 29, 41, 53, and 67. Four arms
start from the same initial weights within each seed:

| Arm | Training schedule | Spatial-test RMSE (m/s, mean +/- sample SD) | Spatial-test MAE (m/s, mean +/- sample SD) | Mean elapsed time (s) |
| --- | --- | ---: | ---: | ---: |
| Scratch, short | 250 target updates at 2e-5 | 763.36 +/- 46.94 | 653.14 +/- 34.28 | 0.55 |
| Scratch, matched updates | 500 target updates at 2e-5 | 476.74 +/- 15.56 | 418.99 +/- 14.42 | 0.70 |
| Independent source | 250 independent layered-source updates at 2e-4, then 250 target updates at 2e-5 | 431.38 +/- 1.62 | 381.33 +/- 3.20 | 0.67 |
| Target continuation | 250 target updates at 2e-4, then 250 target updates at 2e-5 | 419.61 +/- 5.30 | 363.45 +/- 4.64 | 0.69 |

The independent-source RMSE exceeds target continuation in every paired seed;
the mean paired difference (independent source minus target continuation) is
+11.76 +/- 5.47 m/s. The 500-update scratch control is 45.36 m/s worse on
average than independent-source pretraining, but its constant 2e-5 learning
rate differs from the two-stage schedule. Target continuation is the stricter
control for *which data* supplied the first 250 steps, because it matches
initial weights, update counts, both learning rates, optimizer reset, final
minibatches, and evaluation samples. Results do not establish an advantage
specific to independent-source transfer. The measured times are for this
small pilot only, not full-network training or per-event inversion.

Raw output, including per-seed values and paired differences:
`outputs/velocity_transfer_matched_compute_5seeds_checkpoints.json`.
The checkpoints permit a separate first-arrival localization sensitivity
check, documented in `experiments/velocity_to_fmm_localization_audit.md`.
Neither pilot evaluates the revised source-conditioned waveform PINN.
