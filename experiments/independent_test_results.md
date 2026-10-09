# Independent Overthrust test results

These events were generated after model selection and were not used for
training, checkpoint selection, or parameter tuning. The FD configuration was
the same as the registered 5 Hz experiment: 40 m grid spacing, 0.002 s time
step, 25 receivers, and six zero-Dirichlet boundaries. Each 9 s record was
cropped to the trained 0--2 s window.

## Fixed model and inversion settings

- Checkpoint: `outputs/overthrust_eikonal_hybrid_train16_5hz_continued3000/conditioned_5Hz_epoch_2000.pth`
- Carrier: frozen source-conditioned eikonal traveltime network.
- Coarse grid: `8 x 8 x 5 x 6` in `(x, y, z, ts)`.
- Local optimization: bounded Adam, 300 iterations, learning rate `5e-3`.
- Receiver stride: 2; time stride: 4.
- Zero-prediction relative L2 baseline: `1.0`.

## Results

| Event | True source `(x,y,z,ts)` | Forward relative L2 | Estimated source `(x,y,z,ts)` | `eloc` (m) | `et` (s) |
| --- | --- | ---: | --- | ---: | ---: |
| independent 01 | (2500, 13000, 1100, 0.16) | 0.907 | (2000.1, 13999.9, 1499.99, 0.2300) | 1187.3 | 0.0700 |
| independent 02 | (7500, 7500, 1400, 0.31) | 1.005 | (8858.2, 8858.1, 1150.01, 0.2300) | 1936.9 | 0.0800 |
| independent 03 | (13500, 3000, 300, 0.42) | 1.720 | (13999.9, 3714.4, 1150.00, 0.3200) | 1217.7 | 0.1000 |

The independent forward mean relative L2 is `1.210`, and the mean location
error is `1447.3 m`. The estimates frequently lie on or near coarse-grid
locations, and origin time errors are close to the coarse time spacing. These
results show that explicit source conditioning fixes the former missing-input
problem but does not yet provide reliable cross-source waveform or location
recovery in the heterogeneous Overthrust model.

## Candidate-ranking audit

For each independent event, the true source was compared with the 16 registered
training-source candidates using the same forward misfit. The true source
ranked first for independent tests 01 and 02, but not for test 03. For test 03
the true-source relative L2 was `1.7195`, whereas the best training-source
candidate had relative L2 `1.0828`. Thus the source-conditioned network retains
some source discrimination, but its edge/shallow extrapolation is not reliable.

The corresponding FD records are in
`data/observations/independent_test/`; forward and inversion audit JSON files
are in the selected checkpoint output directory.
