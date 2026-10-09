# Local 5 Hz source-density control: development results

This experiment follows `local_source_density_protocol.md`. Four new FD
*training* sources were placed at x = 3300 and 3700 m around train_01, and
x = 6300 and 6700 m around train_09, each 200 m from its anchor. Two newly
generated FD events near train_09 (+40 and +160 m) and five previously
recorded displacements near train_01 (+40, +80, +160, +250, and +800 m) were
held out for development. Both checkpoints' metadata contain **zero** of
the seven development files or exact source coordinates. No independent
test events were inspected for this experiment.

## Why the initial continuation pilot was inconclusive

Starting from the existing 16-event receiver-pretrained checkpoint, we ran
1000 identical hybrid updates with either the original 16 or the expanded
20 events. The 20-event model had inadequate training-anchor fit: original
train_01 error rose from 0.0978 to 0.3107; train_09 rose from 0.0974 to
0.1508. Development +40 m errors also worsened at both anchors and the
800 m error increased from 0.9832 to 1.1726. An insufficient adaptation
budget confounds this comparison; these pilot outputs do not determine the
effect of local source coverage. Their raw records are
`local_density_{control16,local20}_{development,training_audit,response_01,response_09}.json`.

## Paired from-scratch two-stage control

Both groups were initialized from the same seed and architecture. First,
receiver-only pretraining ran for 5000 updates at learning rate 1e-4 with
weight 0.1 and 128 receiver samples per event (batch 2048/2560). Then each
model was initialized from its own final receiver checkpoint and trained
for 1000 hybrid PDE/IC/BC/receiver updates, learning rate 1e-5, receiver
weight 0.03, 64 receiver samples per event (batch 1024/1280). The physics
sampling counts were 256 interior, 128 initial, and 16 per boundary face.
The extra data increase the receiver batch and its compute cost; they are
not free source-independent physics supervision.

| Development event | 16-event raw L2 | 20-event raw L2 |
| --- | ---: | ---: |
| train_01 +40 m | 0.1675 | 0.0993 |
| train_01 +80 m | 0.2945 | 0.1091 |
| train_01 +160 m | 0.5340 | 0.1015 |
| train_01 +250 m | 0.7283 | 0.1354 |
| train_01 +800 m | 1.0188 | 0.9685 |
| train_09 +40 m | 0.1313 | 0.0847 |
| train_09 +160 m | 0.3577 | 0.0860 |
| **Arithmetic mean (seven events)** | **0.4617** | **0.2264** |

The mean drops by 50.98%, both +40 m events improve, and the 800 m event
does not exceed the zero-prediction raw L2 baseline of 1.0. These outcomes
meet the prespecified **development-only** improvement criterion. Original
training-anchor fit is retained: train_01 L2 changes from 0.0913 to 0.0943
and train_09 from 0.0890 to 0.0842. All four newly added training sources
fit at L2 = 0.0886--0.0996 in the augmented model.

Finite spatial response error (norm of PINN-minus-FD waveform *change*,
divided by the FD-change norm) falls from 0.4356 to 0.1319 at train_01
+40 m, 0.4552 to 0.1108 at +250 m, and from 0.3273 to 0.0832 at
train_09 +40 m. At train_01 +800 m it remains 0.6198 (control: 0.6526).
These are finite displacements, not validated infinitesimal source
gradients. The +800 m source is still 600 m from the nearest new training
source and has near-zero-baseline waveform quality. The result supports
**local interpolation with denser FD supervision**; it does not establish
accuracy throughout the volume, near faults, at higher frequencies, or
under noise, and does not pass a useful full-domain localization gate.

## Measured cost and scope

| Component | 16 events | 20 events |
| --- | ---: | ---: |
| 5000 receiver-pretraining updates | 21.82 s | 23.80 s |
| 1000 hybrid updates | 46.62 s | 47.17 s |
| Training peak GPU allocation, pretraining / hybrid | 37.2 / 112.8 MiB | 42.8 / 116.3 MiB |
| Additional FD training events | 0 | 4 x ~8.6 s = 34.36 s |

The total measured *additional* time for the augmented workflow is about
37 s (new FD training records plus additional training time) on this GPU,
excluding data loading, process startup, and two new FD development records
(17.18 s). These figures do not imply a PINN advantage over FD: the workflow
still needs costly FD examples near the sources it predicts well. No
independent confirmation, localization accuracy, or uncertainty coverage
is claimed. The next decision should account for FD costs per new region
and compare local coverage against a source-parameterized physical forward
solver before using this as a monitoring workflow.

Checkpoint and raw records:

* `outputs/local_density_{control16,aug20}_scratch_receiver_5000/`
* `outputs/local_density_{control16,aug20}_scratch_hybrid_1000/`
* `experiments/local_density_{control16,aug20}_scratch_development.json`
* `experiments/local_density_{control16,aug20}_scratch_training_audit.json`
* `experiments/local_density_{control16,aug20}_scratch_response_{01,09}.json`

To reproduce from the project root using the installed PyTorch environment,
generate the six FD records with `examples/generate_fd_observation.py` and
the source coordinates in the protocol. Train both receiver models with
`--receiver-only --travel-time-carrier --distance-core-m 20 --wavefield-scale 1e-5`
and the stage parameters above. Run the hybrid stage with `--initial-checkpoint`
pointing to each respective final receiver checkpoint and remove
`--receiver-only`; set `--n-phys 256 --n-ic 128 --n-bc-per-face 16`. Use
`examples/audit_forward_error_decomposition.py` and
`examples/audit_spatial_source_response.py` on the seven held-out development
records. Each checkpoint contains exact run settings and training-file hashes.
