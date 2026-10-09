# Local source-density pilot protocol (development only)

Question: does adding 5 Hz FD receiver supervision 200 m either side of two
existing training sources reduce cross-source waveform and finite spatial
response error, when optimizer updates and per-event receiver samples are
held constant? This is a density diagnostic, not a validation of PINN
localization or a computation-cost advantage.

The 16 original `window_0to2s/train_01`--`train_16` records form the
control training set. The augmented set adds four **training** FD events:

| Anchor | New source (x, y, z, origin time) | Role |
| --- | --- | --- |
| train_01 (3500, 4000, 500, 0.12) | (3300, 4000, 500, 0.12) | train |
| train_01 | (3700, 4000, 500, 0.12) | train |
| train_09 (6500, 3500, 700, 0.20) | (6300, 3500, 700, 0.20) | train |
| train_09 | (6700, 3500, 700, 0.20) | train |

Two additional events remain **development-only**: (6540, 3500, 700,
0.20) and (6660, 3500, 700, 0.20). Existing development records at
train_01 +40, +80, +160, +250, and +800 m also remain evaluation-only;
in particular (3660, 4000, 500, 0.12) is not a training source. No
independent-test event will be inspected or used for selection in this pilot.

Both runs initialize from
`outputs/overthrust_core20_train16_jointbatch_receiver_5hz_5000/conditioned_5Hz.pth`.
Use the same seed 42, uniform 3500 m/s 5 Hz travel-time carrier, 20 m core,
PDE/IC/BC counts 256/128/16 per face, learning rate 1e-5, 1000 updates,
and receiver weight 0.03. Include *all* training events every step, sampling
64 receiver points per event: batch size 1024 for 16 events and 1280 for
20 events. The physics samples and total optimizer updates remain equal;
the augmented batch has 25% more receiver points and therefore a higher
per-step compute cost. The control is the existing
`outputs/receiver_weight_0.03_hybrid_1000/conditioned_5Hz.pth`, whose
settings must be verified against this protocol.

The inherited-checkpoint 1000-step pilot is a preliminary adaptation test.
If the augmented run does not retain the original training-anchor fit, its
comparison is inconclusive about the density intervention. In that case,
run the following prespecified diagnostic on *both* groups: train from
scratch with seed 42 for 5000 receiver-only updates at learning rate 1e-4,
weight 0.1, and **128 receiver points per event** (batches 2048 and 2560),
then initialize separate hybrid 1000-step runs from those respective
checkpoints with the same settings above and 64 points per event. This
fully paired two-stage experiment supersedes the inherited-checkpoint pilot
for conclusions about local-density benefit. It uses development records
only for final fixed-checkpoint assessment, not checkpoint selection.

Primary descriptive outcome: the arithmetic mean of **raw** original-
amplitude relative L2 over the seven prespecified development events,
compared at the fixed final checkpoint. Also report every event, original
training-anchor fit, FD-versus-PINN finite source-response relative L2 at
both anchors, peak GPU memory, wall time, and FD data-generation cost.
For this pilot, an improvement is meaningful only if mean development L2
falls at least 20%, neither anchor's +40 m event gets worse, and the
800 m result does not cross above the zero-prediction baseline of 1.0.
This criterion is diagnostic and does not certify performance across other
geometries, random seeds, frequencies, or independent events.
