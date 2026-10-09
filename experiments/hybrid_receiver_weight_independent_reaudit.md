# Fixed 0.03 hybrid checkpoint: previously evaluated independent events

The hybrid receiver-loss weight 0.03 was selected by the three development
events documented in `hybrid_receiver_weight_sweep_development.md`. We froze
the checkpoint `outputs/receiver_weight_0.03_hybrid_1000/conditioned_5Hz.pth`
and evaluated the original-amplitude relative L2 on the three registered
independent-test FD records. Checkpoint metadata confirms that neither the
observation files nor their source coordinates are in the 16 receiver-training
events. These independent events were already evaluated with earlier
checkpoints; this is a post hoc cross-model re-audit, **not** a fresh blind
validation or evidence for tuning another checkpoint.

| Event | Raw relative L2 | Peak amplitude ratio | Late (1--2 s) relative L2 |
| --- | ---: | ---: | ---: |
| independent 01 | 1.0440 | 0.513 | 1.1770 |
| independent 02 | 0.9874 | 0.216 | 0.9769 |
| independent 03 | 0.9142 | 0.178 | 0.9349 |

Mean raw relative L2 is 0.9819. One event exceeds the zero-prediction
baseline of 1.0, and predicted peaks are only 18--51% of FD peaks. This
checkpoint therefore fails the previously used gate requiring *every*
independent event to have raw relative L2 below 1.0. Global shift-and-gain
diagnostics in the JSON are fitted to each observation and cannot be used
as forward validation. The development-only `heldout_01_0to2s.npz` was also
audited (raw relative L2 0.8344), but is excluded from this table because
`experiments/5hz_protocol.json` registers its source as development-only.

These results do not justify reporting source-location accuracy or uncertainty
coverage for this checkpoint. The next methodological step is to improve
cross-source waveform amplitude and late-coda fidelity on training and
development sources, then freeze the model and use newly generated,
preregistered independent events for confirmation.

Raw event-level results: `receiver_weight_0.03_hybrid_independent_reaudit.json`.
