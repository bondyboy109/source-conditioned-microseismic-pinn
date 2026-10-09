# Training-event FD regeneration control

To test whether the old training waveforms and the current independent-test
waveforms use different numerical generation paths, regenerate the already
trained 5 Hz event at (12000, 12000, 400) m and origin time 0.18 s with the
current FD generator. The regenerated record is saved **separately** at
`outputs/train_04_regeneration_control/regenerated_train_04.npz`; neither the
original training record nor the frozen PINN checkpoint is modified. The
comparison uses the frozen checkpoint's training manifest and exact file
hashes. This training source was chosen for a post-hoc pipeline check after
the new-event results were inspected.

The original nine-second simulation is
`data/observations/train_04.npz`. Its first 1001 samples exactly match the
two-second training record
`data/observations/window_0to2s/train_04_0to2s.npz`. The new two-second
simulation has identical source parameters, all 25 receiver coordinates,
1001 time samples, 5 Hz source description, 40 m velocity geometry, CFL,
zero-Dirichlet boundaries and trilinear receiver sampling.

| Comparison | Result |
|:--|--:|
| Waveform values different | **0 / 25,025** |
| Raw waveform array bitwise equality | **Yes** |
| Relative waveform L2 | **0.000000** |
| Maximum absolute sample difference | **0.000000** |
| Original training record equals 9 s parent's first 2 s | **Yes** |

The original training input also matches the SHA-256 recorded in the frozen
29-source checkpoint. This exact replay rules out a difference between
cropping the original nine-second simulation and running the current
generator for two seconds **for this one source**. Combined with the
previously measured training L2 around 0.145 versus new-source median L2
1.082, it supports investigating source-location generalization and waveform
physics. It does not prove that all historical training records were generated
identically, nor does it repair the frozen forward model or provide a new
independent localization result.

Reproduce the simulation with the `torch_env` Python interpreter:

```powershell
python examples/generate_fd_observation.py --duration 2 --source 12000 12000 400 0.18 --output outputs/train_04_regeneration_control/regenerated_train_04.npz
python examples/audit_train04_fd_regeneration.py
```

The auditor intentionally refuses to replace its existing result. Exact
hashes, provenance checks and numeric values are recorded in
`experiments/train04_fd_regeneration_control.json`. Previously used test
events remain development/diagnostic data; subsequent model changes require
new untouched source locations for a fresh independent evaluation.
