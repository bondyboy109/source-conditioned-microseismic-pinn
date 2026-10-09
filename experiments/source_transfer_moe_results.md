# Independent source-domain MoE transfer pilot

The source-domain velocity is the independent synthetic undulating layered model,
not a smoothed Overthrust model. Sixteen FD source-domain records were generated
at the same *training* source coordinates as the first 16 Overthrust records.
All 33 target-domain training records were used only during the target phase.
Four new Overthrust FD records were reserved for this evaluation; their source
positions were checked against the existing observation files before generation.

Three paired seeds (4101--4103) use the same model architecture, sampling,
optimizer, and target-domain FMM fields. The source-to-target arm receives 1500
source updates and 1500 target updates; the target-to-target arm receives 1500
updates on the first 16 target records and 1500 on all 33 target records.
The target-only arm receives 1500 updates on all 33 target records and is an
equal *target-label exposure* reference, not an equal-total-compute reference.
Each target phase restarts Adam; source FMM fields are replaced by target FMM
fields when transferring learned experts and gate. Training records use the
same 5-Hz FD solver and receiver geometry, with no target waveform used during
source pretraining. All localization arms use the same previously developed
arrival picker, bias correction, search grid, hybrid weight and target FMM field.

| Arm | Median hybrid location error (m) | Mean hybrid location error (m) | Median origin-time error (s) | Mean full-waveform relative L2 |
|---|---:|---:|---:|---:|
| Source to target | 545.1 | 529.8 | 0.0214 | 0.7692 |
| Target to target, equal total updates | 490.6 | 450.3 | 0.0229 | 0.7506 |
| Target only, equal target updates | 399.5 | 433.9 | 0.0285 | 0.9310 |

Each row pools 12 seed-event outcomes (3 seeds x 4 fixed test events); these are
not 12 independent geologies. The paired transfer-minus-target-to-target
location differences are positive (worse) for 8/12 outcomes. Per-seed median
hybrid location errors are 456.2/610.2/582.6 m for transfer and
456.5/443.3/501.9 m for target-to-target. The full 36 event-arm-seed estimates,
including waveform-only results, are in `source_transfer_moe_results.json`.

The pilot does not support a positive claim that independent source pretraining
improves Overthrust localization under the present 5-Hz MoE, 1500+1500 update
budget and fixed FMM arrival-constrained objective. The lower waveform L2 than
the 1500-step target-only arm did not improve the hybrid location statistic.
Four test positions, one synthetic target geology, no noise, and three seeds
are insufficient for a general negative claim. This experiment tests waveform
surrogate transfer; it does not validate the paper's older velocity-MLP
pretraining claim. The previously published zero-shot audit of the frozen
Overthrust-trained model remains a separate experiment and must not be
reinterpreted as confirmation of the retrained source-to-target model.

Reproduce in the `torch_env` environment from the repository root:

```powershell
python examples/run_source_transfer_moe.py prepare
python examples/run_source_transfer_moe.py train
python examples/run_source_transfer_moe.py evaluate
```

The protocol is `source_transfer_moe_protocol.json`. Training checkpoints are
under `outputs/source_transfer_moe/`. Preparation resumes existing records and
training skips existing checkpoints; evaluation refuses to overwrite its result.
