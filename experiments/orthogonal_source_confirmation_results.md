# Frozen-checkpoint orthogonal source confirmation

This one-time synthetic FD evaluation follows
`orthogonal_source_confirmation_protocol.md`, written before generating the
records. Both 5 Hz checkpoints were frozen: the selected 20-event local-
density model and its paired 16-event control. The ten new sources differ
from both training anchors only in one spatial coordinate. The FD solver,
velocity grid, 0--2 s time axis, boundaries, and 25 receivers are unchanged.
Checkpoint hashes match the protocol. Neither the files nor their exact
source coordinates appear in either checkpoint's receiver-training metadata.

| New FD source relative to anchor | 16-event raw L2 | 20-event raw L2 | 20-event peak / FD peak |
| --- | ---: | ---: | ---: |
| train_01, y -120 m | 0.3866 | 0.4088 | 1.053 |
| train_01, y +120 m | 0.3881 | 0.3992 | 1.020 |
| train_01, z -80 m | 0.5451 | **0.5307** | 0.728 |
| train_01, z +80 m | 0.5342 | **0.5852** | 1.362 |
| train_01, x +400 m | 0.8887 | 0.4284 | 0.997 |
| train_09, y -120 m | 0.2685 | 0.2553 | 1.064 |
| train_09, y +120 m | 0.2577 | 0.2424 | 0.965 |
| train_09, z -80 m | 0.4604 | 0.4605 | 0.921 |
| train_09, z +80 m | 0.4134 | 0.4189 | 0.992 |
| train_09, x +400 m | 0.7304 | 0.2649 | 0.921 |
| **Mean of all ten** | **0.4873** | **0.3994** | -- |

The selected 20-event model reduces the overall mean raw L2 by 18.0%,
principally through the two x +400 m events (mean 0.8095 to 0.3466).
Mean y-direction error is essentially unchanged (0.3252 to 0.3264), and
mean depth error grows slightly (0.4883 to 0.4988). Eight of ten selected-
model records have L2 below 0.5, but *both* train_01 depth records fail
the preregistered 0.5 threshold. All ten are below the zero-prediction
baseline of 1.0. The prespecified **across-direction local feasibility gate
fails**; a low mean alone would conceal that failure.

Finite-displacement waveform-response diagnostics support the same limit.
For the selected model, x +400 m change-relative L2 is 0.280 at train_01
and 0.157 at train_09. The y-direction response error is 0.595--0.598 at
train_01 and 0.309--0.313 at train_09. For the four depth perturbations,
it is **1.010, 1.114, 1.220, and 1.358**, with change cosine only
0.211, 0.165, 0.090, and 0.071 respectively. These ratios compare the
predicted waveform *change* with the FD change at a fixed anchor; the FD
change is nonzero. The depth response is therefore poorly directed, not
just uniformly scaled. Fitted common shifts or gains in the audit JSON
are diagnostics and are excluded from all results above.

The ten new 2 s FD solves took a cumulative 86.46 GPU seconds (about
8.65 s per source), with a maximum recorded peak allocation of 512.1 MiB.
No checkpoint was retrained or selected after seeing these results. This
is a one-time synthetic confirmation at two source neighborhoods in the
same Overthrust model, **not** field validation, multi-seed evidence,
all-region coverage, localization accuracy, or calibrated UQ. These
events have now been inspected and cannot serve as untouched confirmation
events for a future revision. Spatially dense supervision improves the
tested x direction but leaves depth sensitivity unsuitable for a general
four-parameter inversion claim. Further development should diagnose and
model depth-dependent propagation using training/development data, then
register *new* confirmation sources only after freezing the new model.

Machine-readable records:

* `experiments/orthogonal_source_{control16,aug20}_confirmation.json`
* `experiments/orthogonal_source_{control16,aug20}_response_{01,09}.json`
* `data/observations/orthogonal_confirmation/{event}.npz` and matching FD JSON

Reproduction (PowerShell, project root, `torch_env` activated):

```powershell
$fd = Get-ChildItem data/observations/orthogonal_confirmation/*.npz | Sort-Object Name | ForEach-Object FullName
python examples/audit_forward_error_decomposition.py outputs/local_density_aug20_scratch_hybrid_1000/conditioned_5Hz.pth $fd --evaluation-label synthetic-confirmation --output experiments/orthogonal_source_aug20_confirmation.json
$a01 = Get-ChildItem data/observations/orthogonal_confirmation/a01_*.npz | Sort-Object Name | ForEach-Object FullName
$a09 = Get-ChildItem data/observations/orthogonal_confirmation/a09_*.npz | Sort-Object Name | ForEach-Object FullName
python examples/audit_spatial_source_response.py outputs/local_density_aug20_scratch_hybrid_1000/conditioned_5Hz.pth data/observations/window_0to2s/train_01_0to2s.npz $a01 --output experiments/orthogonal_source_aug20_response_01.json
python examples/audit_spatial_source_response.py outputs/local_density_aug20_scratch_hybrid_1000/conditioned_5Hz.pth data/observations/window_0to2s/train_09_0to2s.npz $a09 --output experiments/orthogonal_source_aug20_response_09.json
```
