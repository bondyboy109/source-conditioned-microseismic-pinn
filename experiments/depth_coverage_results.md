# Local depth-coverage development results

This development experiment tests whether adding four 5 Hz FD receiver records
at `z = 340, 660, 540, 860 m` improves the depth response around the two
anchors `(3500, 4000, 500, 0.12)` and `(6500, 3500, 700, 0.20) m`. The control
uses the exact 20-event local-density training manifest. The augmented model
uses those same 20 events plus the four registered depth records. Both groups
were trained from scratch with seed 42, 5000 receiver-only updates followed by
1000 hybrid updates. The four `z +/- 240 m` records below were generated only
after the training manifest was fixed and were not included in either model.

## Development waveform errors

| Event | 20-event control raw L2 | 24-event depth raw L2 | Response relative L2 | Response cosine |
| --- | ---: | ---: | ---: | ---: |
| train_01, z=260 m | 0.9139 | 0.3964 | 0.4225 | 0.9112 |
| train_01, z=740 m | 1.2457 | 0.4131 | 0.3578 | 0.9353 |
| train_09, z=460 m | 0.8602 | 0.3746 | 0.4963 | 0.8849 |
| train_09, z=940 m | 0.8521 | 0.4080 | **0.5104** | 0.8611 |
| **Mean** | **0.9680** | **0.3980** | -- | -- |

The augmented model reduces the mean raw waveform relative L2 by 58.88%.
All four waveform errors are below 0.5 and all four response cosines exceed
0.8. However, the `train_09, z=940 m` response relative L2 is 0.5104, just
above the prespecified 0.5 gate. Therefore the complete depth-development
gate is **not fully passed**. The result supports a substantial local benefit
from adding depth coverage, while leaving a measurable error at the deeper
edge of the tested neighborhood.

No development file overlaps either checkpoint's receiver-training manifest.
The previously inspected `z +/- 80 m` records were not reused.

## Training fit and measured cost

The 24-event augmented model retains the two anchor fits at raw L2 0.1284
(`train_01`) and 0.1026 (`train_09`). Its four added training sources have raw
L2 0.1011--0.1219. The corresponding control anchor errors are 0.0943 and
0.0842; the small increase is recorded rather than hidden.

| Component | 20-event control | 24-event depth group |
| --- | ---: | ---: |
| Receiver pretraining | 23.85 s; 42.8 MiB | 25.43 s; 47.3 MiB |
| Hybrid training | 46.70 s; 116.3 MiB | 47.72 s; 119.9 MiB |
| Trainable parameters | 67,458 | 67,458 |
| Additional FD training records | -- | 4 x about 8.61 s |

The FD development records each took about 8.60 s on the RTX 4060 Laptop GPU.
The two final checkpoints use the same architecture, optimizer schedule,
physics sample counts, seed, reference velocity (3500 m/s), distance core
(20 m), and wavefield scale (`1e-5`).

## Interpretation and next decision

This is a development result, not a localization-accuracy result. It shows
that local depth supervision can correct the poor depth response observed in
the 20-event model, but the strict response-error gate is not fully satisfied.
The four development events should not be presented as blind confirmation.
Before making a publication-level claim, either extend depth coverage and
pre-register a new independent confirmation set, or narrow the claim to local
depth interpolation around the two tested anchors. No four-parameter source
localization or calibrated uncertainty claim follows from this experiment.

Machine-readable files:

* `depth_coverage_control20_development.json`
* `depth_coverage_aug24_development.json`
* `depth_coverage_control20_response_01.json`, `depth_coverage_control20_response_09.json`
* `depth_coverage_aug24_response_01.json`, `depth_coverage_aug24_response_09.json`
* `depth_coverage_control20_training_audit.json`, `depth_coverage_aug24_training_audit.json`

Reproduction from the project root:

```powershell
& 'E:\soft\Anaconda3\envs\torch_env\python.exe' examples\prepare_depth_coverage.py --generate
& 'E:\soft\Anaconda3\envs\torch_env\python.exe' examples\run_depth_coverage.py
```
