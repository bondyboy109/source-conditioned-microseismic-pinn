# Reconstructed velocity in first-arrival localization

## Question and controlled setup

Does independent layered-source pretraining improve source location relative
to the same two-stage training schedule using only Overthrust target samples?
We saved all 20 velocity-network checkpoints from the five-seed audit and
reconstructed each on the original 401 x 401 x 94 grid. Three separately
generated 5-Hz FD events (`independent_test_01..03_0to2s.npz`) supply the
same receiver records to each velocity model. This is a **FMM first-arrival
localization sensitivity test**, not a source-conditioned waveform-PINN result.

The search is the pre-existing FMM baseline: 40-m grid decimated by four;
8 x 8 x 5 coarse source candidates in x=2--14 km, y=2--14 km, z=0.1--1.5 km;
3 x 3 x 3 local candidates around the coarse optimum; and clipped median
origin time in [0.05,0.50] s. The three FD observations, threshold (5% of
trace maximum for three samples), search geometry, and source bounds remain
fixed for every seed and velocity. The true Overthrust velocity is a reference
for the numerical baseline, not a learned control.

| Velocity field | Spatial-slab RMSE (m/s) | Mean location error (m), five seeds +/- sample SD | Mean origin-time error (s) |
| --- | ---: | ---: | ---: |
| Random, 250 target updates | 763.36 | 1252.16 +/- 54.75 | 0.0652 |
| Random, 500 target updates | 476.74 | 1185.58 +/- 28.63 | 0.1778 |
| Independent layered source, then target | 431.38 | 1196.04 +/- 23.38 | 0.2213 |
| Target continuation, same two-stage schedule | 419.61 | 1206.49 +/- 0.00 | 0.2230 |
| True Overthrust volume | 0 (identity) | 985.82 (same model in all runs) | 0.1713 |

The mean location error is first averaged over three events *within each
seed*, then averaged over five seeds. The independent-source minus target-
continuation paired location difference is -10.45 m on average. The two
methods pick identical locations in 14 of the 15 seed/event pairs; for the
third event at seed 67, independent-source pretraining yields 1483.3 m and
target continuation 1640.1 m. For both methods, events 01 and 02 yield
1464.5 m and 514.9 m in every seed. The apparent 10-m aggregate difference
thus comes from one coarse-grid decision, not a reproducible gain across
events and initializations. The true velocity still leaves about 986 m of
average location error because the arrival picker, limited 2-s windows,
coarse search, and origin-time/position trade-off contribute errors even
without velocity reconstruction.

The velocity RMSE uses an x-slab held out from fitting; the three event
sources are not all inside that slab. This comparison does not measure
local ray-path velocity accuracy or show that better global velocity RMSE
causes better location. The 500-update random baseline holds update count
constant but uses a single learning rate; the target-continuation control
matches the complete two-stage schedule. The small 64-wide velocity network
is not the archived 256-wide velocity MLP, and neither it nor the FMM solver
feeds velocity derivatives into the wavefield PINN. No waveform-PINN
localization or uncertainty claim follows from these numbers.

## Reproduction

Activate the `torch_env` Conda environment in the repository root and save
the five-seed velocity checkpoints with:

```powershell
python .\examples\audit_velocity_transfer.py `
  --target 'E:\论文代码\代码\3.1\model\vp_true.npy' `
  --output .\outputs\velocity_transfer_matched_compute_5seeds_checkpoints.json `
  --checkpoint-dir .\outputs\velocity_transfer_checkpoints `
  --seeds 17 29 41 53 67
```

The raw five-seed audit, checkpoints, and localization outputs are
under `outputs/` (ignored by Git):

- `velocity_transfer_matched_compute_5seeds_checkpoints.json`
- `velocity_transfer_checkpoints/seed_<seed>_<arm>.pth`
- `velocity_localization_seed<seed>_independent.json`, seed 17, 29, 41, 53, 67

The localization command for a saved seed is:

```powershell
python .\examples\audit_velocity_localization.py `
  --audit .\outputs\velocity_transfer_matched_compute_5seeds_checkpoints.json `
  --truth-velocity 'E:\论文代码\代码\3.1\model\vp_true.npy' `
  --observation .\data\observations\independent_test\window_0to2s\independent_test_01_0to2s.npz `
                .\data\observations\independent_test\window_0to2s\independent_test_02_0to2s.npz `
                .\data\observations\independent_test\window_0to2s\independent_test_03_0to2s.npz `
  --output .\outputs\velocity_localization_seed17_independent.json --seed 17
```
