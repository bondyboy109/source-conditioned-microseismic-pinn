# Frozen 25-source model: one-time synthetic FD confirmation

Protocol: `depth_gap_independent_confirmation_protocol.json` was written
before generating these eight records. The 25-source 5 Hz checkpoint hash is
`3a8b5251f12c308f9899ef5557347984e09b7a3d948ffd7a3146d6242b20041a`
before and after evaluation. No model weights, training sources, loss
weights or regional search parameters were changed after registration.
The newly generated FD files and source coordinates have **zero overlap**
with the checkpoint's 25 receiver-training observations.
Their source coordinates also differ from all 108 previously archived
observation records containing source truth.

## Event-level results

The original-amplitude L2 is relative to each independently generated FD
receiver record, with no waveform gain or time-shift fitting. The six
inversions use a predefined regional 5 x 5 x 7 x 5 (875 candidate) grid,
three lowest-loss starts and 160 Adam iterations per start. They use all
25 receivers and every fourth sample in refinement. A fixed full-record
observation RMS normalizes all candidates. Regional bounds depend on the
registered anchor neighborhood, **not** the unknown event truth.

| Event | FD source (x,y,z in m; t in s) | Waveform L2 | Inverted source (x,y,z in m; t in s) | Location error (m) | Time error (s) |
| --- | --- | ---: | --- | ---: | ---: |
| a09_depth_920 | (6500,3500,920; .20) | 0.1337 | (6500.0,3498.8,926.8; .1987) | 6.95 | .0013 |
| a09_depth_960 | (6500,3500,960; .20) | 0.2357 | (6499.5,3498.4,961.0; .1973) | 1.94 | .0027 |
| a09_mixed_830 | (6430,3560,830; .19) | 0.1866 | (6437.6,3550.1,824.4; .1905) | 13.69 | .0005 |
| a09_mixed_950 | (6570,3440,950; .21) | 0.2848 | (6552.5,3451.3,957.7; .2068) | 22.24 | .0032 |
| a01_depth_720 | (3500,4000,720; .12) | 0.3467 | (3505.1,3991.9,708.8; .1182) | 14.78 | .0018 |
| a01_mixed_560 | (3570,3930,560; .14) | 0.3446 | (3563.3,3947.5,563.5; .1367) | 19.05 | .0033 |
| remote_02 | (11450,4070,960; .28) | 0.3841 | Not inverted, by protocol | -- | -- |
| remote_05 | (8100,3050,1260; .42) | 0.3026 | Not inverted, by protocol | -- | -- |

All **four preregistered gates** pass: each local raw waveform L2 is below
0.5; both remote raw L2 values are below 0.75; each of six local position
errors is below 100 m and each origin-time error below 0.02 s. The local
mean position error is **13.11 m** (maximum **22.24 m**), and mean origin-
time error is **0.0021 s** (maximum **0.0033 s**). Eight-event mean waveform
relative L2 is 0.2774 (worst 0.3841). The six selected parameter vectors
are inside their preassigned search boxes, with no boundary hit. Each
event's true-source waveform misfit, three coarse starting candidates,
their final misfits, and predicted coordinates appear in the raw JSON.
The best modeled misfit is lower than the true-source misfit in every local
event, reflecting remaining forward-model error even where the recovered
location is close.

## Cost and limits

The eight fresh 2 s, 25-receiver FD solves took a combined **68.78 s** of
solver GPU time (about 8.60 s per event) on the same RTX 4060 Laptop GPU.
The six PINN inversions took **8.87 s** total (1.24--2.44 s each, including
the 875-candidate search and three refinements). These are solver/search
timings within the scripts; process startup, FD dataset generation for the
25 training sources, data loading and source-conditioned PINN training
are outside this direct per-event comparison. The result **does not**
establish a favorable end-to-end PINN/FD break-even point.

The FD event metadata's `source_truth` field is checked against the
pre-registered coordinates as a **file-integrity check** before search.
The search function itself receives only the waveform, times, receivers
and preassigned region. Source truth is used again **after** search to
calculate errors. This provenance check is the sole non-inversion use of
truth, clarifying the protocol's shorthand “post-inversion error computation
only”; it does not set bounds, candidates, starting points or losses.

This is an independent-of-training, **one-time synthetic** confirmation
after extensive development on the *same* Overthrust velocity grid and
FD source configuration. Source locations were preregistered, but they
are concentrated near known training neighborhoods, so the result does
not prove location accuracy across the full 3-D volume. The two remote
events were **not** inverted; they support only forward-waveform comparison.
There is one training seed, one velocity model, no noise realizations,
no field records, no high-frequency validation and no calibrated UQ.
Sub-grid-looking location errors must be interpreted in light of the
40 m FD grid, smooth spatial source injection and synthetic consistency;
they are not a physical resolution or uncertainty estimate. These eight
records have now been seen and cannot be reused as untouched tests for
a revised checkpoint.

Full machine-readable output:
`depth_gap_independent_results.json`,
`depth_gap_independent_forward.json`, and
`depth_gap_independent_<event>.json` for every event. To reproduce from
the project root (existing files are verified and reused):

```powershell
& 'E:\soft\Anaconda3\envs\torch_env\python.exe' examples\confirm_depth_gap_25.py
```
