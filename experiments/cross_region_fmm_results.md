# Frozen 25-source PINN versus receiver-field FMM: cross-region confirmation

The one-time eight-event comparison **failed its predeclared overall gate**.
The fixed 5 Hz source-conditioned PINN localized five of eight new sources
within 500 m, compared with a required six, and its median location error
was 311.1 m versus 191.8 m for the FMM first-arrival baseline. These events
were generated after the source list, search settings, FMM picking rule and
thresholds were fixed in `cross_region_fmm_protocol.json`. They cannot serve
as untouched confirmation events for a revised model.

Both methods used the same 25-receiver, 0--2 s, 5 Hz FD records from the same
Overthrust velocity model, but they use **different observations**: PINN fits
full waveforms; FMM fits threshold-picked first arrivals and has access to
the true heterogeneous velocity grid. The PINN checkpoint and velocity SHA-256
hashes match the registered hashes. All eight test files have verified
hashes and distinct source coordinates; none appears in PINN receiver training.
The searches do not receive the true source; it is used for input integrity
checks and post-search error reporting.

| Event | PINN location (m) | FMM location (m) | PINN origin time (s) | FMM origin time (s) | PINN raw waveform relative L2 | Valid FMM picks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| region_02 | 27.4 | 196.8 | 0.0135 | 0.1249 | 0.6487 | 10 |
| region_03 | 52.9 | 222.0 | 0.0013 | 0.0826 | 0.6935 | 9 |
| region_04 | 485.5 | 186.9 | 0.0512 | 0.0648 | 0.6496 | 9 |
| region_05 | 96.6 | 159.5 | 0.0011 | 0.1291 | 0.6519 | 10 |
| region_06 | 136.7 | 154.6 | 0.0097 | 0.1264 | 0.5345 | 11 |
| region_07 | 1855.2 | 82.0 | 0.2535 | 0.0600 | 0.7740 | 11 |
| region_08 | 721.6 | 314.6 | 0.0174 | 0.0838 | 1.2627 | 8 |
| central_gap | 763.8 | 261.5 | 0.0182 | 0.1072 | 0.9313 | 13 |
| **Median** | **311.1** | **191.8** | **0.0154** | **0.0955** | **0.6727** | -- |

The PINN met three of six predeclared gates: six of eight origin-time errors
were below 0.05 s (exactly six), its median location error was below 500 m,
and every FMM event had at least six usable picks. It failed the all-eight
waveform L2 < 0.75 gate (five passed), the six-of-eight location < 500 m gate
(five passed), and the requirement to beat the FMM median location error.
The corresponding full machine-readable estimates, including truth values,
are in `cross_region_fmm_comparison.json`.

The failures cannot be assigned to one cause. At `region_08`, the PINN's
raw L2 at the true source is 1.2627, above even a zero-prediction L2 of 1;
at `central_gap` it is 0.9313. These are substantial forward-model errors
at the known true parameters. At `region_04` and `region_07`, the true-source
normalized search MSE (0.4233 and 0.6001) is **lower** than the returned
estimate's MSE (0.7777 and 0.8841), indicating that the fixed coarse-to-fine
search also missed better candidates under its own objective. The `region_07`
estimate hits the y = 14,000 m search boundary and has 1855.2 m position
error; the other seven returned estimates do not hit a registered bound.
These findings do not isolate the relative contributions of optimization and
forward-model error; a better search alone would not establish waveform
accuracy across the volume.

Earlier one-time confirmation in `depth_gap_independent_confirmation_results.md`
reported 1.94--22.24 m errors for six sources near known training anchors,
searched within preassigned local boxes. The present search covers the common
x,y = 2--14 km, z = 0.1--1.5 km and origin-time = 0.05--0.50 s domain, at
different source locations. Thus the local result does not generalize into
a claim of accurate full-region localization.

## Measured cost and scope

The PINN's eight searches took 31.42 s total (median 3.764 s/event), plus
0.089 s to load its frozen checkpoint; its evaluation peak CUDA allocation
was 51,290,112 bytes (~48.9 MiB). FMM computed and cached 25 receiver-centered
traveltime fields once in 36.88 s (~181.1 MiB cache), then searched all eight
events in 0.672 s total (median 0.084 s/event). Thus PINN takes ~44.8 times
as long per event by the median search time, although FMM has an up-front field
setup cost. The eight FD test-record simulations took 68.84 s in total;
these are **data-generation** costs, not an FMM search cost. PINN training and
the FD simulations for its 25 training sources are additional, excluded costs.
These measurements do not demonstrate an end-to-end PINN speed advantage or
a measured PINN-versus-FD inversion break-even point.

This is a single-seed, single-velocity-model synthetic comparison at 5 Hz,
without noise, field data or calibrated uncertainty. FMM benefits from the
true velocity and depends on its fixed first-arrival picker; PINN uses full
waveforms and depends on training coverage. Neither baseline is a substitute
for a matched-input, fully costed study. The supported finding is local
interpolation near covered training sources, with unreliable cross-region
localization under the current frozen model and global search.

Protocol and raw outputs: `cross_region_fmm_protocol.json`,
`cross_region_fmm_comparison.json`, `cross_region_fmm_baseline.json`,
`cross_region_forward.json`, and `cross_region_pinn_*.json` in this directory.
