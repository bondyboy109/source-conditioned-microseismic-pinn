# Spatial source response near a training event (development only)

We froze the 5 Hz hybrid checkpoint
`outputs/receiver_weight_0.03_hybrid_1000/conditioned_5Hz.pth`. Three new FD
development records move the source of training event 01 by exactly 40, 80,
and 160 m in x. They retain y = 4000 m, z = 500 m, and origin time 0.12 s,
with the same velocity model, receivers, FD solver, boundaries, and 0--2 s
window. Existing 250 and 800 m development events follow the same path.
The source remains outside the receiver-training set for all displaced
events. Regenerating the zero-displacement FD record yielded exactly the
same data and geometry as the original training record (relative L2 = 0).

| Source displacement | PINN vs FD raw L2 | PINN/FD peak ratio | FD change / original FD norm | PINN/FD change norm | Change cosine | Change relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 m (trained) | 0.0978 | 0.998 | -- | -- | -- | -- |
| 40 m | 0.1764 | 0.970 | 0.345 | 1.035 | 0.906 | 0.441 |
| 80 m | 0.3044 | 0.955 | 0.663 | 1.048 | 0.909 | 0.439 |
| 160 m | 0.5468 | 0.953 | 1.176 | 1.067 | 0.912 | 0.439 |
| 250 m | 0.7405 | 0.942 | 1.510 | 1.069 | 0.904 | 0.459 |
| 800 m | 0.9832 | 0.767 | 1.251 | 0.947 | 0.796 | 0.624 |

The response is the **full receiver-waveform difference** between a
displaced source and training source 01, measured separately for FD and
the PINN; it is a finite displacement, not an infinitesimal gradient.
`Change cosine` measures the orientation of these two differences, whereas
`change relative L2` divides their mismatch norm by the FD-change norm.
Neither quantity uses a fitted shift or gain.

The FD waveform itself changes by 34.5% of the original waveform norm after
only a 40 m source displacement. At 40--250 m, the PINN follows much of the
change direction and amplitude, but the *change error* stays near 44--46%.
Consequently the absolute waveform error rises rapidly even when the peak
ratio remains close to one. The 800 m response loses more directional
agreement and is near the zero-prediction raw L2 baseline of 1.0. This is
more specific than an across-event amplitude-bias explanation. Because
this sweep covers only one direction, one source, and one frequency, it
cannot establish a general operating radius or validated inversion gradient.

The earlier fixed-location source-time FD controls (0.16--0.28 s) remain
near raw L2 = 0.10. The contrast isolates spatial source propagation as a
development bottleneck. Existing trials of source-conditioned eikonal
carriers and straight-path velocity features have not passed the waveform
gate; this result motivates testing a propagation operator that resolves
source-dependent heterogeneous waveforms, with spatial-response error as
an explicit development metric. No localization or UQ result follows from
these diagnostics.

To reproduce the new FD records in PowerShell from the project root:

```powershell
foreach ($x in @(3540, 3580, 3660)) {
  $dx = '{0:0000}' -f ($x - 3500)
  python examples/generate_fd_observation.py --config data/observations/experiment_draft.json --source $x 4000 500 0.12 --duration 2.0 --output "data/observations/development_spatial_radius/train01_dx${dx}_0to2s.npz"
}
$new = Get-ChildItem data/observations/development_spatial_radius/train01_dx0*_0to2s.npz | Where-Object Name -NotLike '*recomputed*' | Sort-Object Name | ForEach-Object FullName
$events = $new + @('data/observations/window_0to2s/dev_distance_0250_0to2s.npz', 'data/observations/window_0to2s/dev_distance_0800_0to2s.npz')
python examples/audit_spatial_source_response.py outputs/receiver_weight_0.03_hybrid_1000/conditioned_5Hz.pth data/observations/window_0to2s/train_01_0to2s.npz $events --output experiments/receiver_weight_0.03_spatial_response_development.json
```

Detailed diagnostics are stored in
`experiments/receiver_weight_0.03_spatial_radius_development.json` and
`experiments/receiver_weight_0.03_spatial_response_development.json`.
