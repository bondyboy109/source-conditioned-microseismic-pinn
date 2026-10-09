# Velocity-conditioned carrier pilot

This pilot tests a new forward operator that augments the frozen eikonal
carrier with eight straight-path velocity samples plus source and receiver
endpoint velocities. The interpolated velocity features are frozen during PDE
automatic differentiation because PyTorch does not provide second derivatives
for 3-D `grid_sample`.

## Configuration

- 16 registered 5 Hz training sources.
- Hybrid PDE/initial/boundary/receiver training.
- 500 epochs, `n_phys=256`, `n_ic=128`, `n_bc_per_face=8`.
- Frozen checkpoint: `outputs/eikonal_overthrust_5000/eikonal.pth`.
- Independent code path: `velocity_conditioned_travel_time_carrier`.

## Development results

| Event | Old eikonal carrier | Velocity-conditioned carrier |
| --- | ---: | ---: |
| 250 m | 0.750 | 0.766 |
| 800 m | 0.960 | 1.206 |
| 2000 m | 0.967 | 1.183 |

The pilot does not improve the heterogeneous Overthrust forward model. The
local straight-path velocity samples are insufficient to represent curved
raypaths, multipath arrivals, and nonlocal amplitude effects. The new network
and checkpoint format are retained for future experiments, but this pilot is
not a basis for localization or uncertainty claims.

All 24 automated tests pass after adding the new architecture.
