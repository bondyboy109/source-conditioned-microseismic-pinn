# Boundary-condition sensitivity audit

This development audit compares the existing zero-Dirichlet FD records with
records generated using a lateral/bottom sponge while retaining the same
velocity model, receiver geometry, source parameters, and time sampling. The
sponge is a sensitivity test; it is not a perfectly matched layer and it is
not claimed as a PINN boundary implementation.

## Recorded results

| Event | 0--2 s relative L2 | 2--4 s relative L2 | 4--9 s relative L2 |
|---|---:|---:|---:|
| `train_01` | 0.344 | 0.883 | 0.997 |
| `dev_distance_0250` | 0.344 | 0.890 | 0.997 |

For the 250 m source displacement template test, the relative L2 error was
`1.343` for the zero-boundary pair and `0.805` for the lateral/bottom-sponge
pair when the source change was represented by a constant-speed time shift.
Using integrated-slowness ray travel times gave `1.546` and `1.056`,
respectively.

## Interpretation

The early 0--2 s window is less sensitive to the boundary choice than the late
record. The strong divergence after 2 s means that full-record comparisons
between PINN and FD can be contaminated by boundary treatment even when the
source and receiver settings match. The result supports restricting the
waveform comparison to an early active window or matching the boundary
condition in both solvers. It does not establish that the sponge is physically
superior to the zero-Dirichlet condition.

The machine-readable values are stored in
`experiments/boundary_sensitivity_development_results.json`.
