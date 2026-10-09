# Frozen depth inverse: development diagnosis

Protocol: `depth_inverse_diagnostic_protocol.md`. Machine-readable output:
`depth_inverse_diagnostic.json`. The 24-source 5 Hz PINN checkpoint and FD
records were frozen before this audit. These four sources have already been
inspected in development; none overlaps the checkpoint's receiver training
set. No new independent event, retraining, or full-domain localization test
was performed.

## Source of the remaining depth error

| FD event | Raw waveform L2 | True-depth MSE | Minimum fixed-xyz-time depth (m) | MSE there |
| --- | ---: | ---: | ---: | ---: |
| train_01, z=260 m | 0.3964 | 0.15774 | 280 | 0.15331 |
| train_01, z=740 m | 0.4131 | 0.17064 | 740 | 0.17064 |
| train_09, z=460 m | 0.3746 | 0.14033 | 500 | 0.11501 |
| train_09, z=940 m | 0.4080 | 0.16638 | **900** | **0.12436** |

The fixed-depth scan supplies the actual x, y and origin time. Its minimum
for the 940 m event lies 40 m shallower than truth. This shows bias in the
frozen waveform objective even before a four-parameter search is attempted.
It does **not** measure a general source-localization error. The same
shallower bias appears for the 460 m event (minimum at 500 m); the 740 m
event places its minimum at the true depth within the 20 m scan grid.

The 940 m event's four largest receiver error contributions, normalized by
the full FD record energy, are stations at (5000,2000,100),
(8000,2000,100), (5000,5000,100), and (8000,5000,100) m: 0.042, 0.038,
0.031, and 0.025. They contribute 0.136 out of the full squared relative
error 0.166, or about **82%** of it. Per-station shift-and-gain *diagnostics*
select lags from -0.006 to -0.010 s at those stations. These shifts are
not applied to the primary waveform error or inverse objective; a small
shift alone does not establish a physical arrival-time error.

For the same 940 m event, the `1--1.5 s` window has only 7.5% of observed
energy but contributes 0.057 to squared relative error. The `1.5--2 s`
window has 48.7% of observed energy and contributes 0.082. Thus late
waveform shape contributes strongly, and the error is not confined to the
initial arrival. Compared with the nearest depth *training* source at 860 m,
the FD-to-PINN finite response over just 80 m has relative L2 **1.0305**
and cosine **0.3806**. This is worse than the previously reported 700-to-940
m response (relative L2 0.5104): fitting the 860 m training trace did not
ensure the correct local depth-dependent change. These are finite waveform
differences, not infinitesimal gradients.

## Truth-centred four-parameter diagnostic

On 25 receivers and every fourth time sample, two Adam starts within a
truth-centred box (x/y/z +/-160 m, origin time +/-0.04 s) converge to nearly
identical points for each deeper event:

| Event | Representative estimate (x,y,z in m; t in s) | Location error | Time error | True-source MSE | Final MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| z=740 m | (3505.6, 3985.8, 737.4; 0.1182) | 15.5 m | 0.0018 s | 0.17064 | 0.15862 |
| z=940 m | (6494.6, 3497.3, 917.9; 0.1949) | 22.9 m | 0.0051 s | 0.16638 | 0.10280 |

Neither reported solution hits the local box boundary. The two starts in
each case agree to within about 0.2 m in depth. But the box and starting
locations are defined using the **true source**, so these numbers cannot be
reported as blind location accuracy or as proof that a coarse-to-fine
search would find the same basin. The lower-than-truth 940 m objective
confirms model error can favor a biased depth, independent of optimizer
convergence. The source-time estimate shifts by about -0.005 s as the
optimizer compensates for waveform error; it does not diagnose a unique
physical travel-time bias.

## Decision

The next model-development target is the local depth response near
`train_09` between 860 and 940 m, particularly the first four active
stations and late traces. More optimization iterations or a larger blind
search cannot correct the off-truth depth minimum of the current frozen
forward model. Inspect the FD and PINN trace overlays for those stations,
then change the training coverage/physical carrier only with a newly
registered development protocol. After the method is frozen again, reserve
new, untouched sources for independent confirmation before claiming source
localization or uncertainty coverage.

Reproduce from the project root:

```powershell
& 'E:\soft\Anaconda3\envs\torch_env\python.exe' examples\diagnose_depth_inverse.py
```
