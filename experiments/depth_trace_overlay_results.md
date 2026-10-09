# Frozen 860-to-940 m waveform overlay

The four stations were fixed by `depth_inverse_diagnostic.json` before
plotting. The left column shows FD and the frozen 24-event model at the
860 m **training** source; the middle shows the 940 m **development**
source; the right shows each prediction's change between the two depths.
No registration, lag correction, or amplitude fitting is applied in the
figure or primary error. View `depth_trace_overlay.png` for all four.

| Station (x,y,z), m | Raw trace L2 at 860 m | Raw trace L2 at 940 m | 80 m response L2 | Response cosine |
| --- | ---: | ---: | ---: | ---: |
| (5000,2000,100) | 0.096 | 0.512 | 1.123 | 0.269 |
| (8000,2000,100) | 0.086 | 0.417 | 1.132 | 0.241 |
| (5000,5000,100) | 0.108 | 0.595 | 1.141 | 0.319 |
| (8000,5000,100) | 0.058 | 0.349 | 0.847 | 0.566 |

The 860 m training traces closely track FD. At 940 m, strong initial
arrivals are still visible in both predictions, while later oscillations
and their **depth-dependent changes** disagree. This is why a small raw
waveform error can coexist with a nearly incorrect depth response. A
single global travel-time or amplitude adjustment is not used as a fix;
fitted per-station lags/gains are diagnostic only in
`depth_trace_overlay_metrics.json`.

This observation motivates the pre-registered 900 m training-source pilot
(`depth_gap_900_protocol.json`). Both the source placement and the reuse of
940 m data were selected after development inspection: this is not a new
independent confirmation. The new 820 and 980 m FD events are also
development-only and must not be represented as held-out blind tests.

Reproduce with the established `torch_env` and base Anaconda environments:

```powershell
& 'E:\soft\Anaconda3\envs\torch_env\python.exe' examples\export_depth_trace_overlay.py
& 'E:\soft\Anaconda3\python.exe' examples\plot_depth_trace_overlay.py outputs\depth_trace_overlay_data.npz --output experiments\depth_trace_overlay.png
```
