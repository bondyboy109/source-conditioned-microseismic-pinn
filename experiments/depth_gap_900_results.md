# One additional depth source: registered development pilot

Protocol: `depth_gap_900_protocol.json`. Before generating any records,
we fixed the 24-source checkpoint (SHA-256 in the protocol), added only
one FD *training* source at `(6500,3500,900,0.2)` m/s, and registered new
**development** sources at depths 820 and 980 m. The 940 m source was
already inspected and directly motivated the choice of 900 m; it is reused
as a development diagnostic, **not** as a new independent test. The 25-source
model was trained from scratch with the same seed, two-stage schedule,
architecture, per-event receiver sampling, and physics sample counts as
the 24-source control. The control checkpoint was not retrained or changed.

## Waveform and source-response comparison

The rows below are all development events at fixed x=6500 m, y=3500 m,
origin time 0.2 s. Response errors and cosines compare each event's FD
and model waveform changes relative to the **860 m training record**;
none of these metrics use an adjusted gain or time shift.

| Development depth | 24-source raw L2 | 25-source raw L2 | 24-source response L2 / cosine | 25-source response L2 / cosine |
| --- | ---: | ---: | ---: | ---: |
| 820 m (new) | 0.1444 | **0.1078** | 0.7018 / 0.7221 | **0.3529 / 0.9401** |
| 940 m (previously inspected) | 0.4080 | **0.1784** | 1.0305 / 0.3806 | **0.4768 / 0.8860** |
| 980 m (new) | 0.6083 | **0.3029** | 1.0714 / 0.3388 | **0.5676 / 0.8272** |

For the prespecified 940 m fixed-true-x/y/origin-time depth scan (20 m
steps), the control has true-depth normalized MSE **0.16638** and favors
900 m (MSE 0.12436). The augmented model has true-depth MSE **0.03195**,
and its lowest scanned misfit is at the true **940 m**. On the new 980 m
development event the augmented scan favors 960 m, a remaining 20 m bias;
the control favors 900 m. The 820 m minimum matches the true depth in
both groups. The scan fixes three *true* source parameters and cannot be
interpreted as four-parameter localization performance. Curves are in
`depth_gap_900_scans.png`.

All preregistered **development-only** criteria pass: three raw errors
below 0.5; 940 m response L2 below 0.5; its depth minimum within 20 m;
both original anchor fit errors below 0.18; and new 820/980 m response
L2 below 0.75 with cosines above 0.8. The original training anchors
`train_01` and `train_09` have raw L2 0.1269 and 0.0985 in the 25-source
model (control: 0.1284 and 0.1026). The new 900 m *training* record has
raw L2 0.1039; this is a fit check, not held-out validation. The result
supports a **local synthetic 5 Hz interpolation improvement** near one
anchor after targeted FD supervision. It does not establish performance
across the velocity model, independent events, other frequencies, real
field records, full four-parameter source inversion or UQ coverage.

## Measured incremental cost

| Component | 24-source control | 25-source model |
| --- | ---: | ---: |
| Receiver pretraining (5000 updates) | 25.43 s | 25.74 s |
| Hybrid training (1000 updates) | 47.72 s | 46.84 s |
| Peak GPU allocation (receiver / hybrid) | 47.3 / 119.9 MiB | 50.4 / 120.7 MiB |
| New FD training source | none | 8.60 s |
| Two new FD development sources | -- | 8.59 + 8.58 s |

Per-run GPU wall-clock differences are small enough that their sign does
not demonstrate a training speedup. One additional training event costs
an FD forward solve; this pilot does not establish a PINN/FD break-even
advantage.

Next, freeze this 25-source checkpoint and register genuinely new,
geographically varied and depth-varied FD events for a one-time independent
confirmation, including a local four-parameter inversion with no truth-
centred box or starting point. Any subsequent training adjustment based
on that confirmation would consume it as development data.

Machine-readable records: `depth_gap_{control24,aug25}_{development,response,training,depth_scans}.json`.
Reproduce from the project root:

```powershell
& 'E:\soft\Anaconda3\envs\torch_env\python.exe' examples\run_depth_gap_900.py
& 'E:\soft\Anaconda3\python.exe' examples\plot_depth_gap_900.py experiments\depth_gap_control24_depth_scans.json experiments\depth_gap_aug25_depth_scans.json --output experiments\depth_gap_900_scans.png
```
