# Uniform-medium calibration result

This calibration uses a separate constant velocity model with `v=3500 m/s`,
the same 40 m grid, 25 receivers, 0.002 s time step, and 5 Hz source as the
Overthrust experiments. Both events use the lateral/bottom sponge diagnostic
(800 m width, maximum damping 60/s) and are evaluated over 0-2 s. The train
source is `(3500, 4000, 500, 0.12)` m/s and the test source is `(3750, 4000,
500, 0.12)`; the test source is 250 m away and never entered training.

The model is a translation-invariant relative-source travel-time carrier. It
uses only relative source-observer coordinates, delayed time, and reference
speed `3500 m/s`; it does not receive absolute source coordinates as learned
features. Training uses receiver supervision only, seed 42, 5000 steps,
batch size 1024, and output scale `1e-5`. This is a representation calibration,
not a PINN accuracy experiment: the PDE loss was not optimized.

| Event | Original-amplitude relative L2 | Best global-gain diagnostic L2 | FD peak (magnitude) | Model peak (magnitude) |
| --- | ---: | ---: | ---: | ---: |
| Training source | 0.435 | 0.403 | 2.655e-6 | 2.642e-6 |
| 250 m test source | 0.461 | 0.443 | 3.302e-6 | 2.651e-6 |

Both events beat the zero-wavefield baseline of 1.0, and the held-out error
is close to the training error. The receiver overlay and residual plot is
`outputs/dev_uniform_3500_translation_invariant_5hz_0to2_version2_5000/traces.png`.
The full records and provenance are under
`data/observations/uniform_3500/`.

The source-centre PDE diagnostic remains poor: at the registered training
source, the model gives `PDE lhs=0.0281`, source forcing `1.0`, and residual
`-0.9719`; at 100 m offset the residual is `-0.1356`. The finite smooth core
in the distance feature removes the previous NaN at exact source coincidence,
but it does not make receiver-only fitting satisfy the wave equation. The
result therefore supports a limited conclusion: a translation-invariant
representation can interpolate a nearby source in a simple medium, while
the current supervised model is still not a validated physics-informed
forward operator.

The version-2 checkpoint rejects older translation-invariant checkpoints that
lack the smooth source-distance correction. Fourteen automated tests pass.
