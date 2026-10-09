# Relative-source carrier diagnostic (development only)

Both runs used the same eight 5 Hz FD training events (`train_01` through
`train_08`), seed 42, 5000 optimizer steps, receiver batch size 1024,
receiver weight 0.1, output scale `1e-5`, and harmonic carrier with the
initial-time gate. Neither run used PDE, initial-condition, or boundary losses.
The second run replaces absolute coordinates with relative source coordinates
and a straight-ray arrival-time hint using a fixed 3500 m/s reference speed.
That hint is an architectural feature, not an accurate Overthrust traveltime.

| Event | Absolute carrier relative L2 | Relative carrier relative L2 |
| --- | ---: | ---: |
| train_01 | 0.679 | 0.707 |
| train_02 | 0.656 | 0.612 |
| train_03 | 0.680 | 0.745 |
| train_04 | 0.727 | 0.785 |
| train_05 | 0.722 | 0.750 |
| train_06 | 0.662 | 0.659 |
| train_07 | 0.838 | 0.782 |
| train_08 | 0.724 | 0.696 |
| heldout_01 (development) | 1.063 | 1.031 |

The zero-wavefield baseline has relative L2 = 1.0. Both models fit training
events better than zero, but neither predicts the development source better
than zero. The development event has already been used to choose and assess
architectures; it is not an independent test. Previous `blind_01` through
`blind_03` events were not used in this comparison, but have already been
evaluated elsewhere and cannot serve as untouched tests for future revisions.
These supervised checkpoints are not PINN forward models or localization
results. Do not use them as evidence of generalization or computational gain.

Records and checkpoints: `outputs/dev_receiver_only_5hz_train8_5000/` and
`outputs/dev_relative_carrier_5hz_train8_5000/`.

## Leave-one-source-out check

For each of the original eight registered training events, a separate
relative-carrier model was trained on the other seven events for 5000 steps
and evaluated only on the omitted event. All folds used the same settings as
above. Each audit verified that the omitted source and observation file were
absent from the checkpoint training records. This is a development diagnostic,
not eight independent blind tests.

| Omitted event | Nearest training source (m) | Relative L2 | Predicted / FD peak |
| --- | ---: | ---: | ---: |
| train_01 | 4042 | 1.112 | 0.235 |
| train_02 | 3662 | 1.095 | 0.112 |
| train_03 | 4142 | 1.185 | 0.240 |
| train_04 | 3121 | 1.091 | 0.447 |
| train_05 | 3662 | 1.352 | 0.276 |
| train_06 | 4042 | 1.134 | 0.146 |
| train_07 | 3121 | 1.211 | 0.389 |
| train_08 | 4128 | 1.062 | 0.117 |

None of the eight folds beats the zero-prediction baseline of 1.0; median
relative L2 is 1.123. The predicted peak is only 11%-45% of the FD peak.
Nearest-source distances cover only 3.1-4.1 km, so this sample cannot
establish whether distance causes the errors. More training epochs alone
are not supported by this diagnostic. The primary unresolved issue is
waveform amplitude and source-to-source generalization, even without PDE loss.
Fold logs, checkpoints, audits, and machine-readable results are in
`outputs/dev_loo_relative_carrier_5hz/summary.json`.

## What causes the high held-out error?

The eight omitted-event audits show that an optimal **global** amplitude
correction leaves relative L2 at approximately 1.0. A separate per-receiver
alignment diagnostic uses each FD trace to select its best lag within
plus/minus 0.5 s and an optimal signed gain. Before shifting, the median
absolute waveform cosine is 0.04-0.10 across folds; after shifting, it is
only 0.18-0.24. No receiver in any fold reaches 0.5 after shifting. These
optimistic post-hoc diagnostics indicate a waveform-shape mismatch as well
as amplitude and timing errors. Detailed per-trace results are in
`outputs/dev_loo_relative_carrier_5hz/waveform_alignment.json`.

To separate sparse multi-kilometre coverage from local source sensitivity,
three additional **development** sources were registered in
`experiments/source_distance_development.json` before their FD records were
generated. They share the depth and origin time of `train_01`, and move only
along x. The same frozen eight-source checkpoint was used for every audit.

| Source | Distance from train_01 | Original relative L2 | Best global gain relative L2 |
| --- | ---: | ---: | ---: |
| train_01 (in training) | 0 m | 0.707 | 0.698 |
| dev_distance_0250 | 250 m | 1.275 | 0.987 |
| dev_distance_0800 | 800 m | 1.325 | 0.998 |
| dev_distance_2000 | 2000 m | 1.215 | 1.000 |

None of the new records entered training, and all three exceed the zero
baseline of 1.0 despite one being only 250 m away. Thus, current evidence
does not support treating additional FD training events alone as a confirmed
fix. The 5 Hz wavelength spans approximately 460-1230 m across the velocity
range; a 250 m move can materially change waveform phase. The next proposed
method change should represent source-dependent propagation and be tested on
development data before any new untouched evaluation set is created.

## Receiver-template propagation control

A fixed, physically motivated control used `train_01`'s FD receiver traces
as templates for the three additional development sources. For each receiver,
the script shifted its template by the **difference** in source-to-receiver
time and multiplied it by the old/new propagation-distance ratio. Two
prespecified times were tested: distance divided by 3500 m/s, and straight
segment slowness integrated through the original 40 m velocity grid with
64 midpoint samples. No development waveform was used to fit the times,
amplitudes, or model parameters.

| Development source distance | Existing network L2 | Fixed-speed template L2 | Integrated-slowness template L2 |
| --- | ---: | ---: | ---: |
| 250 m | 1.275 | 1.343 | 1.546 |
| 800 m | 1.325 | 1.842 | 1.911 |
| 2000 m | 1.215 | 1.919 | 1.927 |

At 250 m, the fixed-speed template scores 0.766 in the first 2 s, which
contain only 14.6% of the FD trace energy. By 4 s, error rises to 1.212;
the full 9 s error is 1.343. The two farther sources also fail in the
first 2 s (errors 1.276 and 1.635). The source shift controls therefore
do **not** solve full-waveform generalization; the integrated straight ray
is not an eikonal or multipath solution and performs worse in this check.
Zero-Dirichlet boundaries can return late waves into the 9 s records, but
the present diagnostic cannot attribute their error specifically to those
reflections. Numerical records are in `outputs/dev_ray_template_5hz.json`.

## Nine-second FD boundary sensitivity

Two pre-registered sources (`train_01` and `dev_distance_0250`) were
re-simulated at 5 Hz with the same 40 m grid, source, stations, and
0.002 s time step, but a damped-wave-equation sponge on the four lateral
sides and bottom. The top remains zero-Dirichlet; the 800 m quadratic
sponge reaches 60/s at each outer face. This is an experimental absorbing
layer, not a validated PML. The original six-zero-face FD data and all
previous checkpoints were left intact.

| Window | Original energy share: train_01 / 250 m | Sponge/original energy: train_01 / 250 m | Sponge-versus-original L2 |
| --- | ---: | ---: | ---: |
| 0-1 s | 5.4% / 7.0% | 100% / 100% | 0.000 / 0.000 |
| 1-2 s | 5.6% / 7.6% | 83% / 87% | 0.482 / 0.478 |
| 2-4 s | 23.0% / 21.2% | 29.5% / 26.2% | 0.883 / 0.890 |
| 4-9 s | 66.1% / 64.3% | 1.1% / 1.2% | 0.997 / 0.997 |

The two FD variants agree through 1 s, then diverge strongly. The
original zero-boundary FD records contain most of their energy after 4 s;
the lateral/bottom sponge removes nearly all energy in that window. This
strong sensitivity is consistent with artificial boundary returns, but
the experiment does not separately identify every reflected ray or
validate sponge waveforms as ground truth. With both template and target
simulated under the sponge boundary, the 250 m fixed-speed shifted-template
L2 falls from 1.343 to 0.805 across 9 s; the integrated straight-ray
version remains at 1.056. These are receiver-template development controls,
not learned PINN or localization results. Adopting sponge data for model
training would require a matching PINN boundary treatment and new untouched
test events. Configuration and detailed measurements are in
`experiments/boundary_sensitivity_development.json` and
`outputs/dev_fd_boundary_sensitivity.json`.
