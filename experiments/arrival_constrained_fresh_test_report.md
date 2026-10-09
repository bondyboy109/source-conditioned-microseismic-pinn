# Locked arrival-constrained localization: fresh FD sources

The eight source locations were registered in
`arrival_constrained_fresh_test_protocol.json` before their 5 Hz FD waveforms
were generated. None duplicated an existing source record. The fixed 25-receiver
geometry and Overthrust velocity volume were reused. This is an unseen-source
test, not an unseen-velocity-model or field-data test.

On all 16 historical development events, the predetermined decision rule
selected the six-sigma MAD picker and `alpha = 1.0`: the threshold picker
failed on one event under the common six-valid-pick requirement, while MAD
failed on none. The mean development location errors for MAD were 1017 m
(arrival only), 2372 m (waveform only) and 1047 m (hybrid at alpha 1.0).
Thus the selected hybrid did **not** outperform its own FMM baseline on
development. This choice and its SHA-256 were saved before FD generation.

## Fresh-source results

The same reciprocal 160 m FMM receiver fields, 6 x 6 x 4 coarse grid,
FMM-centered local candidates, candidate origin-time estimate, and valid picks
were used for all three objectives. Each waveform score came from the frozen
prototype-gated source-conditioned model.

| Event | Valid receivers | FMM error (m) | Hybrid error (m) | Waveform error (m) | FMM/hybrid origin-time error (s) |
|---|---:|---:|---:|---:|---:|
| fresh_01 | 25 | 1336 | 1336 | 1284 | 0.109 |
| fresh_02 | 25 | 1008 | 1008 | 1344 | 0.227 |
| fresh_03 | 25 | 1425 | 1425 | 4811 | 0.212 |
| fresh_04 | 25 | 364 | 364 | 3278 | 0.277 |
| fresh_05 | 25 | 686 | 686 | 686 | 0.160 |
| fresh_06 | 25 | 954 | 954 | 2720 | 0.299 |
| fresh_07 | 25 | 1161 | 1161 | 4533 | 0.140 |
| fresh_08 | 25 | 1318 | 1318 | 2258 | 0.409 |
| **Mean** | **25** | **1031** | **1031** | **2614** | **0.229** |

The FMM and hybrid objectives selected **exactly the same source candidate
on all eight events**. The hybrid contributed no measurable gain over the
same-picker FMM baseline in this locked test. Mean location error for FMM and
hybrid was 1031 m (sample standard deviation 362 m), and the median was
1084 m. There were zero failures, but the mean origin-time error was 0.229 s.

The MAD picks were earlier than the 5%-of-peak picks by a median of
0.166-0.274 s per event (mean of those event medians: 0.223 s earlier).
All eight recovered origin times were earlier than the true origin time.
This is a picker-dependent timing bias diagnostic, not proof that either
picker represents the physical first break exactly. Parameters must be
revised on development data and then tested on another untouched source set.

The FMM receiver-field setup took 3.84 s and mean post-setup search/scoring
time was 0.066 s per event on the local machine; these timings exclude FD
observation generation and model training. The new FD solves took about
38-41 s per 9 s record on the RTX 4060 Laptop GPU.

## Interpretation

The earlier 6.9% hybrid improvement on a repeatedly inspected holdout did
not replicate under this frozen picker/weight protocol. That earlier result
should not be presented as an independent final benefit. The present test
supports the FMM baseline's relative spatial stability, but does not establish
that the learned waveform model improves localization or origin-time recovery.
Moreover, the weight candidates had been viewed on an earlier holdout before
this protocol was written; this is a prospective synthetic confirmation,
not a claim of fully preregistered external validation.

Inputs and machine-readable outputs:

- `arrival_constrained_fresh_test_protocol.json` and
  `arrival_constrained_picker_development.json`
- `arrival_constrained_fresh_test_manifest.json` (FD record hashes)
- `arrival_constrained_fresh_test_results.json` (all estimates, components,
  failures, and run times)
- `examples/validate_arrival_constrained.py` and
  `examples/generate_fresh_arrival_test.py`

The generated `.npz` FD observations are excluded by `.gitignore`. They must
be archived with their manifest to reproduce this test outside this machine.
