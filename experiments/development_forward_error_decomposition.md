# Frozen PINN waveform-error decomposition on development events

We evaluated the already selected 5-Hz receiver-only checkpoint
`outputs/overthrust_core20_train16_jointbatch_receiver_5hz_5000/conditioned_5Hz.pth`
on the registered 250, 800, and 2000 m development events. None of the three
observation files or true source coordinates appears in the checkpoint's
training record. No weights, hyperparameters, or event definitions were
changed. This is a residual diagnosis, not an independent test or a new
performance claim.

| Event offset from train_01 | Raw relative L2 | Peak-amplitude ratio | Median receiver-energy ratio | Best common shift + gain L2 (diagnostic only) | Best shift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 m | 0.7598 | 0.946 | 1.020 | 0.6686 | -8 ms |
| 800 m | 1.0012 | 0.779 | 0.882 | 0.8585 | -14 ms |
| 2000 m | 0.9336 | 0.791 | 0.754 | 0.8905 | -4 ms |

The common shift is small and reduces the error by only 0.04--0.14, so a
single origin-time or phase offset does not explain the generalization gap.
The test-fitted gain and shift use each evaluated trace and are strictly
diagnostic; they cannot be reported as model performance. Predicted peak
amplitudes are 78--95% of observed peaks, while median per-receiver energy
ratios fall from approximately 1.02 at 250 m to 0.75 at 2000 m. Thus the
amplitude deficit grows with source displacement, but amplitude correction
alone cannot account for the late-window error.

| Event | Observed energy in 0.5--1.0 s | Error energy in 0.5--1.0 s / full observed energy | Observed energy in 1.0--2.0 s | Error energy in 1.0--2.0 s / full observed energy | Relative L2 in 1.0--2.0 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 m | 47.5% | 10.2% | 51.7% | 47.2% | 0.955 |
| 800 m | 27.3% | 13.4% | 54.2% | 81.6% | 1.228 |
| 2000 m | 26.0% | 16.4% | 48.1% | 53.4% | 1.054 |

Most residual energy occurs in the 1--2 s coda, especially for the 800 m
event. The first 0.25 s contains less than 0.01% of the record energy in all
three files, so a local relative error there is undefined and is omitted.
The FD boundary-sensitivity audit independently shows strong divergence
between zero-boundary and sponge records after 1 s. This is consistent with
late coda and boundary-return complexity contributing to the mismatch, but
does not prove that boundary reflections are its sole cause: the network was
trained on zero-boundary records and the PINN physics losses were disabled in
this receiver-only checkpoint.

For context, the existing 1000-step hybrid physics/IC/BC checkpoint was
evaluated on the same development files with the same diagnostic. Its raw
relative L2 values are `0.7815`, `1.2436`, and `1.2942` for 250, 800, and
2000 m, respectively, versus `0.7598`, `1.0012`, and `0.9336` for the
5000-step receiver-only checkpoint. The hybrid event at 2000 m has relative
L2 values of `1.210` and `1.658` in the 0.25--0.5 s and 0.5--1.0 s windows.
This is not a controlled ablation: the hybrid run has fewer updates, a
different learning rate, and different source sampling settings. It warns
that the current physics-enabled recipe does not resolve the cross-source
residual, but does not show that PDE constraints themselves cause it.

## Interpretation

The present blocker has two measured parts: source displacement reduces
receiver energy and the network fails to reproduce the later coda. A common
time shift or global gain does not remove either issue. The next development
experiment should keep the acquisition, initialization, update budget,
learning-rate schedule, and train/development split fixed, then compare
receiver-only and receiver-plus-PDE/IC/BC objectives while reporting both
early-arrival and coda metrics. Do not
use the three independent events evaluated in the one-time test to choose
among those configurations; they are now consumed for this checkpoint and
must be replaced by fresh preregistered events for a future confirmation.

Raw metrics are in `outputs/dev_forward_error_decomposition.json`. Reproduce
with:

```powershell
python .\examples\audit_forward_error_decomposition.py `
  .\outputs\overthrust_core20_train16_jointbatch_receiver_5hz_5000\conditioned_5Hz.pth `
  .\data\observations\window_0to2s\dev_distance_0250_0to2s.npz `
  .\data\observations\window_0to2s\dev_distance_0800_0to2s.npz `
  .\data\observations\window_0to2s\dev_distance_2000_0to2s.npz `
  --output .\outputs\dev_forward_error_decomposition.json
```
