# Local FD depth interpolation baseline

This development diagnostic asks whether the difficult nearby depth waveforms are locally representable from available FD records. For each of anchors a10 and a14, the records at depth offsets -160, 0 and +160 m are treated as the only inputs. The -80 m target is predicted by `0.5 * (-160 m) + 0.5 * (0 m)` and the +80 m target by `0.5 * (0 m) + 0.5 * (+160 m)`. The four ±80 m target records were not used in this interpolation. All records share the same receivers, time axis and FD configuration.

| Target event | Linear interpolation full L2 | Nearest training record full L2 | Linear active L2 | Linear response L2 |
|:--|--:|--:|--:|--:|
| a10 -80 m | 0.133 | 0.483 | 0.124 | 0.343 |
| a10 +80 m | 0.131 | 0.317 | 0.131 | 0.344 |
| a14 -80 m | 0.200 | 0.417 | 0.196 | 0.481 |
| a14 +80 m | 0.147 | 0.591 | 0.140 | 0.392 |
| **Mean** | **0.153** | **0.452** | **0.148** | **0.390** |

The linear FD interpolation has mean full-record L2 0.153, mean active-sample L2 0.148, mean response L2 0.390 and mean zero-lag correlation 0.993. The nearest-record baseline has full L2 0.452, active L2 0.413 and response L2 1.092. This establishes that the local ±80 m waveform response is smooth enough to be approximated from neighboring source records; the FD data themselves do not make this local task impossible.

For context, the archived 29-source neural model has mean full L2 0.392 and active L2 0.333 on the same four ±80 m events; the 33-source depth-coverage neural model has 0.224 and 0.180. The interpolation baseline is lower still, but it has access to two neighboring FD waveforms bracketing each target and is therefore a diagnostic interpolation oracle rather than a deployable forward operator. It does not provide source-coordinate gradients, solve the wave equation, or perform localization.

The result points to the next modeling requirement: the source-conditioned network must learn a smooth, physically constrained mapping of source position to waveform from sparse neighboring supervision. Adding a FMM time hint alone did not achieve this in the paired-head experiments. A future method should be compared against this local interpolation baseline on a predeclared source neighborhood, then evaluated with a truth-independent four-parameter search on untouched events. These four targets and the six bracketing records are already development data and cannot be promoted to independent confirmation.

The fixed protocol, input hashes, and event-level numbers are in `depth_linear_interpolation_protocol.json` and `depth_linear_interpolation_results.json`. Run `python examples/audit_depth_linear_interpolation.py`; the script refuses to overwrite existing results.
