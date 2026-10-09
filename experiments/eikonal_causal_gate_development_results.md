# Approximate onset-gate audit (development only)

On the twelve already-inspected 5 Hz depth-displacement FD records, apply the fixed gate registered in `eikonal_causal_gate_development_protocol.json` to predictions of both frozen 29-source models. This is postprocessing of receiver waveforms at the **true** source; it does not retrain the PINN or perform source inversion. The gate uses source-to-receiver Euclidean distance, the maximum velocity in the registered volume (6166.213 m/s), a 200 m source-support allowance, 0.20 s Ricker lead and 0.10 s smooth rise. Despite the experiment name, the gate does **not** calculate an eikonal travel time. Its onset is only an approximate, early bound: both the Ricker pulse and Gaussian source have nonzero tails.

| Measure (mean across 12 events unless count) | Eikonal model before / after | Straight model before / after |
|:--|--:|--:|
| Full-record receiver relative L2 | 0.647 / 0.640 | 0.678 / 0.672 |
| FD-active-sample relative L2 | 0.582 / 0.582 | 0.617 / 0.617 |
| Finite depth-response relative L2 | 0.802 / 0.794 | 0.855 / 0.851 |
| Finite depth-response cosine | 0.609 / 0.614 | 0.598 / 0.601 |
| Picks before origin minus 0.10 s, among 142 usable FD station/event pairs | 65 / 0 | 71 / 0 |
| Picks at first record sample | 39 / 0 | 41 / 0 |
| Paired picks | 142 / 142 | 142 / 142 |
| Mean per-event threshold-pick RMSE (s) | 1.151 / 0.493 | 1.153 / 0.487 |

The squared FD waveform difference introduced by applying the same gate to FD, divided by FD waveform energy, averages 1.14e-23 and is at most 1.30e-22 across these records. This establishes that the chosen gate removes essentially none of the *sampled* FD signal in this development set; it does not establish a strict causal bound for the continuous source. The full-record L2 decrease is small and comes from suppressing predicted signal outside the FD-active samples. The 142 paired picks remain available, but approximately half a second of pick RMSE remains; a threshold pick after gating must not be reported as validated propagation time. At 280 m depth offsets the underlying eikonal model still has mean full-record L2 around 0.92 after gating, so the depth-response problem remains.

This result isolates a correctable pre-arrival output artifact, not an improvement in independent localization. All twelve records were already used in development. The gate depends explicitly on the candidate source, but applying it at the known true source here does not demonstrate that an inverse search prefers that source. To support a localization or geoscientific claim, a separately specified differentiable forward operator and inversion protocol would need new, untouched source locations, with accuracy and runtime measured against FD/FMM baselines. Do not replace the reported held-out results or claim a new independent test from this audit.

Exact parameters and per-event results: `eikonal_causal_gate_development_protocol.json`, `eikonal_causal_gate_development_results.json`; executable audit: `examples/audit_eikonal_causal_gate.py`. All inputs are checked against recorded hashes; no checkpoint is modified.
