# Frozen MoE scoring versus the refined FMM baseline

The previously accepted FMM refinement reduces mean spatial error on eight
new 5 Hz synthetic FD events to **176.6 m**. These events have already been
inspected for the FMM result. The comparison here is therefore a **fixed-rule
retrospective diagnostic**, not an independent validation of any MoE benefit.
No network was retrained and no weighting parameter was selected here.

The script reconstructed the exact same refined FMM optimum on each event
before scoring the union of all visited candidates with the frozen
source-conditioned prototype-gated MoE. All three branches saw the same
candidate locations, FMM-derived candidate origin times, corrected MAD
picks and observed waveforms. We retained the previously selected hybrid
weight `alpha=1.0` and `arrival_scale_s=0.1`. The saved checkpoint and all
input hashes are fixed in `refined_fmm_moe_diagnostic_protocol.json`.

| Event | Refined FMM (m) | MoE waveform only (m) | Fixed hybrid (m) |
|---|---:|---:|---:|
| spatial_01 | 225 | 6053 | 300 |
| spatial_02 | 247 | 8420 | 228 |
| spatial_03 | 113 | 200 | 200 |
| spatial_04 | 185 | 463 | 154 |
| spatial_05 | 160 | 6852 | 418 |
| spatial_06 | 118 | 11625 | 339 |
| spatial_07 | 138 | 1058 | 107 |
| spatial_08 | 226 | 1146 | 226 |
| **Mean** | **177** | **4477** | **247** |

Mean origin-time errors were **0.0142 s**, **0.1092 s** and **0.0184 s**
respectively. The hybrid beat FMM spatially on only **3/8** events; its
mean spatial error was **39.7% higher** than FMM. Pure MoE favored distant
sources with smaller waveform MSE despite arrival-time residuals as large
as 1.31 s. Once the same FMM arrival score is enforced, the hybrid stays
closer to the correct region but still provides no mean localization gain.

A post-selection diagnostic evaluated the frozen MoE at each **true source**.
Using exactly the scorer's receiver/time subsampling and full-record RMS
normalization, a zero-waveform predictor scored a mean MSE of **0.993**.
The MoE at the true sources scored **1.169**, worse on **8/8** events.
The MoE scored worse than zero also at every refined FMM source. The
true-source calculation was performed after candidate ranking and used
only to diagnose the forward waveform fit. It indicates that poor
forward-model agreement, rather than just a coarse search, limits the
waveform branch under this metric. It does not establish whether the cause
is wavelet scaling, time alignment, training coverage, boundary treatment,
or model capacity.

Shared FMM field setup took **4.06 s**. Constructing and ranking the
refinement candidate pool averaged **0.033 s per event**; evaluating MoE
waveforms for the same pool averaged **0.473 s per event** on this machine.
These timings omit FD data generation and model training. The candidate
pool contained 273-351 unique positions per event.

This result removes any basis for claiming that the current frozen MoE or
its fixed hybrid improves location accuracy over a properly refined FMM
baseline. Before further localization experiments, validate the model's
waveform prediction at known source coordinates against FD traces,
including source normalization, amplitude and time alignment. An improved
model would then require a newly registered source set for confirmation.
These eight synthetic events share one velocity volume, one array, one
source wavelet and one simulator; they say nothing about field performance.

Full per-event candidate estimates, losses, error components and timings:
`refined_fmm_moe_diagnostic_results.json`. The true-source zero baseline:
`refined_moe_truth_waveform_audit.json`. Both scripts refuse to overwrite
their result files, and FD `.npz` observations must accompany the manifest
when reproducing this result outside the local workspace.
