# Development-only arrival-picker bias: unseen-source FD evaluation

The historical development set contains 16 FD events. For each event, we
computed the median receiver offset between the six-sigma MAD pick and the
origin time plus reciprocal FMM travel time at its known source. The median
across development events was **-0.262792 s**, with an event-median MAD of
**0.008281 s**. In leave-one-event-out development validation, mean origin-time
error fell from 0.21485 to 0.08103 s; mean location error rose from 1017 to
1070 m, with no new failures. This met the predetermined development criteria.

Before generating more FD data, we registered eight distinct new source
locations, the correction constant, the exact selection procedure and the
acceptance thresholds in `arrival_bias_new_source_protocol.json`. The FD
observation hashes are in `arrival_bias_new_source_manifest.json`. We ran the
unmodified reciprocal 160 m FMM search on the same 25 valid MAD picks in both
branches, subtracting the development bias only in the corrected branch.
Truth was read after both searches for scoring. No model weights or search
settings were tuned on these eight events.

| Event | Raw location (m) | Corrected location (m) | Raw origin error (s) | Corrected origin error (s) |
|---|---:|---:|---:|---:|
| bias_01 | 856 | 856 | 0.190 | 0.018 |
| bias_02 | 1735 | 1735 | 0.330 | 0.118 |
| bias_03 | 717 | 789 | 0.200 | 0.013 |
| bias_04 | 1161 | 1329 | 0.088 | 0.073 |
| bias_05 | 1177 | 1177 | 0.370 | 0.111 |
| bias_06 | 1183 | 975 | 0.180 | 0.057 |
| bias_07 | 1128 | 1128 | 0.285 | 0.023 |
| bias_08 | 1151 | 1151 | 0.220 | 0.031 |
| **Mean** | **1138** | **1142** | **0.23293** | **0.05540** |

There were zero failures in both branches. The mean origin-time error
decreased **76.2%**, while the mean location error increased **0.37%**.
This passes the frozen acceptance rule (at least 25% time-error reduction,
at most 10% location-error increase, and no additional failures). Five raw
origin-time estimates hit the 0.05 s lower search bound, while none of their
corrected estimates did. The improved time estimates partly reflect removal
of that clipping. A fixed pickup offset did not solve spatial localization:
the corrected mean error remains **1.14 km**, and three of the eight selected
spatial candidates changed.

The offset is an empirical difference between the MAD picking rule and a
travel-time convention for this synthetic 5 Hz wavelet and FMM grid. It is
not a physical propagation delay, nor evidence that the corrected pick is
an unbiased first break. Both cohorts reuse the same Overthrust velocity,
FD solver and surface receiver geometry. These results do not validate new
geology, alternate wavelets, noise robustness or field data. This experiment
evaluates **FMM arrival-time localization only**: it supplies no evidence
that the learned mixture-of-experts waveform branch beats FMM.

Reproduction on this machine:

```powershell
& 'E:\soft\Anaconda3\envs\torch_env\python.exe' examples\validate_arrival_bias_new_sources.py evaluate
```

The result file is deliberately write-once, so the command above will refuse
to replace `arrival_bias_new_source_results.json`. The `.npz` waveforms are
ignored by Git; archive them with the manifest if sharing the result.
