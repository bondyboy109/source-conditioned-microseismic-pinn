# Neighboring source waveform response audit

31 matched-time FD training-source pairs among 33 events; 251 samples per receiver trace. Both predictions and FD records are evaluated at identical receiver/time coordinates.

| Training path | Median predicted/FD difference amplitude | Median relative L2 of difference | Median difference cosine |
|---|---:|---:|---:|
| source_to_target | 0.624 | 0.852 | 0.535 |
| target_to_target | 0.621 | 0.802 | 0.600 |

Each median is over matched pairs and three frozen seeds. A relative L2 below 1 improves on a zero predicted difference; a cosine near zero means the predicted change is poorly aligned with FD. Per-pair and per-seed measurements are in the JSON file.

All pairs are target-domain training sources. This is an in-sample source-response diagnostic, not an independent localization result; no test/development event is used. Normalization divides each FD record by its own RMS, matching training.
