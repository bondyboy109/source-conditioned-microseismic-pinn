# FD-midpoint pseudo-label fine-tuning

This development-only experiment starts from the registered 33-source, 5 Hz checkpoint and fine-tunes two identical model copies for 500 updates. The `control` model receives ordinary supervision from the 33 FD records. The `candidate` receives the same batches plus four midpoint pseudo-records formed by averaging raw FD endpoint waveforms at the same `(x,y,ts)` and adjacent depths.

The pseudo-record depths are 920 and 1080 m at anchor a10, and 1120 and 1280 m at anchor a14. Their actual FD records are used only for the post-training development comparison; the pseudo-labels themselves are computed from endpoint records. All endpoint and evaluation data had already been inspected in earlier development, so this is not independent validation.

| Depth group | Role | Control full L2 | Candidate full L2 | Control active L2 | Candidate active L2 |
|:--|:--|--:|--:|--:|--:|
| ±80 m | pseudo-label midpoint locations | 0.202 | 0.180 | 0.174 | 0.157 |
| ±160 m | true FD endpoint training records | 0.077 | 0.131 | 0.028 | 0.094 |
| ±280 m | outside the local midpoint brackets | 0.646 | 0.642 | 0.563 | 0.554 |
| ±80 and ±280 m | eight non-endpoint development records | 0.424 | 0.411 | 0.369 | 0.355 |
| All 12 | includes four endpoint training records | 0.308 | 0.318 | 0.255 | 0.268 |

The candidate improves the four midpoint locations by about 10.8% in full-record L2. Across the eight non-endpoint development records, full L2 improves by 3.0% and active L2 by 3.6%. The pseudo-label loss falls from 0.1089 on the first update to 0.0161 on the last update, confirming that the network learns the supplied midpoint targets.

The same candidate degrades the four endpoint records: mean full L2 increases from 0.077 to 0.131 and active L2 from 0.028 to 0.094. Its final ordinary receiver loss is also higher than the control value (0.03524 versus 0.01083). Equal weighting of receiver and teacher losses therefore trades endpoint fidelity for smoother midpoint behavior.

This is a modest positive development signal, but it is not yet a usable forward model. It does not reach the direct local interpolation result at ±80 m (full L2 0.153), and the ±280 m error remains high. The next controlled experiment should reduce the teacher weight while retaining full endpoint replay, selecting the smallest weight that improves midpoint records without materially degrading endpoint FD fits. No new localization test should be opened until the forward gate is met.

Inputs, hashes and detailed per-event metrics are in `depth_midpoint_teacher_finetune_protocol.json` and `depth_midpoint_teacher_finetune_results.json`. Saved development checkpoints are under `outputs/depth_midpoint_teacher_finetune_500/`.
