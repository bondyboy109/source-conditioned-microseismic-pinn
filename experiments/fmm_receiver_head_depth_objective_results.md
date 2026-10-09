# Fixed-geometry depth objective: paired receiver heads

This **post-hoc development diagnosis** uses the 12 previously inspected FD depth-offset records around two *training* anchors. For each observation, x, y and origin time are fixed to that anchor's **known true values**. Seven candidate depths are scored at offsets -280, -160, -80, 0, +80, +160 and +280 m from the anchor. Both learned-carrier and FMM-carrier heads, with and without the same previously fixed approximate onset taper, use the same candidates and unshifted 25 x 1,001-sample waveform objective `sum((prediction-FD)^2)/sum(FD^2)`. The actual observation is not used to generate each candidate prediction. No search optimizer or model fitting is performed.

| Frozen receiver head | True depth lowest / 12 | Best depth within 80 m / 12 | Mean absolute selected depth error |
|:--|--:|--:|--:|
| Learned carrier | 7 | 12 | 33.3 m |
| FMM carrier | 5 | 10 | 56.7 m |
| Learned carrier + fixed gate | 7 | 12 | 33.3 m |
| FMM carrier + fixed gate | 5 | 10 | 56.7 m |

The true-depth waveform L2 reconstructed from these objective values agrees with the archived forward audits to within 1.2e-10. At 80 m offsets, the learned head favors the anchor depth for two a10 events, whereas the FMM head favors it for those and a14 -80 m. For the 160 m offsets, the learned head selects the neighboring 80 m depth on three of four events; the FMM head does so on two and returns to the anchor on one. The fixed taper does not change any selected depth or true-depth rank. Its suppression of early threshold crossings therefore does not resolve depth misranking. The FMM carrier also does not improve this tightly constrained depth search.

The selected-depth errors in this table are **not localization errors**: all candidates have the exact x, y and origin time, the search is confined to seven already examined depths near two training sources, and the FD observations were inspected before this analysis. FMM travel times are looked up on an 80 m grid with up to 60 m receiver snapping and supply no continuous source-coordinate gradient. The models are receiver-supervised heads with no PDE/IC/BC loss, so this result does not establish a valid source-conditioned PINN or independent source recovery. The present operator is not ready to support a four-parameter localization or uncertainty claim in the manuscript. A next candidate must be tested on waveform source response and on a truth-independent search before new held-out events are used.

Complete seven-depth loss curves, event ranks, frozen input hashes and the script are in `fmm_receiver_head_depth_objective_results.json`, `fmm_receiver_head_depth_objective_protocol.json` and `examples/audit_fmm_receiver_head_depth_objective.py`. The script refuses to overwrite its existing result.
