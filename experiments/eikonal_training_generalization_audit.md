# Frozen eikonal operator: source-position generalization audit

This is a **post-hoc development diagnostic**, not a new independent test. The 5 Hz, 29-source eikonal-carrier checkpoint is unchanged (`ce744ba5e7249850fe99644449855c886fc255aebacfacab86f76d59990f2ec7`). At each known FD source, the frozen operator predicts the same 25 receiver traces over 0-2 s. Full relative L2 is `||PINN - FD||_2 / ||FD||_2`; active-sample L2 applies the same formula where `|FD|` exceeds 5% of that event's overall waveform peak. The audit checks the original checkpoint training manifest, observation hashes and group membership.

| Existing group | Events | Mean distance to nearest trained spatial source (m) | Mean full L2 | Mean active L2 | Mean peak-amplitude ratio | Mean zero-lag correlation |
|:--|--:|--:|--:|--:|--:|--:|
| Exact training sources | 29 | 0 | 0.148 | 0.083 | 0.980 | 0.989 |
| Training site, changed origin time | 3 | 0 | 0.156 | 0.112 | 0.988 | 0.988 |
| Small x shifts near training site 01 | 3 | 53 | 0.157 | 0.111 | 0.972 | 0.988 |
| Orthogonal shifts around two training sites | 10 | 120 | 0.274 | 0.230 | 0.997 | 0.958 |
| Four previously inspected nearby sites | 4 | 213 | 0.557 | 0.499 | 1.173 | 0.859 |
| Eight previously inspected development sites | 8 | 674 | 0.808 | 0.746 | 0.733 | 0.570 |
| Eight already spent independent-test sites | 8 | 1340 | 0.910 | 0.853 | 0.510 | 0.417 |

The model fits the training traces well, so these results do not point to a failure to fit its 29 supervised sources. The largest gap is spatial generalization: the mean true-source waveform error rises from 0.148 on training records to 0.910 on the eight already spent independent events. The latter per-event L2 values agree with the original frozen independent report within `3.4e-7`; that test's failures remain unchanged. A zero waveform predictor has relative L2 exactly 1, so values near 1 do not demonstrate a useful forward prediction.

Controlled existing FD records help separate the effects. Changing only the origin time at site 01 from 0.12 s to 0.16, 0.20 or 0.28 s keeps mean L2 at 0.156. Shifting site 01 by 40, 80 or 160 m along x keeps L2 around 0.153-0.160; the PINN's finite waveform changes agree with FD changes with a cosine of 0.991-0.993 and change-relative L2 of 0.123-0.137. Thus the new eikonal-carrier checkpoint **does improve the short-range x response** relative to earlier checkpoints, and it would be wrong to call every unseen source a failure. The 160 m x-shift point is only 40 m from another trained source at x=3700 m; the x sweep is not a clean measurement of a 160 m distance from *all* training data.

Orthogonal spatial changes are less uniformly captured: 120 m y shifts give L2 0.254-0.384 across the two sites; 80 m z shifts give L2 0.140-0.220. The four nearby mixed-position events have L2 0.427-0.702. Direction, event depth, amplitude and distance vary between these groups; their aggregates describe the current checkpoint, not a universal operating radius or a causal distance law.

These findings locate the next development problem at **source-position generalization over the intended monitoring area**, including receiver amplitude and phase, rather than the time-coordinate interface alone. Measure forward accuracy as a function of source separation and direction on a fixed, predefined development grid, then assess a method that retains waveform fidelity over the regions where events must be located. Any revised operator and search must be frozen and tested on different, previously unseen sources; none of the groups above is an untouched confirmation set.

Detailed group metrics, each source's nearest-training distance and hashes: `eikonal_training_generalization_audit.json`. Finite x-displacement response: `eikonal_spatial_response_development.json`. Reproduction scripts: `examples/audit_eikonal_training_generalization.py` and `examples/audit_spatial_source_response.py`.
