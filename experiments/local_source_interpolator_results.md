# Time-aligned local source interpolation baseline

This development-only diagnostic asks whether the 33 archived FD training sources already contain enough local information to predict nearby source waveforms without a neural model. For each target source, training records are aligned to the target origin time, the nearest source records are selected in scaled `(x,y,z)` space, and their complete receiver waveforms are combined using inverse-distance weights.

The scan evaluated 75 configurations on eight already inspected depth records. The selected configuration uses two neighbours, a 1000 m horizontal scale, a 50 m depth scale, and inverse-distance-squared weights. Hyperparameter selection and evaluation use the same spent development records, so the numbers are descriptive and cannot be treated as independent validation.

| Depth group | Records | Local interpolation full L2 | Neural control full L2 | Local interpolation active L2 | Neural control active L2 |
|:--|--:|--:|--:|--:|--:|
| ±80 m, inside local brackets | 4 | 0.153 | 0.202 | 0.148 | 0.174 |
| ±280 m, outside local brackets | 4 | 0.721 | 0.646 | 0.626 | 0.563 |
| All eight | 8 | 0.437 | 0.424 | 0.387 | 0.369 |

At ±80 m, the local interpolator lowers full-record L2 by about 24% relative to the neural control. All four targets lie exactly halfway between same-location FD depth records, and the selected weights are 0.5/0.5. This confirms that the archived data contain a strong local source-response structure that the current network does not fully reproduce.

At ±280 m, the targets lie outside the local depth brackets. The convex interpolator cannot extrapolate the waveform response and is worse than the neural model, while both errors remain large. The all-record mean therefore slightly favors the neural control, but this average combines two different regimes: successful local interpolation and failed source-depth extrapolation.

The next model experiment should use the accurate local FD interpolants as explicit midpoint supervision rather than penalizing consistency between the network's own endpoint predictions. That test can determine whether interpolation-based data augmentation transfers the known local structure into the source-conditioned network. It remains a development experiment; a new independent localization set is justified only after forward error passes a predefined gate.

Full configurations, neighbours, weights, hashes and per-event metrics are in `local_source_interpolator_protocol.json` and `local_source_interpolator_results.json`. Run `python examples/audit_local_source_interpolator.py` from the repository root to reproduce the scan; the script refuses to overwrite an existing result.
