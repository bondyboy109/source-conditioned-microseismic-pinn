# Eikonal carrier with local depth coverage (development)

This paired development experiment adds four previously inspected 5 Hz FD records at depth offsets of +/-160 m around training anchors a10 and a14 to the existing 29-event receiver-training manifest. The resulting 33-event model was trained from scratch with the same frozen eikonal carrier, seed 42, network, sampling density per event and two-stage schedule as the 29-event baseline. Eight disjoint but already-inspected records at +/-80 and +/-280 m provide the development comparison. They are absent from both training manifests, but they are not new confirmation data.

| Mean metric | 29-source baseline | 33-source depth model | Relative change |
|:--|--:|--:|--:|
| Full-record waveform L2 | 0.661 | 0.435 | -34.2% |
| FD-active-sample waveform L2 | 0.593 | 0.372 | -37.3% |
| Finite depth-response L2 | 0.795 | 0.466 | -41.4% |
| Finite depth-response cosine | 0.614 | 0.886 | +44.3% |
| Early threshold picks | 47 | 42 | -10.6% |
| Mean threshold-pick RMSE (s) | 1.158 | 1.043 | -9.9% |

All eight events improve in active-sample L2 and depth-response L2. At the harder +/-280 m offsets, active-sample L2 decreases from 0.853 to 0.564 and response L2 decreases from 0.658 to 0.479; response cosine increases from 0.744 to 0.872. Both anchor neighborhoods improve: a10 active L2 changes from 0.681 to 0.421 and a14 from 0.506 to 0.323. Receiver pretraining took 33.25 s and hybrid training took 81.53 s on the local RTX 4060 Laptop GPU.

The registered waveform/depth-response development condition is therefore met. The result also shows the remaining limit: 42 model threshold crossings still occur before origin time minus 0.10 s, so raw threshold arrivals remain invalid. The fixed approximate onset taper can suppress that artifact without modifying sampled FD active energy, but it must be treated as an explicit part of any subsequent candidate-source operator.

This is evidence that receiver-data coverage, rather than the earlier onset gate or straight-path velocity features, was the effective intervention for the tested depth response. It does not establish general source localization, field validity, multi-model generalization or calibrated uncertainty. Four of the twelve depth records now train the model and the other eight have been repeatedly inspected. The next development check must apply the frozen candidate to an already-consumed cross-region localization set, preserving the existing FMM-seeded protocol. Only after that succeeds should new source coordinates be registered for one-time confirmation.

Machine-readable values and hashes: `eikonal_depth_coverage33_development_results.json`. Reproduction: `examples/run_eikonal_depth_coverage33.py` and `examples/audit_eikonal_depth_coverage33.py`.
