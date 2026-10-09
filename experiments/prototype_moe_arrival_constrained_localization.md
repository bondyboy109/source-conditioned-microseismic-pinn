# Arrival-constrained source localization

The waveform-only objective of the source-conditioned MoE was augmented with a
receiver first-arrival term. For each FMM-seeded local candidate, the objective
was evaluated as

`J_alpha = J_waveform + alpha (RMSE_arrival / 0.10 s)^2`.

The FMM travel times were computed on the continuous heterogeneous velocity
model downsampled to 160 m spacing. Receiver first arrivals were obtained with
the registered threshold picker. Eight independent holdout events were tested,
with a 27-candidate local neighborhood around each FMM estimate. The same
experiment was repeated with prototype-MoE seeds 2027 and 2028.

## Weight sensitivity

| objective | mean location error | median location error | mean origin-time error |
|---|---:|---:|---:|
| Waveform only | 2,635 m | 2,504 m | 0.151 s |
| FMM arrival only (`alpha=1`) | 1,097 m | 1,128 m | 0.121 s |
| Hybrid (`alpha=0.10`) | **1,010 m** | 1,050 m | **0.073 s** |
| Hybrid (`alpha=0.25`) | 1,021 m | 1,050 m | 0.097 s |
| Hybrid (`alpha=0.50`) | 1,078 m | 1,055 m | 0.106 s |

The `alpha=0.10` hybrid reduces the mean location error by 7.96% relative to
the FMM-only result and by 61.7% relative to the waveform-only result in this
holdout protocol. The weight sensitivity shows that a small waveform term
refines the FMM estimate, whereas making the arrival term dominant returns the
FMM solution.

Because this alpha sweep was run on the same eight-event holdout, `alpha=0.10`
is an exploratory diagnostic and must not be presented as an unbiased final
test. For a publication result, select alpha on a development set, freeze it,
and run one final evaluation on a separate holdout.

The results are identical for the two prototype-MoE seeds at the selected
candidate level:

| seed | hybrid `alpha=0.10` mean location error | hybrid mean origin-time error |
|---:|---:|---:|
| 2027 | 1,010 m | 0.073 s |
| 2028 | 1,010 m | 0.073 s |

This seed agreement should not be overinterpreted: the local search uses a
coarse discrete candidate grid and the arrival term dominates candidate
selection. The implementation is therefore an arrival-constrained local
search, not a fully differentiable FMM objective. A differentiable eikonal
surrogate or an interpolated FMM travel-time field would be required for
gradient-based source refinement.

Machine-readable results and scripts:

- `experiments/prototype_moe_hybrid_local_results_v2.json`
- `experiments/prototype_moe_hybrid_seed2028_results.json`
- `examples/run_prototype_hybrid_local_refinement.py`
- `examples/run_prototype_hybrid_seed2028_refinement.py`
