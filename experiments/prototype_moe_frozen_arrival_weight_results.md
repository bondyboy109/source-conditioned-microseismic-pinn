# Frozen arrival-weight evaluation

The arrival-loss weight was selected only on the registered development set,
then frozen before evaluating the independent eight-event holdout. The search
used 27 FMM-seeded local candidates per event, 160 m FMM spacing, and the same
threshold first-arrival picker for development and holdout.

The development set contained 15 usable events. One wide-search record had no
valid FMM starting source and was excluded before weight selection; it was not
silently counted as a failure.

## Development selection

| objective | mean location error | mean origin-time error |
|---|---:|---:|
| Waveform only | 2038 m | 0.170 s |
| FMM arrival only | 584 m | 0.137 s |
| Hybrid, alpha = 0.10 | 701 m | 0.152 s |
| Hybrid, alpha = 0.25 | **572 m** | 0.140 s |
| Hybrid, alpha = 0.50 | 584 m | **0.137 s** |
| Hybrid, alpha = 1.00 | 584 m | 0.137 s |

The spatial development criterion selected `alpha=0.25`. No holdout truth was
used to choose this value.

## Frozen holdout evaluation

| objective | mean location error | median location error | mean origin-time error |
|---|---:|---:|---:|
| Waveform only | 2635 m | 2504 m | 0.151 s |
| FMM arrival only | 1097 m | 1128 m | 0.121 s |
| Frozen hybrid, alpha = 0.25 | **1021 m** | 1050 m | 0.097 s |

The frozen hybrid reduces mean location error by 6.9% relative to FMM-only
local refinement and by 61.2% relative to the waveform-only objective. These
numbers are based on eight independent holdout events and one prototype-MoE
checkpoint. The improvement is a diagnostic result, not a claim of universal
localization accuracy.

## Limitations

The FMM term is evaluated on a discrete local candidate grid, so this is an
arrival-constrained grid refinement rather than a fully differentiable FMM
optimization. The first-arrival picker is threshold based, and one development
record lacked a valid FMM initialization. A final publication experiment
should add a robust picker and a differentiable eikonal or interpolated FMM
travel-time interface.

Machine-readable outputs:

- `experiments/prototype_moe_arrival_weight_development.json`
- `experiments/prototype_moe_hybrid_frozen_alpha25_holdout.json`

Scripts:

- `examples/select_arrival_weight_development.py`
- `examples/run_prototype_hybrid_frozen_alpha25.py`
