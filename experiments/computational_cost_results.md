# Measured computational cost and break-even analysis

This report aggregates completed measurements; it does not retrain or retune a model.

| Quantity | Measured value |
|---|---:|
| Device | cuda (PyTorch 2.5.1+cu121) |
| Trainable parameters | 67,458 |
| PINN training time | 72.58 s |
| PINN peak CUDA allocation | 120.7 MiB |
| PINN inverse time per event | 3.928 s mean; 3.764 s median |
| FMM field setup | 36.88 s |
| FMM search time per event | 0.084 s mean; 0.084 s median |
| FMM cached receiver fields | 181.1 MiB |
| FD forward time per event | 8.605 s mean; 8.604 s median |

## Arithmetic crossover

The crossover is the smallest event count at which the accumulated measured cost is no larger than the comparator. It is a cost calculation, not an accuracy comparison.

| Comparison | Events to amortize |
|---|---:|
| PINN vs FD, inference only | 16 |
| FMM vs FD, inference only | 5 |
| FMM becomes cheaper in total than PINN | 1 |
| PINN vs FD, including FD generation for 25 supervised records | 62 |

The PINN per-event inverse is slower than the cached-field FMM search in this run. FD timings here measure forward-record generation, not a complete matched FD inversion. The accuracy results must therefore be reported separately from this arithmetic cost analysis.
