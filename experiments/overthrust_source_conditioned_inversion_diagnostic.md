# Overthrust source-conditioned inversion diagnostic

This experiment evaluates the source-conditioned wavefield interface after the
forward-model gate was reached at one development checkpoint. It is a
diagnostic, not an independent test: the three development sources were used
to select the checkpoint, and `heldout_01` had been used in earlier
development runs.

## Checkpoint and inversion settings

- Checkpoint: `outputs/overthrust_eikonal_hybrid_train16_5hz_continued3000/conditioned_5Hz_epoch_2000.pth`
- Frequency: 5 Hz; 16 registered training sources; eikonal carrier frozen.
- Forward-model relative L2 on the three development sources: `0.750`,
  `0.960`, and `0.967` for 250, 800, and 2000 m offsets, respectively.
- Coarse grid: `8 x 8 x 5 x 6` candidates in `(x, y, z, ts)`.
- Local optimization: bounded Adam, 300 iterations, learning rate `5e-3`.
- Receiver stride: 2; time stride: 4.

## Results

| Event | True `(x,y,z,ts)` | Estimated `(x,y,z,ts)` | `eloc` (m) | `et` (s) | All source gradients nonzero |
| --- | --- | --- | ---: | ---: | :---: |
| dev 250 m | (3750, 4000, 500, 0.12) | (3714.5, 3714.4, 800.0, 0.0500) | 415.7 | 0.0700 | no |
| dev 800 m | (4300, 4000, 500, 0.12) | (5428.4, 5428.3, 800.0, 0.2300) | 1844.8 | 0.1100 | yes |
| dev 2000 m | (5500, 4000, 500, 0.12) | (5428.7, 3714.2, 450.0, 0.0500) | 298.8 | 0.0700 | no |
| heldout 01* | (10500, 10500, 650, 0.23) | (10572.7, 10573.4, 1149.9, 0.1400) | 510.5 | 0.0900 | yes |

`* heldout 01 is included only as a development diagnostic and must not be
reported as an independent generalization result.`

The candidate-ranking diagnostic on `heldout_01` gave the true source the
lowest relative L2 among the true source and the 16 training-source
candidates (`0.9091` versus `1.0457--1.2821`). This confirms that the explicit
source input changes the forward prediction with the candidate source. It does
not establish metric localization accuracy: the subsequent continuous local
optimization still produced a 510.5 m error, mainly from a 500 m depth error
and a 0.090 s origin-time error.

As a search-resolution control, the 800 m development event was rerun with a
`12 x 12 x 8 x 12` coarse grid, 500 local iterations, and learning rate
`2e-3`. The estimate became `(5272.8, 4181.6, 700.0, 0.1727)` and `eloc`
decreased to `1009.6 m` (`et=0.0527 s`), but the location remained far from
the true `(4300, 4000, 500, 0.12)`. Thus grid coarseness explains part, but
not all, of the localization error.

## Interpretation

The source-conditioned interface removes the former structural defect in which
the wavefield was independent of the candidate source. It is differentiable
with respect to all four source parameters, but the current Overthrust model
still has a broad or poorly conditioned misfit surface. The coarse grid and
the 0.05 s origin-time spacing also limit the reported errors. These results
should therefore be used to motivate a denser multistart inversion and an
independent test set, rather than as the paper's final localization accuracy.

Generated audit files are in the same output directory, including the three
`*_inversion_diagnostic.json` files and `heldout_01_candidate_ranking.json`.
