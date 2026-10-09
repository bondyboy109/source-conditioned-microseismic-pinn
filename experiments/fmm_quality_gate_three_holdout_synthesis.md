# FMM quality-gate holdout synthesis

## Main finding

Neither tested gate is supported as a reliable general-purpose quality filter. The maximum-azimuth-gap rule rejected every estimate in the second holdout, including all six successful FMM estimates. The development-thresholded arrival-RMSE rule retained only four of nine successful estimates in its follow-up dataset. Both rules abstain from estimates; neither changes or improves the underlying FMM location.

## Locked holdout results

| Dataset and frozen rule | Eligible cases | Accepted | FMM successes (<=500 m) | Accepted successes | Coverage | Precision among accepted | Successful estimates retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| Holdout 1: max azimuth gap | 15 | 5 | 3 | 3 | 33.3% | 60.0% | 100.0% (3/3) |
| Holdout 2: max azimuth gap | 13 | 0 | 6 | 0 | 0.0% | NA | 0.0% (0/6) |
| Follow-up: arrival RMSE | 31 | 11 | 9 | 4 | 35.5% | 36.4% | 44.4% (4/9) |

The first two holdouts test the same azimuth rule. Together they contain 28 eligible cases, 5 accepted cases, and 9 successful FMM estimates; the rule retains 3/9 successes (33.3%). This descriptive pooling must be read with its zero-coverage second holdout.

The arrival-RMSE follow-up is a different rule, so its counts should not be pooled with the azimuth-rule counts as if one fixed method had been tested. Its 36.4% accepted precision is only slightly above the 29.0% always-accept success rate, at 35.5% coverage, and it discards five of the nine successful FMM estimates.

## Arrival-RMSE follow-up details

The frozen threshold was `arrival_rmse_s <= 0.09508478690085563 s`, fitted using the original development set only. The residual feature was selected for this follow-up after the azimuth rule had failed on two holdouts. The follow-up is independent of threshold fitting, but the feature choice was made after prior holdout results were reviewed; it is not a pristine preregistered comparison.

There were 120 generated observations from 12 source locations: 12 clean records and 108 noisy records (1, 2, and 3 dB, three realizations per source and SNR). Only 31 records met the locked 4-to-5-valid-pick eligibility condition. The eligible counts and gate outcomes were:

| SNR | Eligible | Accepted | FMM successes | Accepted successes |
|---:|---:|---:|---:|---:|
| Clean | 1 | 1 | 1 | 1 |
| 1 dB | 1 | 0 | 1 | 0 |
| 2 dB | 11 | 4 | 3 | 0 |
| 3 dB | 18 | 6 | 4 | 3 |

Only three source locations contributed the four successful accepted estimates. The 120 records are not 120 independent source tests: noise conditions are clustered within 12 source waveforms, and only 31 entered the specified sparse-pick analysis.

## Interpretation and next use

These results do not support describing either gate as a validated improvement to FMM localization. At most, the arrival residual is a development-set candidate for selective abstention whose apparent benefit is small, noisy, and accompanied by loss of more than half of the FMM successes. Do not retune either threshold or feature on these holdouts and then report the same cases as validation.

Further gate work should begin only with a rule specified entirely on development data and a new untouched source set. For the paper's main localization claim, report the ungated FMM errors and the source-level holdout performance directly; keep the gate as a limitation or exploratory result unless an independently selected rule generalizes with useful coverage and retains most successful estimates.

## Reproducibility artifacts

- First azimuth holdout: `experiments/fmm_quality_gate_confirmation_protocol.json` and `experiments/fmm_quality_gate_confirmation_results.json`.
- Second azimuth holdout: `experiments/fmm_quality_gate_replication2_protocol.json` and `experiments/fmm_quality_gate_replication2_results.json`.
- Arrival-RMSE follow-up protocol, generated-record manifest, results, and detailed report: `experiments/fmm_quality_gate_residual_confirmation_protocol.json`, `experiments/fmm_quality_gate_residual_confirmation_manifest.json`, `experiments/fmm_quality_gate_residual_confirmation_results.json`, and `experiments/fmm_quality_gate_residual_confirmation_results.md`.
