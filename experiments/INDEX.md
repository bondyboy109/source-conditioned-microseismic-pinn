# Experiment evidence index

Read the paired protocol, JSON and Markdown report for each experiment. Recorded paths may name the original Windows checkout; runnable final tools resolve files relative to the cloned repository.

| Record prefix | Role |
|---|---|
| `traveltime_aligned_moe_development` | Forward-model development audit, 33 training and 16 inspected development sources |
| `traveltime_aligned_moe_localization` | Development comparison and fixed objective/search settings |
| `traveltime_aligned_moe_confirmation` | Archived independent source-position confirmation on 8 unused sources; historical runner source is unavailable |
| `traveltime_aligned_moe_stress` | Noise/receiver-loss follow-up; read its own protocol and hash limitation |
| `traveltime_aligned_moe_velocity_bias` | Velocity-bias follow-up; read its own protocol and hash limitation |
| `source_transfer_final_blind` | Separate source-pretraining/target-transfer ablation; does not establish positive transfer gain |
| `source_conditioned_ensemble_confirmation` | Separate M=5 carrier ensemble; uncalibrated dispersion, not final-MoE uncertainty |
| `cross_model_geology` | Separate geology audit; no field-data claim |
| `computational_cost` | Recorded compute costs; no established speed advantage over FMM |
| `reliability_adaptive_fusion_results_v2` | Current reliability-development result; `_legacy` is superseded |

Remaining protocols/reports cover inspected development trials, negative results and legacy diagnostics. They are preserved for provenance, not pooled with the final confirmation. A failed gate is part of the record. `REPRODUCIBILITY_AUDIT.md` is a historical report: distribution status and hash discrepancies are documented in `../docs/PACKAGE_AUDIT.md` and `protocol_hash_resolution.json`.
