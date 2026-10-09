# FMM sparse-pick quality-gate development

The development events were used in earlier arrival-picking work. This stage selects a quality-rejection threshold only; the locked-source confirmation is needed to assess generalization.

Four-to-five-pick development cases: **24**.
Always-accept success precision (<=500 m): **0.417**.

| Feature (lower is better) | LOEO coverage | LOEO precision |
|---|---:|---:|
| max_azimuth_gap_deg | 0.500 | 0.750 |
| arrival_rmse_s | 0.417 | 0.700 |
| log10_directional_condition_number | 0.708 | 0.471 |
| log10_directional_gdop | 0.667 | 0.438 |

Selected gate: **accept FMM estimate only if max_azimuth_gap_deg <= 168.387367228**
Gate enabled: **True**.

Threshold selection used leave-one-source-out validation. The locked-source test must evaluate the selected rule without changing it.
