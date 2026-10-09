# Frozen FMM quality-gate holdout synthesis

The tested rule was selected on development events and then frozen at `max_azimuth_gap_deg <= 168.38736722796608` degrees. It only accepts or abstains from an FMM estimate; it does not alter the estimate.

| Dataset | Eligible 4-5 pick cases | Accepted | FMM successes <=500 m | Accepted successes | Coverage | Precision among accepted | Successful estimates retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent source confirmation 1 | 15 | 5 | 3 | 3 | 33.3% | 60.0% | 100.0% (3/3) |
| Independent source replication 2 | 13 | 0 | 6 | 0 | 0.0% | NA | 0.0% (0/6) |
| Pooled descriptive counts | 28 | 5 | 9 | 3 | 17.9% | 60.0% | 33.3% (3/9) |

In the second replication, every eligible estimate had a maximum azimuth gap above the frozen 168.39-degree cutoff (range 180.0-286.5 degrees). The gate therefore rejected all 13 estimates, including all six whose location errors were within 500 m. In the first confirmation, three accepted estimates came from only three source locations; two additional accepted estimates from one source had errors of 1215.6 m and 1725.6 m.

The first confirmation suggested a selective-risk signal at low coverage, but the second independent source set did not reproduce it. The pooled 60% conditional precision is based on only five accepted cases and hides that the gate retained just one third of all successful FMM estimates. The present azimuth-gap rule should not be presented as a generally reliable quality gate or as a localization-accuracy improvement.

The second replication was designed after the first confirmation had been reviewed. Its source locations were selected by spatial coverage and maximin distance to archived source truths, without using source-specific localization outcomes; the gate threshold, picker, noise levels, and FMM search remained frozen. This is useful replication evidence, but it is not a preregistration made before any confirmation results were observed.

Do not retune this rule on either holdout and then reuse these same cases as validation. Any replacement gate must be selected using development data and evaluated on a new, untouched source set.

## Artifacts

- First confirmation: `experiments/fmm_quality_gate_confirmation_results.md`
- Second replication: `experiments/fmm_quality_gate_replication2_results.md`
- First protocol: `experiments/fmm_quality_gate_confirmation_protocol.json`
- Second protocol: `experiments/fmm_quality_gate_replication2_protocol.json`
- Frozen development gate: `experiments/fmm_quality_gate.json`
