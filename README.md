# Traveltime-Constrained MoE for Microseismic Localization

Code and data for **Traveltime-Constrained Mixture-of-Experts for Microseismic
Localization: A Reproducible Evaluation**, by Ruyun Tian and Xiaopeng Wu.
[中文说明](README_zh.md) · [Data](data/README.md) · [Provenance](docs/PROVENANCE.md)

## Current workflow

The method combines cached reciprocal Eikonal fields, differentiable trilinear
interpolation, source-conditioned retarded-time features, three jointly trained
waveform experts and a prototype gate. Localization optimizes source position
and origin time through a waveform–arrival hybrid objective. The final network
uses receiver-waveform supervision; the repository name retains its origin.

## Primary neural ablation

The two branches share the same frozen surrogate. Waveform-only sets the
arrival-loss weight to zero; hybrid adds the registered arrival term. All eight
previously unused confirmation sources are included.

| Branch | Median location error (m) | Median origin-time error (s) | Below 500 m | Pipeline failures |
|---|---:|---:|---:|---:|
| Waveform-only | 2400.2 | 0.0923 | 0/8 | 0/8 |
| Hybrid | 362.9 | 0.0352 | 5/8 | 0/8 |

The location-error median decreases by 84.9% within this neural ablation, and
all eight location estimates improve. The final acquisition has 25 receivers,
1001 samples, dt=0.002 s and a 0–2-s window. Training uses 33 sources and
development uses 16 inspected sources. The FD grid is 401×401×94 at 40 m,
with zero Dirichlet boundaries and no absorbing layer. The reciprocal cache
uses 25 fields at 80-m spacing and has no temporal time step.

## Quick start

Use Python 3.10 and a suitable PyTorch build for your GPU:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/verify_package.py
```

Download `receiver_fmm_fields_stride2.npy.zip` and
`independent_layered_undulating.npy.zip` from the
[existing data Release](https://github.com/bondyboy109/source-conditioned-microseismic-pinn/releases/tag/research-artifacts-2026-10-09),
then install the hash-verified assets:

```bash
python scripts/install_release_assets.py --asset-dir /path/to/downloaded-assets
python scripts/verify_package.py --require-assets
python scripts/replay_confirmation.py --check-inputs
python scripts/replay_confirmation.py --events 1
python scripts/replay_confirmation.py
```

The primary replay runs waveform-only and hybrid. It writes new outputs under
`outputs/reproduction/` and refuses to overwrite existing results. This is a
current-version replay of archived events, not a new independent confirmation.
Historical runner source/hash differences are documented in
[PROVENANCE.md](docs/PROVENANCE.md).

## Train a new model

The 33/16 observation files and relative manifests are included. After installing
the reciprocal cache, use a new output directory:

```bash
python scripts/train_moe.py --check-inputs
python scripts/train_moe.py --output outputs/new_training/moe_run1
```

Defaults preserve seed 3101, three experts, four 128-unit hidden layers, SiLU,
153228 parameters, 3000 Adam updates and learning rate 3e-4. This creates a new
training run and does not alter the archived checkpoint or experiment records.
Generating new FD observations or rebuilding the cache requires the external
Overthrust velocity export; see the data instructions.

## Files and evaluation scope

- `scripts/`: current replay, fresh training and package/asset verification.
- `traveltime_aligned_moe.py`: source-conditioned MoE and differentiable fields.
- `data/`: manifests, included synthetic observations and saved figure arrays.
- `outputs/traveltime_aligned_moe_development/`: final archived model weights.
- `results/final_neural_ablation.json`: derived neural view with all trials retained.
- `docs/figures/`: latest nine manuscript figures; Fig. 7/8 use single columns.
- `archive/`: complete unmodified supporting records and legacy research snapshot.

Regenerate the current single-column stress and velocity-bias figures from the
preserved inputs with `python scripts/render_neural_figures.py`. It writes only
to a new `outputs/figure_reproduction/` directory and retains every trial.

Stress evaluation keeps its gate failures: 11/24 at 5 dB and 7/24 with 40%
receiver loss. Conditional error summaries use completed trials; failure rates
use every registered trial. The scope is synthetic, fixed-array, known-velocity
acoustic data. Separate transfer/carrier audits remain supporting records;
carrier intervals do not validate uncertainty of the final MoE. FD figure
snapshots illustrate reference propagation rather than MoE full-space accuracy.

## Data access and software terms

The processed Overthrust velocity array is not redistributed; follow its original
terms. Large cached/synthetic assets use the existing Release and unchanged
checksums. Repository and asset access follow the owner's GitHub visibility.
No software reuse license has been selected; this update does not introduce a
new license. Cite the associated manuscript using CITATION.cff.
