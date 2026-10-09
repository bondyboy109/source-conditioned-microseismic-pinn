# Differentiable FMM-Aligned MoE for Microseismic Localization

Code and experiment artifacts for **A Differentiable FMM-Aligned Mixture-of-Experts Interface for Microseismic Localization**, by Ruyun Tian and Xiaopeng Wu.
[中文说明](README_zh.md) · [Experiment index](experiments/INDEX.md) · [Reproducibility limits](docs/PACKAGE_AUDIT.md) · [Large assets](https://github.com/bondyboy109/source-conditioned-microseismic-pinn/releases/tag/research-artifacts-2026-10-09)

## Current method

The final implementation combines a receiver-supervised, source-conditioned waveform surrogate, reciprocal heterogeneous fast-marching (FMM) travel-time fields, differentiable trilinear interpolation in source coordinates, three jointly trained experts with a prototype gate, and a waveform–arrival hybrid localization objective. Source position and origin time explicitly affect the forward interface.

The final model is trained on receiver waveforms; it is not a PDE-only PINN. The repository name and original PINN modules are retained for provenance. Earlier fixed-source, physics-only, carrier, and transfer experiments are development/legacy audits; they do not independently validate the final interface.

## Final archived comparison

The registered experiment used 33 training sources, 16 development sources and 8 previously unused confirmation sources at 5 Hz, with known velocity. Each final observation has 25 receivers × 1001 samples, dt = 0.002 s and a 0–2 s window. The FD grid is 401 × 401 × 94 at 40 m spacing, with zero Dirichlet boundaries on all six faces and no absorbing layer. The FMM cache contains 25 reciprocal fields at 80 m spacing; FMM does not use a time step.

| Branch | Median location error (m) | Median origin-time error (s) | Location <500 m | Pipeline failures |
|---|---:|---:|---:|---:|
| Waveform-only | 2400.2 | 0.0923 | 0/8 | 0/8 |
| Hybrid | 362.9 | 0.0352 | 5/8 | 0/8 |
| FMM baseline | 276.8 | 0.0082 | 7/8 | 0/8 |

All-event statistics include all eight estimates. Exceeding 500 m is an accuracy-threshold miss, not a pipeline failure. Waveform-only uses the same FMM-feature MoE and sets the arrival-loss weight to zero. Hybrid improves on waveform-only in all eight events, while the FMM baseline has lower overall errors. These results do not establish superior speed or accuracy over FMM, positive transfer gain, or generalization to field data.

The separate carrier ensemble uses M=5 (seeds 42, 7, 123, 2028, 4096). Its **nominal 95% ensemble-dispersion intervals** are uncalibrated and reflect initialization dispersion; their coverage does not validate uncertainty in the final three-expert MoE.

## Install

Python 3.10 is the recorded environment. Install the appropriate PyTorch build for your hardware, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The measured environment is in [docs/environment_snapshot.json](docs/environment_snapshot.json): RTX 4060 Laptop GPU, PyTorch 2.5.1+cu121 and CUDA 12.1. `environment.yml` is a convenience environment, not a guarantee of identical hardware or future dependency resolution. Some optional historical scripts need additional packages such as SciPy, ObsPy, segyio or tqdm; these are not required for the final replay.

## Install large assets and replay

Download the three ZIP attachments from the [artifact Release](https://github.com/bondyboy109/source-conditioned-microseismic-pinn/releases/tag/research-artifacts-2026-10-09) to a local folder. Keep them out of ordinary Git commits:

```bash
python scripts/install_release_assets.py --asset-dir /path/to/downloaded-assets
python scripts/verify_package.py --require-assets
python scripts/replay_confirmation.py --check-inputs
python scripts/replay_confirmation.py --events 1
python scripts/replay_confirmation.py
```

The installer checks exact archive paths, file sizes and SHA-256. The final replay needs `receiver_fmm_fields_stride2.npy.zip`; the other two assets support transfer and geology audits. The cached replay does not require the external Overthrust velocity array. Original Overthrust exports are not redistributed; see [data/README.md](data/README.md).

Replay outputs go to `outputs/reproduction/`, and existing outputs are never overwritten. This is a **current-version replay of archived events**, not a new independent confirmation. See the historical-source limitation below before interpreting reproducibility.

## Files

- `traveltime_aligned_moe.py`: final travel-time interpolator and MoE.
- `examples/run_traveltime_aligned_moe_development.py`: registered receiver-supervised training.
- `examples/run_traveltime_aligned_moe_localization_current.py`: available current localization runner.
- `scripts/replay_confirmation.py`: safe replay entry point using current functions.
- `experiments/`: original protocols, JSON results and reports; [INDEX.md](experiments/INDEX.md) distinguishes evidence roles.
- `outputs/`: selected small checkpoints and metadata; `data/observations/`: selected synthetic observations.
- `docs/figures/`: eight current experiment figures.
- `tests/`: numerical/interface checks; `legacy/`: original fixed-source code.

Training uses seed 3101, three experts with four hidden layers of width 128 and SiLU, 153228 trainable parameters, 3000 Adam updates and learning rate 3e-4. Full settings, source lists and receiver coordinates remain in the protocols, checkpoints and observation archives. The original training entry point can write its recorded experiment output paths; run new training in a separate checkout and preserve archived results.

## Provenance limitation

Three historical protocols record runner SHA-256 `add81f…`, whose source has not been recovered. The available runner is `174a5…`; it must not be labelled as that historical frozen implementation. Two stress/velocity-bias protocols also retain an earlier confirmation-result hash. Original scripts, protocols and results are preserved byte-for-byte rather than retroactively changing their records. [docs/core_hash_audit.json](docs/core_hash_audit.json) and [experiments/protocol_hash_resolution.json](experiments/protocol_hash_resolution.json) list these differences.

Use `reliability_adaptive_fusion_results_v2.json`; the annotated `_legacy.json` is superseded. The package integrity manifest is a current distribution checksum list, not a historical experimental protocol. Older audit reports reflect their original dates; use the current package audit for distribution status.

## License and access

No software reuse license has been selected. This snapshot does not grant MIT or other license permissions. Repository visibility is controlled by the owner; a private repository and its Release assets require access. Citation does not make restricted original data redistributable.
