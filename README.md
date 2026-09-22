# Source-Conditioned PINN for Microseismic Source Imaging

This repository implements a source-conditioned physics-informed neural
network (PINN) for three-dimensional acoustic microseismic source inversion.
The operator represents

```text
u_theta(x, y, z, t; xs, ys, zs, ts)
```

so source location and origin time enter the forward model explicitly. A
single trained operator can evaluate candidate sources, and waveform-misfit
gradients can update all four source parameters.

## Status

The source-conditioned implementation is complete and passes a one-step CUDA
smoke test. Full multi-source validation is still in progress. Results from the
earlier four-input implementation must not be interpreted as validation of the
revised method.

## Repository layout

```text
.
|-- source_conditioned_pinn.py       # network, PDE losses, checkpoints
|-- train_source_conditioned_pinn.py # frequency-progressive training
|-- invert_source_conditioned.py     # coarse search and local inversion
|-- examples/
|   `-- create_example_observation.py
|-- legacy/                          # archived fixed-source implementation
|-- data/README.md                   # required external data layout
|-- outputs/.gitkeep                 # local outputs; contents are ignored
|-- requirements.txt
|-- environment.yml
|-- CITATION.cff
`-- README_zh.md
```

## Installation

Python 3.10 and an NVIDIA GPU are recommended. Create an environment with
Conda:

```bash
conda env create -f environment.yml
conda activate source-conditioned-pinn
```

Alternatively, install the packages in an existing PyTorch environment:

```bash
python -m pip install -r requirements.txt
```

Choose the PyTorch build that matches the local CUDA driver. See the official
PyTorch installation page when a CUDA-specific wheel is required.

## Data preparation

Obtain the SEG/EAGE 3-D Overthrust model under its original distribution
terms. Convert or reconstruct the velocity array and place it at:

```text
data/velocity/vp_model.npy
```

The expected physical domain is 20,000 m by 20,000 m by 4,650 m. The array must
be a three-dimensional NumPy array in the same axis order used to generate the
receiver data. See [data/README.md](data/README.md) for observation fields.

## Smoke test

```bash
python train_source_conditioned_pinn.py \
  --velocity data/velocity/vp_model.npy \
  --frequencies 5 --epochs 1 --n-phys 16 --n-ic 8 \
  --n-bc-per-face 2 --save-every 1 --output outputs/smoke
```

The command should create `conditioned_5Hz.pth` and
`training_summary.json`.

## Frequency-progressive training

```bash
python train_source_conditioned_pinn.py \
  --velocity data/velocity/vp_model.npy \
  --frequencies 5 10 15 20 30 40 60 \
  --epochs 2000 --n-phys 4096 --n-ic 2048 \
  --n-bc-per-face 341 --save-every 200 \
  --output outputs/source_conditioned
```

This is a long GPU run. Record any reduced batch sizes in publications rather
than reporting the defaults.

## Source inversion

An observation archive must contain `data`, `times`, and `receiver_coords`.
`source_truth` is optional for inference and required for error reporting.

```bash
python invert_source_conditioned.py \
  outputs/source_conditioned/conditioned_60Hz.pth \
  data/observations/event_01.npz \
  --output outputs/source_conditioned/event_01.json
```

The result reports the coarse and refined source, runtime, errors when truth is
available, and the waveform-loss gradient with respect to
`(xs, ys, zs, ts)`.

Only load checkpoints created by this project or another trusted source.
PyTorch checkpoints can contain pickled Python objects.

## Validation required for scientific claims

Use independent finite-difference observations for multiple held-out sources.
Report component errors, location error, origin-time error, waveform error,
runtime, and failure rate. Compare against travel-time localization, matched FD
waveform inversion, and the archived average-velocity shift. Noise experiments
must use multiple realizations at each SNR.

## Legacy code

The files under `legacy/` reproduce earlier workflow components. In particular,
`source_inversion_legacy.py` uses a fixed-source/four-input interface and an
average-velocity approximation. It is retained only for audit and negative
control experiments.

## License

No reuse license has been selected yet. Add an appropriate `LICENSE` file
before advertising reuse permissions.
