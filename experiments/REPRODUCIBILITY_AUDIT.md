# Reproducibility Audit

Audit date: 2026-10-06

This audit covers the local working tree of `source-conditioned-microseismic-pinn`. The
live GitHub remote could not be contacted during the audit (`github.com:443` was
unreachable), so the remote comparison below uses the local `origin/main` ref only.

## Repository state

- `origin/main` and the local `HEAD` both point to `6922ee7` (`Initial release of source-conditioned PINN`).
- The `origin/main` tree contains 26 files and contains no `experiments/` directory, experiment scripts, checkpoints, velocity models, or observation archives.
- The working tree contains 91 protocol JSON files, but they are not committed. The current experiment scripts, result JSON/Markdown files, checkpoints, FMM caches, velocity models, and observations are also uncommitted or ignored by `.gitignore`.
- The current `.gitignore` excludes `*.pth`, `*.npy`, `*.npz`, `outputs/*`, `data/velocity/*`, and `data/observations/*`. A protocol-only push therefore cannot reproduce the recorded runs.

## Hash verification

The protocol-to-result references were checked against the current working tree. The
following references are consistent:

- `experiments/traveltime_aligned_moe_confirmation_results.json` -> `traveltime_aligned_moe_confirmation_protocol.json`: `21c2584e2ce11b209f50243d6e3ee96058acad2b762c8649e66d34651d4c9286`
- `experiments/traveltime_aligned_moe_localization_results.json` -> `traveltime_aligned_moe_localization_protocol.json`: `62c69be47a4037d919e9704f407bae352b7c36c67d717e02e8877d4df6c9c167`
- `experiments/source_transfer_final_blind_results.json` -> `source_transfer_final_blind_protocol.json`: `3696b99ef9849df9baaafa91811ab4fe7d5ee065ae1f453ebd8bd726f4d25352`
- `experiments/reliability_adaptive_fusion_results_v2.json` -> `reliability_adaptive_fusion_protocol.json`: `ce1fcbe64ce0f8ba7799cf8db9a1b88fc96103bffe95fd81ca57841de6c39282`

Four stale references require an explicit decision before publication:

1. `experiments/reliability_adaptive_fusion_results.json` records protocol hash
   `94b5c0309eb732a83722d9880f9717d1acd980e1833fb2bc37fe9d0544a27ae1`, while the
   current protocol hash is
   `ce1fcbe64ce0f8ba7799cf8db9a1b88fc96103bffe95fd81ca57841de6c39282`. The newer
   `reliability_adaptive_fusion_results_v2.json` is internally consistent and is the
   result named by the protocol's `output_result` field. Treat the unsuffixed result
   as a superseded stale artifact unless its original protocol is recovered.
2. Three historical protocols record the localization runner hash
   `add81f105956c073f8b2331f15da37dbddc3906dc996bc5b6d33eb2429694c0e`, while the
   current runner is
   `examples/run_traveltime_aligned_moe_localization_current.py` with hash
   `174a5d727b88ab957dcd659d043a3210cfc6efb8d73e97caeb15887afaaa7ce9`. The historical
   hashes are retained as provenance; the current runner is not claimed to have
   generated those historical results.

The original reliability result has been retained as
`reliability_adaptive_fusion_results_legacy.json`, explicitly marked with its stale
protocol hash, and superseded by `reliability_adaptive_fusion_results_v2.json`.
The three result JSON files for confirmation, stress, and velocity bias now record
the SHA-256 of their current protocol files.

## Key current hashes

These are the current working-tree hashes for the files involved in the mismatches:

| File | SHA-256 |
|---|---|
| `examples/run_traveltime_aligned_moe_localization.py` | `174a5d727b88ab957dcd659d043a3210cfc6efb8d73e97caeb15887afaaa7ce9` |
| `experiments/traveltime_aligned_moe_confirmation_protocol.json` | `21c2584e2ce11b209f50243d6e3ee96058acad2b762c8649e66d34651d4c9286` |
| `experiments/traveltime_aligned_moe_stress_protocol.json` | `3626cd01b85780ba3d07cde810847dd9686abb319b5a3a5826075554b4d992d6` |
| `experiments/traveltime_aligned_moe_velocity_bias_protocol.json` | `149a17f0c683c8344677b82dc48cc77670294501d520a8de5a20b75a515eaf92` |
| `experiments/reliability_adaptive_fusion_protocol.json` | `ce1fcbe64ce0f8ba7799cf8db9a1b88fc96103bffe95fd81ca57841de6c39282` |
| `experiments/reliability_adaptive_fusion_results_v2.json` | `09153ad9ee51ea17f2cfa266727e9f2238787c5088603221256daecac70195c6` |
| `experiments/traveltime_aligned_moe_confirmation_protocol.json` | `6fc5136f12fdf7eca84df7abfa153ace7adcce0689cc3d434192bedd1f30d861` |
| `experiments/traveltime_aligned_moe_stress_protocol.json` | `815e5232261fc3b48c9a67f1cb776a0fe88b4f95916a59024923393daa1cb9df` |
| `experiments/traveltime_aligned_moe_velocity_bias_protocol.json` | `18c5e3023c69c022bac2a18a923a25ee12bc04803d37408e6e10d0b50e5835b4` |
| `experiments/run_traveltime_aligned_moe_localization_current.py` | `174a5d727b88ab957dcd659d043a3210cfc6efb8d73e97caeb15887afaaa7ce9` |

The manifest deliberately does not claim that large ignored data and model files are
available on GitHub. They must be published through the repository, a release asset,
or a documented external archive, with their recorded SHA-256 values retained.

## Recheck commands

Run these from the repository root after any repair:

```powershell
Get-FileHash .\examples\run_traveltime_aligned_moe_localization.py -Algorithm SHA256
Get-FileHash .\experiments\traveltime_aligned_moe_confirmation_protocol.json -Algorithm SHA256
git status --short --branch
git ls-tree -r --name-only origin/main experiments
git ls-remote origin HEAD refs/heads/main
```

The last command is the authoritative live-remote check; it was unavailable during
this audit because the GitHub endpoint could not be reached.
