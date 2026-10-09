# Distribution validation — 2026-10-09

- Prepared snapshot: 68 tests passed in 4.09 s.
- Three Release ZIP assets installed with exact uncompressed size, path and SHA-256 checks.
- Current replay preflight verified checkpoint, reciprocal cache, localization protocol, current runner and all eight observation hashes.
- First archived event replay: waveform-only 2699.4 m, hybrid 215.6 m, FMM 261.8 m, matching archived values at reported precision. This is an installation check on already-inspected data, not independent confirmation.
- Hash manifests identify supplied files without editing historical protocol records. The missing historical add81f runner and older stress/bias result hashes remain documented limitations.
- Publication uses a separate clean clone. The original training checkout and its experiments are preserved.

The current replay output is saved in validation/current_one_event_check.json. Successful checks do not establish new scientific claims or full byte-identical reproduction of the missing historical implementation.
