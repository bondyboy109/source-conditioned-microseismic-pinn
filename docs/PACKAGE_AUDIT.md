# Current distribution and historical provenance audit — 2026-10-09

Scripts, experimental protocols, JSON results, selected checkpoints and observation archives are copied byte-for-byte from the local research checkout. `.gitattributes` disables newline transformations. `SHA256_MANIFEST.json` identifies this distribution; it is not a preregistration or a replacement for historical protocol hashes.

## Known unresolved historical differences

- Confirmation, stress and velocity-bias protocols identify localization runner `add81f105956c073f8b2331f15da37dbddc3906dc996bc5b6d33eb2429694c0e`. That source has not been recovered.
- Both available runner files contain `174a5d727b88ab957dcd659d043a3210cfc6efb8d73e97caeb15887afaaa7ce9`. They are current code, not the historical frozen implementation.
- Stress and velocity-bias protocols retain confirmation-result hash `fdc1f68704bd60687f782b4fe88ab3b9c0ab34e4541613b0746ed3b816c36ad5`; the current archived result has `a8fcdfbe2413f0332a819ae5de6853ebaba12e4ed4822b1940f2ad442d807e2d`.
- The missing `source_transfer_final_blind_results.partial.json` was a temporary output of a completed run, not a required scientific input.

These differences are listed in `core_hash_audit.json`. Original protocols are not rewritten to pretend that the differences have disappeared. The historical confirmation entry point rejects changed frozen inputs and existing results; use `scripts/replay_confirmation.py` for a current-version replay into new output files. Such replay is an installation/interface check, not new independent confirmation or complete historical byte-identical reproduction.

The superseded reliability JSON carries `_legacy` and status annotations. Use `reliability_adaptive_fusion_results_v2.json`. No frozen result is overwritten by package validation. Some old Markdown reports describe remote/upload status at their original dates; this document is the current distribution guide.

## Distribution

Ordinary Git files contain the reviewed small checkpoints/observations and eight current result figures. Three large ZIPs are supplied separately through the artifact Release. The original Overthrust velocity export and the unpublished full LaTeX manuscript are not included. No reuse license has been chosen. Repository visibility remains the owner's existing setting.

The local training checkout is not cleared or reset during publication; a clean clone receives the replacement snapshot. Older remote contents remain recoverable from Git history.
