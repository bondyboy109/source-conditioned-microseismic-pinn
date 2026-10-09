# Data and assets

Final confirmation archives contain `data` (25×1001 float32 receiver samples), `times` (0–2 s, dt=0.002 s), `receiver_coords` (25×3, metres), and `source_truth` (x,y,z in metres and origin time in seconds). Sidecar JSON records source/FD provenance. Truth is used only after searches to score estimates. Receiver coordinates are stored in every NPZ; do not substitute a different geometry.

Older diagnostics may use different lengths and grids. Read the associated protocol rather than treating all NPZ files as final observations. Selected training/development/confirmation archives and small checkpoints are committed. Large caches and the independent synthetic geology array are separate Release ZIP assets; paths and SHA-256 are in `release_assets_manifest.json`.

The original `data/velocity/vp_model.npy` Overthrust export (401×401×94 float32) is not redistributed. Obtain it under the source dataset's terms for regenerating FD data or FMM fields. Its recorded SHA-256 is in the asset manifest. The final archived-event replay uses the supplied reciprocal FMM cache without loading that external velocity array.

FD for final confirmation: 40 m spacing, 5 Hz, dt=0.002 s, 0–2 s, zero Dirichlet on six faces, no absorbing layer. FMM: 80 m spacing, 25 reciprocal receiver fields, no time-stepping parameter. The supplied independent layered/undulating array is a separate synthetic geology audit, not an Overthrust export.
