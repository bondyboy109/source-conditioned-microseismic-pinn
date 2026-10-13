# Data included in the current package

The relative manifests list 33 training and 16 inspected development NPZ files.
Eight final confirmation records and four independent-geology records are
included with their original sidecars. Receiver coordinates are in
`observations/receiver_coords.npy`; confirmation data are 25 receivers by 1001
samples, dt=0.002 s, 0–2 s. NPZ keys include data, times, receiver_coords and
source_truth. The source truth is read after inversion for scoring, not fed to
the localization search. Training supervision uses the training source truth.

The final checkpoint is committed under its original outputs path. The 25-field
reciprocal cache is installed from the existing Release asset, with its exact
SHA-256. Noise/receiver-loss trials are generated from the eight confirmation
records using the registered seeds. Velocity-bias trials scale the fields by
the inverse factor. Protocols retain their historical bytes and documented
implementation/version limitations.

The original SEG/EAGE Overthrust volume and processed `velocity/vp_model.npy`
are external data and are not redistributed. Obtain them under their original
distribution terms and verify the expected SHA-256 in release_assets_manifest.
Cached inference and cached training do not need this array; generating FD data
or rebuilding the receiver cache does. The independent synthetic layered model
is available in the existing Release. Figure-array snapshots are FD fields;
they are not full-space predictions from the final receiver-only MoE.
