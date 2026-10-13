# Implementation and evidence provenance

The final weights, observation records and original protocols are preserved.
The main displayed result file is a two-neural-branch projection. The complete
supporting snapshot, including original result JSONs, is separately archived.

The historical confirmation/stress/bias protocols record runner `add81f…`;
that source has not been recovered. The available runner is `174a5…` and is
explicitly named current. Two follow-up protocols also refer to an earlier
confirmation-result hash. Neither original protocol hashes nor measured records
are rewritten to make them appear consistent. See core_hash_audit.json and
experiments/protocol_hash_resolution.json. A current replay of already inspected
events is not a new independent confirmation or byte-identical historical run.

The current primary ablation compares waveform-only and hybrid using the same
frozen receiver-supervised MoE and observations. Reported stress failures remain
part of this scope. The archive preserves separate transfer, carrier and other
development records. Carrier M=5 intervals are uncalibrated nominal 95%
ensemble-dispersion intervals and do not validate the final MoE's uncertainty.
The final configuration is synthetic 5-Hz acoustic propagation with a fixed
known velocity and fixed array. This release makes no new performance claims.
