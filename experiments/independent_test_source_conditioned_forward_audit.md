# One-time source-conditioned PINN test on independent events

## Frozen protocol

This audit evaluates the already selected checkpoint
`outputs/overthrust_core20_train16_jointbatch_receiver_5hz_5000/conditioned_5Hz.pth`
without retraining, checkpoint selection, or parameter changes. It tests the
three previously registered independent FD events in
`data/observations/independent_test/window_0to2s/`. SHA-256 checks confirm
that none of the files occurs in the checkpoint's receiver-training records;
their true source coordinates also do not match a training source.

The forward metric is original-amplitude relative L2 against the independent
second-order FD record. The zero-prediction baseline is 1.0. A global amplitude
gain fitted using each test record is reported only as a diagnostic, not as a
validation score. Candidate ranking compares the true source against the 16
recorded training-source locations, with time stride 2; it is not a continuous
source inversion.

## Results

| Independent event | True source `(x,y,z,ts)` | Relative L2 | Test-fitted gain diagnostic L2 | True candidate ranks first |
| --- | --- | ---: | ---: | :---: |
| 01 | (2500, 13000, 1100, 0.16) | 1.1386 | 0.9749 | Yes |
| 02 | (7500, 7500, 1400, 0.31) | 1.0020 | 0.9840 | Yes |
| 03 | (13500, 3000, 300, 0.42) | 0.9222 | 0.9180 | Yes |

The median relative L2 is 1.0020; only one of three events beats the zero
baseline. The network ranks the true source ahead of all training-source
candidates for each event, indicating some source-dependent discrimination,
but the first-ranked predictions still fail to match two records at their
original amplitudes. The PINN peak amplitudes are approximately 57%, 20%, and
20% of their FD peaks, respectively. Test-fitted amplitude gain reduces the
errors below one, but it uses the observations being evaluated and therefore
cannot be used as evidence that the forward model passed.

This result does **not** pass the predeclared forward feasibility gate
requiring every independent event to have relative L2 below 1.0. We therefore
do not run continuous location refinement on these events and do not claim
source-localization accuracy from this checkpoint. The three events have now
been evaluated and must not be described as untouched tests in later model
revisions. A future confirmatory evaluation needs newly generated, preregistered
events after the forward model passes on development data.

## Reproduction

```powershell
python .\examples\audit_forward_waveforms.py `
  .\outputs\overthrust_core20_train16_jointbatch_receiver_5hz_5000\conditioned_5Hz.pth `
  .\data\observations\independent_test\window_0to2s\independent_test_01_0to2s.npz `
  --output .\outputs\independent_test_forward_audit_01.json

python .\examples\audit_source_candidates.py `
  .\outputs\overthrust_core20_train16_jointbatch_receiver_5hz_5000\conditioned_5Hz.pth `
  .\data\observations\independent_test\window_0to2s\independent_test_01_0to2s.npz `
  --stride 2 --output .\outputs\independent_test_candidate_rank_01.json
```

Equivalent commands were run for events 02 and 03. Raw outputs are ignored by
Git under `outputs/` and use the filenames above.
