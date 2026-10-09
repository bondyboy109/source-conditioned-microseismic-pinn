# FMM noise sensitivity with the original threshold picker

Using the original 5%-of-peak threshold picker, the FMM arrival baseline is
strongly affected by synthetic noise. Across four independent events and two
realizations per SNR, mean location error was `3.13 km` at 20 dB, `7.17 km` at
10 dB, and `6.60 km` at 5 dB. No trace-selection failures occurred, but the
picked arrivals were often false early crossings.

This result motivated the robust MAD picker reported in
`holdout_20260927_noise_picker_results.md`; the original threshold protocol
should not be used for a noise-robust localization claim.
