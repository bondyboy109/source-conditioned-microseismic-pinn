# Objective-landscape diagnosis

The MoE forward model was evaluated on holdout event 01 while varying source
parameters without retraining. At the true source `(4200, 7200, 380, 0.18)`,
the normalized waveform loss was `1.1482`.

Holding the true spatial coordinates fixed and scanning origin time gave the
lowest loss `1.1390` at `t = 0.35 s`, rather than at the true `0.18 s`. The
time curve was also multimodal, with additional low values near `0.325 s` and
`0.475 s`.

Holding the true origin time fixed and scanning a local spatial grid found a
lower loss at `(5400, 7200, 240) m` (`1.0935`) than at the true source. Other
low-loss points occurred near `(3000, 9600, 380) m` and `(3000, 7200, 100) m`.

This shows that the failed localization is not only an optimizer issue. The
current waveform surrogate does not preserve a unique minimum at the true
source, even on an independent event. The next method change should therefore
add a physically anchored arrival/onset or travel-time term to the inversion
objective and test it first with fixed-source diagnostics before running a
multi-event localization benchmark. The present result is diagnostic only and
does not support a localization claim.
