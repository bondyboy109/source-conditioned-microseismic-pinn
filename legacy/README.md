# Legacy implementation

These scripts archive the earlier workflow for reproducibility and negative
control experiments. They are not the proposed source-conditioned method.

In particular, `source_inversion_legacy.py` combines a fixed-source/four-input
network with an average-velocity waveform shift. Its local network call does
not expose all spatial source coordinates to the waveform-loss gradient. Do not
use its reported localization output as evidence for the revised method.

The scripts may require additional optional packages such as Matplotlib, tqdm,
ObsPy, or segyio and external data described in the manuscript.
