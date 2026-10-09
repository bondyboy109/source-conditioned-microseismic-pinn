# Source-conditioned low-rank noise sensitivity

The operator, arrival picker, search grid, refinement settings, and hybrid weight were frozen from the clean confirmation protocol.

| SNR (dB) | Objective | n | Hybrid available | Mean eloc (m) | Median eloc (m) | Std eloc (m) | Mean et (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | Waveform only | 24 | 100% | 842.5582344737969 | 741.398938587189 | 511.238370869935 | 0.09655040331805746 |
| 20 | Waveform + FMM arrival | 18 | 75% | 781.1479662216755 | 692.9276936591476 | 286.7220934928369 | 0.1610784207781156 |
| 10 | Waveform only | 24 | 100% | 860.7548110715705 | 737.413555806145 | 519.7470558960453 | 0.11401229631155729 |
| 10 | Waveform + FMM arrival | 15 | 62.5% | 771.7421972265522 | 667.8400439453682 | 355.962531947302 | 0.19249589244524637 |
| 5 | Waveform only | 24 | 100% | 853.8133890100997 | 720.1092577859745 | 533.3641303781461 | 0.1063936607291301 |
| 5 | Waveform + FMM arrival | 8 | 33.3% | 1011.0062546591686 | 1045.0822999955979 | 381.5491622639226 | 0.2009984478354454 |

On the paired subset where the FMM objective was available, it reduced spatial error in 7/18 (38.9%), 8/15 (53.3%), and 4/8 (50.0%) combinations at 20, 10, and 5 dB, respectively. These paired fractions are reported because the arrival picker removes different events at each SNR.

Synthetic confirmation records, one frozen operator and one receiver geometry; this is a noise-sensitivity test rather than field-data validation.
