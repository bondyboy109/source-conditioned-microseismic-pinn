# Hybrid receiver-weight sweep on development events

This is the physically meaningful follow-up to the receiver-only scan. The
PDE, initial-condition, and boundary-condition terms were enabled. The four
runs used the same 5 Hz carrier, 16 training events, seed 42, 1,000 updates,
learning rate `1e-5`, batch sizes, sampling configuration, and initialization
from `outputs/overthrust_core20_train16_jointbatch_receiver_5hz_5000/conditioned_5Hz.pth`.
Only `receiver_loss_weight` changed. The three registered events at 250, 800,
and 2000 m remained evaluation-only.

| weight | 250 m raw L2 | 800 m raw L2 | 2000 m raw L2 | mean raw L2 | 250 m shift+gain | 800 m shift+gain | 2000 m shift+gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.03 | 0.7405 | 0.9832 | 0.9369 | 0.8869 | 0.6537 | 0.8429 | 0.8950 |
| 0.10 | 0.7555 | 0.9924 | 0.9355 | 0.8945 | 0.6600 | 0.8439 | 0.8909 |
| 0.30 | 0.7598 | 0.9924 | 0.9336 | 0.8953 | 0.6670 | 0.8464 | 0.8892 |
| 1.00 | 0.7616 | 0.9903 | 0.9342 | 0.8954 | 0.6685 | 0.8467 | 0.8895 |

The lowest mean raw error in this development scan occurs at weight 0.03,
but the differences are small and the best setting is not consistent across
events. All settings retain large waveform error, especially at 800 and
2000 m. The shift-and-gain columns are diagnostic only; they do not replace
the original forward error.

The result supports using 0.03 as a candidate for a later held-out test, but
it does not establish a final hyperparameter. Selection must be frozen before
running the independent test set. A stronger claim would require repeated
seeds and a larger development set.

Event-level JSON files are stored as
`receiver_weight_{0.03,0.1,0.3,1.0}_hybrid_development.json`.
