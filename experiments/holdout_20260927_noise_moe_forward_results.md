# Noise-augmented MoE forward audit

The residual experts and source gate were fine-tuned for 1,000 updates with
synthetic `SNR=10 dB` receiver noise added to every training batch. The shared
base model remained frozen. Training took `5.9 s` on the development GPU.

| Evaluation target | Frozen base L2 | Noise-augmented MoE L2 |
|:--|--:|--:|
| Clean holdout records | **1.16474** | 1.16530 |
| Noisy holdout records (`SNR=10 dB`) | **1.15090** | 1.15141 |

The noise-augmented residual model was slightly worse in both evaluations. The
experiment therefore does not support a claim that receiver-noise augmentation
improves the source-conditioned forward operator. It remains a negative,
one-seed diagnostic over four independent events and two noise realizations;
no localization or uncertainty claim is made.
