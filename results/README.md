# Current neural ablation

`final_neural_ablation.json` is a derived view containing waveform-only and
hybrid. It retains every event/trial, the gate failures, source order and
statistical denominators. The complete original JSON records and their hashes
are in `../archive/supporting-research-records.zip`.

Waveform-only uses the same traveltime-featured MoE and zero arrival-loss
weight. Hybrid adds the registered arrival constraint. The primary comparison
is within this frozen neural formulation. The separate transfer and M=5
carrier records are supporting audits, not extra confirmation of the final MoE.
