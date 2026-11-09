# Evidence boundary

Candidate A implements teacher-guided selection of typed anatomical measurements
whose values are predicted by the student and decoded under a finite packet
budget. It does not implement direct teacher/student feature fusion.

The encoder, query attention decoder, quantization/STE and supervised losses are
implementation choices, not individually claimed as novel. Novelty and efficacy
require comparison against teacher-free, fixed/random selection, type-specific,
supervision, precision/budget and capacity controls. Those recipes are defined in
[EXPERIMENTS.md](EXPERIMENTS.md). A weaker or inconclusive result remains valid.

Software tests establish contracts and failure handling. Synthetic loss curves,
Dice or packet sizes do not establish patient generalization, GPU efficiency,
SOTA, 96 Dice or clinical usefulness. Prior research reports remain available at
their recorded commits and must not be treated as trained-model evidence.
