# Evidence boundary

Candidate A implements teacher-guided selection of typed anatomical measurements
whose values are predicted by the student and decoded under a finite packet
budget. It does not implement direct teacher/student feature fusion.

Candidate A-3D is the native full-volume extension: local 3D REGION/BOUNDARY
queries, anisotropic physical distance and a MedNeXt-L adapter with a separate
boundary path. A Conv3d, a larger teacher or 3D geometry alone is not a novelty
claim. The A and A-3D tracks require matched teacher-free, global-only,
fixed/random, type-specific, precision/budget and capacity controls.

The encoder, query attention decoder, quantization/STE and supervised losses are
implementation choices, not individually claimed as novel. Novelty and efficacy
require comparison against teacher-free, fixed/random selection, type-specific,
supervision, precision/budget and capacity controls. Those recipes are defined in
[EXPERIMENTS.md](EXPERIMENTS.md). A weaker or inconclusive result remains valid.

Software tests establish contracts and failure handling. Synthetic loss curves,
Dice or packet sizes do not establish patient generalization, GPU efficiency,
SOTA, 96 Dice or clinical usefulness. Prior research reports remain available at
their recorded commits and must not be treated as trained-model evidence.
