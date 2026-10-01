# Self-Audit v3: reviewed pseudo teacher and annotation workflow

Read [the current integrity review and runbook](pseudolabel_v3_review_runbook.md)
for commands, protocol changes, supervision limits and regression coverage.

The earlier `103ce77c` implementation was a research scaffold, not an experimentally
validated cardiac annotation system. The review fixes UNKNOWN loss, circular seed
supervision, patient-split isolation, native geometry, mask discovery and CI coverage.

## Offline path

Full-cine image-only training patients -> appearance + pairwise registration ->
anonymous prototypes -> raw image-only prior evidence -> partial semantic seed learning
-> conservative evidence/validity gate -> motion-aligned temporal rejection -> immutable
named pseudo-label export. Exporting development predictions does NOT train on them.

## Deployment annotator

Frozen TRAINING pseudo-labels -> small semantic encoder/head -> A0 -> optional shared
canonical Dynamic Window refinement. Compact/balanced/accurate use 0/1/3 internal
attention passes. Encoder computation is shared. An entropy/profile cap is an engineering
selector, not calibrated correctness or evidence that added refinement improves Dice.

## Important limits

- No manual scribble or manual segmentation-mask input is introduced.
- Seeds are still heuristic, not guaranteed high-precision anatomy.
- No evidence means exact UNKNOWN; zero foreground supervision stops student training.
- Cross-slice voting without correspondence is disabled. Pairwise temporal checking is
  not a proven full-cycle physiological model.
- End-to-end software tests use synthetic NIfTI, not real ACDC/M&Ms quality experiments.
- No ACDC >=0.91 result, GPU latency, UI or clinical utility claim is made.
- Schema 3 requires fresh output directories; old freezes are not silently upgraded.
