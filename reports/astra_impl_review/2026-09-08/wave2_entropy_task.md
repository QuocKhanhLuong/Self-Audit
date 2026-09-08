# W2.2 — explicit entropy contracts (TASK SPEC, not implemented)

Execute only after Astra dispatch. AGY owns annotation_expert.py, necessary exports
and focused tests. Read docs.md and execution_plan.md W2.2. Preserve others' changes.
No training, commit/push, backbone/objective/state-machine changes.

## Objective

Current annotation_entropy guesses logits/probabilities from batch-wide min/max.
AnnotationExpert.forward passes logits. A constant shift or another batch member can
therefore change which formula is used. Preflight shift error was .2659366.

## Minimal change

- Explicit entropy_from_logits and entropy_from_probabilities APIs.
- Expert calls logits API. Probability API validates shape, finite values, range and
  per-pixel simplex with a stated numerical tolerance; do not silently renormalize bad input.
- Preserve expert normalized entropy convention; existing valid probability callers
  preserve their scale/formula. Do not change train_auditor._entropy merely to consolidate
  helpers: that path already receives probabilities and is outside the bug.
- Remove heuristic inference from annotation_entropy: either require an explicit input
  type for compatibility or retain a clearly documented logits-only alias, after tracing
  all actual callers. Never keep range-based dispatch hidden in a compatibility wrapper.
- Add explicit model/entropy semantic version constant for future lineage hashing;
  no extra parameter tensors and no checkpoint state_dict key changes.

## Tests / PASS

Constant logit shift invariance; batch-composition invariance; uniform logits entropy≈1;
one-hot and ordinary valid simplex outputs; invalid/NaN simplex rejected; finite gradients;
expert actually calls typed logits formula for values entirely inside [0,1].
Run focused -> all tests/ -> source compile with exact commands and outputs.
Report old checkpoints/tau may change predictions due this bug fix; no Dice improvement
claimed and no calibration compatibility assumed. New concise wave2_entropy_worker.md.
