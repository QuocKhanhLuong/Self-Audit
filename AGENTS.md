# Current source of truth

Read `ACTIVE_PIPELINE.json`, `docs/CURRENT.md`, and `docs/RUNBOOK.md` first.
Use `python train.py`; do not revive legacy stable-guidance/v3/supervised runners.
The current implementation is `experiments/layer_guided_student_v1` imported
from `c4973553a743648a04c603c6b4634240933b8b80`.

No automatic architecture, pretrained, data or checkpoint fallback is allowed.
An explicit model identity is mandatory before optimizer updates. Respect all
bank/source/weight hashes. Preserve UNKNOWN=255; references do not enter training,
semantic mapping, seed selection or pseudo-label preparation. Exposed validation
is development evidence, not an unseen confirmation set.

Do not describe coarse-only training as learned native detail or self-learning.
Do not silently replace Q with Z or enable guard/refiner under the old schema.
For native-detail work use `docs/prompts/ASTRA_NEXT.md` and version the policy.

The immutable reference is a test oracle, not a training configuration. Keep
its published bytes and the six layer modules unchanged unless a new version
explicitly documents the mathematical/behavioral difference. Existing tests
cover original limitations as well as supported behavior.

Before delivery run `python scripts/check_repo.py` and `python -m pytest -q`.
Report what actually ran; synthetic CPU checks are not patient/GPU evidence.
Keep outputs outside inputs/frozen directories. Do not reset live worktrees,
kill other jobs, remove data, rewrite Git history, or accept external terms.
Retired material is accessible in the archive branch; consult it only for
an explicitly identified historical reproduction, not current training.
