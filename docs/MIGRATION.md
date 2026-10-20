# Migration — 2026-10-20

Before any deletion, the complete previous main was retained as
`archive/pre-semisupervised-20261020` at
`924265f7df604d3dca84dda49578b1a0cf28d821`.
No history is rewritten; other research branches are not removed.

This is an explicit supervision/architecture change authorized by the user's
request. Old no-GT results are not relabeled semi-supervised successes. The exact
D/U reference and original tests remain unchanged in the archive, not replaced
under the same paths with different mathematics.

Removed from the active tree: frozen-Q/CineMA bank trainers/adapters, legacy
coarse/detail policy, dormant guard/refiner code, obsolete prompts/runbooks,
old experiment entrypoints, copied report bundles and their old tests. Current
research/provenance remains recoverable via the archive and its pinned commits.
Existing split/checkpoint metadata is retained without changing its contents.

New canonical package: self_audit/. Entry: train.py. Checkpoint/data schemas are
v2. Historical --bank / --phase coarse commands fail rather than train a different
model silently. Common old script names remain error-only stubs.

No untracked dataset, run, checkpoint, credentials, dependency environment or
active GPU process is touched by Git publication. A live checkout may retain
old untracked directories after pull; check_repo stops there. Create a fresh
worktree instead of deleting those directories blindly.

```bash
git fetch origin archive/pre-semisupervised-20261020
git worktree add --detach ../Self-Audit-historical origin/archive/pre-semisupervised-20261020
```

The remote package is tested with manufactured CPU data. Real DINOv3 loading,
NIfTI patient preparation, GPU profile and patient evaluation are not claimed
completed in this delivery. Do not launch a million-step run as a smoke test.
