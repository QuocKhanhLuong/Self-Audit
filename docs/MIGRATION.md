# Cleanup and migration — 2026-10-17

## Recoverability

Before cleanup, the main snapshot was retained as branch
`archive/pre-cleanup-20261017` at `92a60744efef85558d7c0226c84b7c49d47217ec`.
The imported new-model trainer was already published at
`c4973553a743648a04c603c6b4634240933b8b80` on
`codex/astra-student-training-20261016`.
No history is force-rewritten and no research branch is deleted.

Historical reproduction belongs in a separate worktree:

```bash
git fetch origin archive/pre-cleanup-20261017
git worktree add --detach ../Self-Audit-historical origin/archive/pre-cleanup-20261017
```

Do not cherry-pick old launchers into the new worktree. Do not update code under
an active server process. The cleanup only changes tracked repository content;
untracked datasets, local runs, credentials and GPU processes are untouched.

## Removed from active tree

Old small-student/self-audit/pseudolabel trainers and their scripts; unrelated
baseline implementations and copied external code; historical run_configs,
configs/environment variants; stale model docs and prompt chains; old report
bundles and W&B logs; scratch/agent runtime state and legacy CI workflows.
All of these remain recoverable from Git history and the archive branch.

Six common old command names remain as tiny error-only stubs. They contain no
trainer, import no old code and exit 2 even when passed --help. This is intentional:
old shell commands fail visibly instead of silently selecting a new algorithm.

## Retained deliberately

- Current new-model source, tests, source hashes, bank and checkpoint contracts.
- One immutable 19-file reference oracle and its materializer for exact parity.
- Current layer theory and tests, including guard/refiner utilities not yet enabled.
- The minimal historical io.py used to read existing banks/NIfTI consistently.
- Existing split/checkpoint metadata; the bank still controls the training split.

The internal experiment directory and the six reference layer modules retain
their paths/bytes. This prevents accidental source-identity changes merely for
cosmetic packaging. No attempt is made to reinterpret an old CNN checkpoint as
a checkpoint for the new architecture. No new clinical-quality result is claimed.

## What changed operationally

The supported entrypoint is now train.py. It checks model identity, rejects legacy
recipes, requires an explicit coarse phase, defaults to a 32-update pilot and
requires --allow-full-run above 128 updates or for an uncapped epoch budget.
The Astra-named launcher is only an alias to that implementation. Fresh docs,
AGENTS.md, layout tests and CI all refer to the same command.

Remaining work: verified official-weight GPU pilot; separately versioned native
image-detail supervision; Z comparison on matched evidence; then self-learning.
None of those is silently enabled by this cleanup.
