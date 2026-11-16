# PG K8 monitoring and authorized repair

Read this report and the latest launch receipt before acting. Scope only the
new PG run `/root/mount_data/runs/pg-k8-20261116`, checkout
`/root/Self-Audit-pg-20261116`, services `pg-campaign` and this task's download /
preparation helpers. Read `/etc/vast-agents-guide.md` after SSH reconnect.
The user authorizes code-error diagnosis, a tested narrow fix and prompt
resume. They also authorize picking up the other team's main changes at the
S1/S2 stage boundary. Do not change a healthy running stage's code or inputs.

Each check: verify SSH host identity, queue state, exact active process,
recent stage log/diagnostic/checkpoint changes, GPU, disk and receipts. Healthy
and progressing: no edits and no repeated user notice. Never launch a second
GPU writer. A process absent or errored is not proof of training completion.

On error, inspect the traceback and last valid checkpoint first. Fix only the
responsible PG/setup code, run the corresponding focused test, commit/push
with exact paths after fetching other-team work, then deploy at a safe stopped
boundary and invoke the queue with a concrete `--resume-failed` reason. Keep
attempt history, checkpoints, original errors and all data. Do not repeat an
identical failed invocation without a fix. Do not alter architecture, split,
labels, budget, thresholds or use TEST for decisions. OOM permits only the
common adjustment explicitly registered in the PG request, across every arm;
otherwise report the measured blocker.

Before resuming S1, inspect its actual resume support: the original base code
stores an encoder artifact, not enough projector state for an exact restart.
Do not claim exact mid-S1 resume without validating the newer implementation.

Order is shared S1 → S2 → S3 → A4 → A3 → A2 → A1 → finalizer. Newer code may
provide a campaign queue; preserve this ordering and shared artifact SHAs.
Never repeat a committed final TEST transaction. On completion, verify all
four artifacts and receipts, gather reports, commit/push results without
weights unless separately requested, verify remote SHA, notify once and
disable only this PG automation. Leave unrelated historical automations alone.

Current M&Ms scope is download/integrity/geometry preparation only; do not
start B/B0 or any M&Ms training. Historical A/A3D/Q3D/FUSE source is frozen.
