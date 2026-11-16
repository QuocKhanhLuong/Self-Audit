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

Launch-specific note: real profile exposed S3 `_qval_metrics` incorrectly
requesting U keys from VAL plus native-grid aggregation incompatibility. The
other code team is publishing its codebook.py/regression fix to main. Confirm
that fix exists in the fetched S2 boundary commit; retain the failed profile
receipt. S1 has no dependency on this evaluator and is authorized to train
while the fix is prepared. Do not rerun the profile concurrently with S1.
Remote launch receipt is /root/dataset/receipts/PG_LAUNCH_RECEIPT.json; local
report receipt is under this directory. Read LIVE_STATUS.json only as a dated
snapshot and always verify live state.

The S3 fix is now published: fd12e9111c199906443eb1ab64fb17bc103464c7.
Before any later recovery, verify this commit is an ancestor of the current
main. The existing boundary hook runs CPU checks after its fetch; it does not
run a second GPU profile. Do not report GPU recheck PASS until measured. The
finalization metric parity check remains strict at atol 1e-7 (OPEN-B1f in the
other-team review); diagnose a real failure if one occurs, do not silently
loosen it or rerun an already committed TEST transaction.

Latest diagnosed S2 blocker (2026-11-16 12:19:39 UTC): read
`S2_OOM_RECOVERY.md` and `S2_OOM_RECOVERY_RECEIPT.json`. Original S2 failed
at the first post-warmup second-strong forward. Reviewed fix `a0a8887`
accumulates stream gradients separately without changing the two-view recipe;
169 PG tests passed both locally and on server. Its single recorded resume
still failed with CUDA OOM in an individual U-stream forward. The exact
stream/update/patient was not recorded. Last durable checkpoint remains
update 800, SHA `09564256071f9811bc3e4521968284ad0b281da7258afda6c36c0d69b28c0138`,
with 30 committed epoch rows; completed S1 and config/data hashes are intact.
Both attempts are archived. Do not repeat that invocation or claim recovery
from software checks or temporary GPU activity. A subsequent recovery needs
a concrete reviewed mitigation and ownership coordination. The registered
common one-view adaptation is not yet applied or proven sufficient. Recheck
live state; this is a dated blocker, not a permanent prohibition on an
authorized tested repair. Keep the PG automation enabled until the existing
completion requirements are satisfied.

Subsequent recovery (2026-11-16 12:46 UTC): read
`S2_CHECKPOINT_MEMORY_RECOVERY.md` and its matching `_RECEIPT.json`.
Reviewed source `df646fc9f800b1e90f2adda963b21b19ec984a10` adds per-native-block
activation recomputation and precise OOM context without changing the recipe.
174 PG tests and repository checks passed locally and on the stopped server;
original update 800 strictly loaded on CPU. One supervised full-budget resume
reached verified durable update 850, with finite active-U losses and measured
7.35 GiB allocated / 11.26 GiB reserved peaks through that diagnostic.
Original 800, S1, data/config and prior failures remain preserved. The run is
progressing, not complete, and all-stage GPU fit is not established. Do not
reapply the fix, restart a healthy process or refresh its active checkout.
Recheck live state and retain the existing completion/automation requirements.
