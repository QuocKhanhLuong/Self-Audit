# Completion and publication handoff

Authorized task: finish A-aug, publish results as soon as TEST commits.
Do not start or modify B/B0. Do not retry failed training/TEST automatically.

Server `vast-gpu`; new checkout `/root/Self-Audit-fuse-aug-20261114`;
new run `/root/mount_data/runs/fuse-acdc-k8-aug-20261114`.
Use `/venv/main/bin/python`; read `/etc/vast-agents-guide.md`.

Read `AUG_FLOW.json`, `RUN_STATE.json`, `STATE.json`, `MONITOR_STATUS.json`,
service status and current log. Healthy/progressing: no edits or user notice.
Failure/OOM: preserve all artifacts, no retry, report exact exception and last
checkpoint. Do not turn an exited process into a completion claim.

When `PUBLISH_READY.json` exists and flow is COMPLETED, verify campaign completed
T/J/VAL3/EXPORT/TEST_ONCE and TEST_ONCE/TEST_TRANSACTION.json is COMMITTED, role
test, all three modes. Check export sidecar SHA against bytes, selected J SHA,
recipe identity vs EXPERIMENT_LOCK. Fetch `report/`, raw RESULT, RUN_STATE, STATE,
AUG_FLOW, PUBLISH_READY, TEST transaction, export sidecars, profile receipts,
and checkpoint SHA receipts into this report directory (do not commit weights).
Verify copied SHA against server. If reporter failed after committed TEST, fix
reporting only and regenerate from existing metrics; NEVER reopen TEST.

Require old/new VAL/TEST tables for teacher/student/fusion: FG, RV, MYO, LV Dice
and HD95; Phase0; timing; primary student A-aug-minus-A paired VAL Wilcoxon and
patient bootstrap 10000/seed42. TEST is descriptive, explicitly:
**TEST lần 2, sau khi TEST đã mở**. This is semi K8, not full-label.

Commit only authorized report files to main, push without force and verify
remote SHA. Preserve any unrelated changes. Report completion exactly once,
then disable only the named A-aug completion automation from LAUNCH_RECEIPT.
