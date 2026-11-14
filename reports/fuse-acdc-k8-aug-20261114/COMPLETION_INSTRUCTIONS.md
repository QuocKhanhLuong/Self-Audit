# Completion and publication handoff

Authorized task: finish A-aug, publish results as soon as TEST commits.
Do not start or modify B/B0. Do not retry failed training/TEST automatically.

Server `vast-gpu`; new checkout `/root/Self-Audit-fuse-aug-20261114`;
new run `/root/mount_data/runs/fuse-acdc-k8-aug-20261114`.
Use `/venv/main/bin/python`; read `/etc/vast-agents-guide.md`.

Read `AUG_FLOW.json`, `RUN_STATE.json`, `STATE.json`, `MONITOR_STATUS.json`,
service status and current log. Healthy/progressing: no edits or user notice.
This run uses its own Supervisor instance. Inspect with
`supervisorctl -c /root/mount_data/runs/fuse-acdc-k8-aug-20261114/SUPERVISORD.conf status`.
Do not reread or modify the global Supervisor; archived Q3D log paths prevent
its global reread, and A-aug does not depend on it.
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
Check that the generated report contains all 12 old/new VAL/TEST mode rows,
all three Phase0 rows, both profile receipts and nonempty campaign elapsed
time. The report writer can finish with missing optional inputs; that is not
sufficient for publication. The legacy profile is a new matched control using
the old recipe, not an old recorded runtime. Full old-A elapsed time was not
saved in its committed receipts; report NOT_AVAILABLE rather than treating
the old TEST-only ledger interval as the campaign duration.

Commit only authorized report files to main, push without force and verify
remote SHA. Preserve any unrelated changes. Report completion exactly once,
then disable only the named A-aug completion automation from LAUNCH_RECEIPT.
