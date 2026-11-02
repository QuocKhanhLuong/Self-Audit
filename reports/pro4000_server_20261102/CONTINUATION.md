# Completed-run recovery and joint profile repair

The original queue is retained at
`/workspace/selfaudit-20261101/campaign_20261102`. S0 and B8 each finished
10,000 training steps, but Drive quota failures left both `UPLOAD_PENDING`.
Neither needs another training run. Joint stopped after its discarded seven-step
profile because the runner incorrectly required `STOPPED_BOUNDARY`: the trainer
correctly reports `COMPLETED` when the profile's shortened budget is exhausted.

The repaired profile accepts completion only at the exact requested profile
budget and checks complete step-log coverage. It never imports profile weights
into the 10,000-step experiment. A real canonical-trainer fixture exercises this
boundary, alongside rejection tests for inconsistent reports.

`scripts/retry_completed_s0_backup.py` is a backup/evaluation-only recovery entry
point. It requires an already completed 10,000-step S0, the original source and
lock, and an explicitly pinned final manifest. It contains no training call.
The original source remains unchanged. Executed metadata-only evaluator and
backup-status repairs, original hashes and final weights identity are recorded
under the existing run's `evaluation_repair/`. The backup status repair prevents
duplicate `best_step`/`best_macro4` arguments after a post-TEST upload retry.

S0 recovery command on the server (only while its status is `UPLOAD_PENDING`):

```bash
RCLONE_CONFIG=/root/.config/selfaudit-20261101/rclone.conf \
/venv/selfaudit-pro4000/bin/python \
  /workspace/Self-Audit-continue-20261102/scripts/retry_completed_s0_backup.py \
  --source /workspace/Self-Audit-pro4000-20261101 \
  --out /workspace/selfaudit-20261101/runs/s0_acdc_l8_b32_seed42 \
  --lock-sha256 e82f2740ce548924e4ab1143ba1ad1f8c13df29d1864bd022621675bce596f49 \
  --final-manifest-sha256 b9263aa8b4fcbc9511d40d5b377f7b2d1044bbf2c1c2ba942c2020a618c50d39 \
  --rclone-destination KhanhDrive:Self-Audit-v2-artifacts --device cuda
```

Do not run that command separately while the continuation queue owns it. The new
queue uses a new state directory, pins the completed B8 report/final payload
before invoking its existing `--resume` backup/evaluation path, and has no S0 or
B8 training stage. Joint remains independent of teacher backup/quality failures.
S1/S2 remain conditional on verified backup, raw-student identity, the locked
Macro4 gate and the exact TRAIN-U cache. Historical receipts, failed profiles and
all local weights are retained. No recipe, label budget or quality gate changes.

Live state and logs, after launch:

```bash
cat /workspace/selfaudit-20261101/campaign_continue_20261102/state/STATE.json
tmux select-window -t ssh_tmux:selfaudit-continue-live
```

Software tests are not evidence of patient quality, successful backup, or a
completed experiment. Actual launch and validation receipts are separate.
