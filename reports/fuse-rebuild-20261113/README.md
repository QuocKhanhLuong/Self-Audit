# Candidate FUSE implementation and launch

Uses labels of all TRAIN80 patients; not a few-label result.

Implementation: isolated `fuse-rebuild` branch from 50a7441939b1d9d05611ff2dc34a2a9b2f6f5a28. Parallel Luna Fast workers, one independent Claude reviewer. Frozen A/A3D/Q3D modules and Q3D configs unchanged.

SOFTWARE_ONLY: 66 focused tests passed in 21.36s; check_repo PASS; native CPU smoke T/J/VAL3/export with all three cold-loads bitwise equal. Separate-process CPU campaign completed T/J/VAL3/EXPORT/TEST_ONCE with the TEST transaction COMMITTED. Exact output: pytest.txt, check_repo.txt, smoke.txt, CPU_CAMPAIGN.md. Full historical repository pytest NOT_RUN under the latest user amendment.

Independent Claude review: PASS for logic/formulas; five Low residuals documented in CLAUDE_REVIEW.md. No blocking findings. Resume after a crash near best-artifact selection requires verifying the selected artifact against the last committed validation; automatic retries are disabled.

PATIENT_DATA_INTEGRITY_CPU: all 98 TRAIN/VAL cached cine volumes equal normalization of the original NIfTI bit for bit (CACHE_RAW_EQUALITY.json). This is data integrity evidence, not efficacy.

P0 server inspection is read-only and historical. GPU profile, switch receipt and launch evidence will be added separately. Patient efficacy, complete VAL/TEST outcomes, deployment decision and inference timings: NOT_RUN at implementation commit.

## GPU profile and launch

P5 switched to new checkout `/root/Self-Audit-fuse-20261113`, implementation SHA `0813b9f6e18144c7a7b720fb37bcb0708e808e3a`. Old checkout, datasets and runs remain untouched. Old Q3D alarm was disabled, watchdog stopped, then campaign stopped. Last committed K8 V1-u checkpoint is J update 1600; 34 subsequent updates remain only in the historical log. The latest fast-switch instruction superseded waiting for another boundary. SWITCH_RECEIPT.json inventories SHA and preserved archive; three unfinished jobs are SUPERSEDED in this new receipt.

New data: TRAIN80, 160 original labeled ED/ES frames and 1,988 unlabeled cine frames; VAL18/36 frames and TEST50/100. SOURCE_RECEIPT.json, BUILD_RECEIPT.json and DATA_VALIDATION.json pin identities and provenance. Existing geometry and split are unchanged.

GPU profile: NVIDIA RTX PRO 4000 Blackwell, bf16, 20 real T and 20 real J updates. Largest native patient057 volume (8,512,428) is included in both streams.

| Stage | Peak allocated VRAM | Mean seconds/update |
|---|---:|---:|
| T | 8.38 GiB | 0.9408 |
| J | 12.44 GiB | 0.7229 |

No OOM/nonfinite loss. Measured compute/data-load projection is 3.70 hours for 8,000 T + 8,000 J updates; VAL and checkpoint I/O add overhead not measured by this profile. EPOCH_BUDGET_LOCK.json keeps T50/J100 before any production VAL outcome. GPU determinism is BEST_EFFORT_GPU_WARN_ONLY; no claim of bitwise GPU training reproducibility. Profile created no checkpoints; only its receipt/log remain.

Production run `/root/mount_data/runs/fuse-20261113` launched under Supervisor `fuse-campaign`. Automatic stage order T→J→VAL3→export→TEST3; validation each epoch. `fuse-watchdog` checks every 420 seconds; Luna alarm `0985130d-e9d1-4897-b0d8-2252dddc3291` every 600 seconds. Automatic retry limit zero; failures are reported for diagnosis rather than silently rerun. Visible Orca window **FUSE LIVE training**, server tmux **fuse-training**, shows per-update loss and per-epoch FG/RV/MYO/LV.

Real training is RUNNING, not completed. Final VAL18/TEST50 table, final deploy selection, patient inference timing and completed-campaign efficacy: NOT_RUN. Current observations are timestamped in LAUNCH_RECEIPT.json and can become stale.

First production VAL after teacher epoch 1 (update 160): FG 48.06%, RV 32.42%, MYO 49.38%, LV 62.38%. Preliminary training progress only; no efficacy conclusion or final teacher selection. Training advanced into epoch 2 after this evaluation.
