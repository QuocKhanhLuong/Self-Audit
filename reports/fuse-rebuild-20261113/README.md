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

The 20-update profile itself had no OOM or nonfinite loss. Measured compute/data-load projection is 3.70 hours for 8,000 T + 8,000 J updates; VAL and checkpoint I/O add overhead not measured by this profile. EPOCH_BUDGET_LOCK.json keeps T50/J100 before any production VAL outcome. GPU determinism is BEST_EFFORT_GPU_WARN_ONLY; no claim of bitwise GPU training reproducibility. Profile created no checkpoints; only its receipt/log remain.

Production run `/root/mount_data/runs/fuse-20261113` launched under Supervisor `fuse-campaign`. Automatic stage order T→J→VAL3→export→TEST3; validation each epoch. `fuse-watchdog` checks every 420 seconds; Luna alarm `0985130d-e9d1-4897-b0d8-2252dddc3291` every 600 seconds. Automatic retry limit zero; failures are reported for diagnosis rather than silently rerun. Visible Orca window **FUSE LIVE training**, server tmux **fuse-training**, shows per-update loss and per-epoch FG/RV/MYO/LV.

The first production VAL after teacher epoch 1 (update 160) reported FG 48.06%, RV 32.42%, MYO 49.38%, LV 62.38%. This is a historical launch snapshot, not the final selection result.

## Production campaign completion and patient evidence

The real GPU campaign completed at 2026-11-13 10:46:31 UTC. `RUN_STATE.json` reports all stages complete (`T`, `J`, `VAL3`, `EXPORT`, `TEST_ONCE`), `error: null`, and a committed one-shot test transaction on attempt 1. Validation coverage is complete: all 50 T epochs and all 100 J epochs have RV/MYO/LV metrics; no nonfinite training-loss rows were found.

This is patient-data execution using the complete TRAIN80 labeled split, not a few-label run: 80 training patients, 160 labeled ED/ES frames, and 1,988 image-only cine frames; VAL18 has 36 labeled frames and TEST50 has 100. T used 8,000 labeled-frame exposures across 50 epochs with physical batch 2 after exposure 1,100, totaling 4,558 optimizer steps. J used 8,000 updates across 100 epochs with its locked 1L+1U pairing.

| Split / model | Patients | FG Dice | RV Dice | MYO Dice | LV Dice |
|---|---:|---:|---:|---:|---:|
| VAL18 teacher | 18 | 0.906321 | 0.904167 | 0.884625 | 0.930170 |
| VAL18 student | 18 | 0.897585 | 0.891818 | 0.878702 | 0.922236 |
| VAL18 fusion | 18 | 0.909016 | 0.906451 | 0.887407 | 0.933190 |
| TEST50 teacher | 50 | 0.906247 | 0.898215 | 0.887557 | 0.932970 |
| TEST50 student | 50 | 0.907701 | 0.900933 | 0.887883 | 0.934288 |
| TEST50 fusion | 50 | 0.908257 | 0.899690 | 0.892713 | 0.932367 |

The predeclared VAL18 rule selected `fusion.pt`: student FG Dice (0.897585) was 0.011431 below fusion (0.909016), exceeding the 0.01 margin for choosing the student. TEST50 was used only for the committed one-shot evaluation, not selection. The selected export SHA-256 is `086c80a0eb6162444199b368862db2a7ca45112f9ecd51e4f3ef6f15f0365869`.

The run logs retain the initial resume error (`TypeError: RNG state must be a torch.ByteTensor`), fixed on pinned source `5f2485d03517e0dbfa44f479c9fd61b571aa1ec9`, and a T allocator OOM warning at exposure 1,120 (478,150,656-byte request; 349,831,168 bytes free). T subsequently completed all 8,000 exposures. The final campaign state has no current error. The 420-second `fuse-watchdog` is stopped, and the 10-minute Orca automation `0985130d-e9d1-4897-b0d8-2252dddc3291` is disabled.

These are cohort metrics from the recorded ACDC patient splits, not evidence of clinical efficacy or external generalization. Patient inference timing remains NOT_RUN. Full metrics, provenance hashes, transaction status, export hashes, and incident details are recorded in [FINAL_EVIDENCE.json](FINAL_EVIDENCE.json).
