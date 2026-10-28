# Self-Audit real-patient PRO4000 execution

**Decision: NO_GO.** The declared joint/guard/U query gates all failed under verified fresh equal-time compute. This is one pilot seed, not a clinical conclusion.

## Dataset and provenance

| Dataset | Raw files | NIfTI | Patients | Image/mask pairs | Integrity |
|---|---|---|---|---|---|
| ACDC | 1053 | 750 | 150 | 300 | PASS |
| M&Ms | 692 | 690 | 345 | 345 | PASS |

ACDC original data: `data/ACDC/`; M&Ms original data: `data/MnM/`; immutable downloaded ZIPs: `data/archives/MnM/`. Remote paths were discovered with `rclone lsd KhanhDrive:` and verified by checksum/CRC. Original files, archive names and directory structures were preserved.

ACDC locked TRAIN80/VAL18/TEST50 split: L8, U72, TEST50 has 100 labeled frames. Two image-duplicate training IDs (055, 076) were already excluded by the reviewed split. M&Ms original TRAIN175/VAL34/TEST136 split is unchanged; eight duplicate-image IDs were quarantined only in derived manifests, yielding active TRAIN170/VAL32/TEST135. Generalization uses VAL32. No M&Ms data entered ACDC training. TRAIN-U manifests have no mask paths and no U GT reads; TEST/VAL GT was opened only by the offline scorer after prediction export.

M&Ms label mapping: original 1→LV(3), 2→MYO(2), 3→RV(1). All 690 NIfTI files have zero temporal spacing; it is treated as unknown and no physical time is invented. Image/mask shape, affine, spatial zoom, orientation and class checks passed. Separate loading/preprocessing smoke tests passed for both datasets.

Training manifest SHA256: `e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`. ACDC TEST50 manifest SHA256: `8f89d631bc67f7eccef94740866513fe6f794872e7cb6e61302b1663e17e8f9e`. M&Ms prepared train manifest SHA256: `a7fc92227c555d0de3895b5b37154e49bd3de5b44d26958bde34863d86dfaae3`.

DINOv3 ConvNeXt-Tiny source commit `6876159a11b4df116f30f667f8c9888617df0751`; authorized backup weight SHA256 `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`. This is self-supervised LVD1689M pretraining; no historical segmentation teacher or random substitute was used. Independent upstream full-weight-hash certification is unavailable.

## Execution and hardware

Base commit `93f7c7a1f9606a68955f849dd580b90462ad329e`; executable overlay `SOURCE_PATCH_B128_FINAL.patch` SHA256 `aedb2cd50c3a59a78be0da30e973de22894f9ccee7ed08613d515166094dc561`. No model architecture or canonical engine changes. PyTorch 2.14.1+cu132 on RTX PRO 4000 Blackwell 24GB; reuse `/venv/main`. FP32, no AMP. Batch 128 in all arms; full_ema uses four microbatches of up to 40 with one globally normalized optimizer step and one EMA step. Shared support-only warm start: 512 steps/27.37s, manifest SHA256 `2b5cb5961958eab8aeff2cf852625cf3c53940fda09e176a627dafe4ec8ca2fb`.

Resource sampling during core B128 execution: 1145 samples, GPU utilization mean 51.7%, median 66.0%, sampled max device memory 20980 MiB; cgroup RAM peak 79.9 GiB of 120.7 GiB. Arm-level PyTorch peak reserved VRAM was 20616 MiB for head arms and 19770 MiB for full_ema. GPU samples include startup/extraction/IO and therefore are lower than active-kernel utilization.

Measured phase sums: equal-update 3874.8s; fresh equal-time 6027.0s; shared warm 27.4s once. Native ACDC export sums: equal-update 548.9s, equal-time 547.5s. These are measured costs, not a sum claimed as a single uninterrupted job wall clock. Fresh equal-time phases span 600.43–607.80s, spread 1.23% (<5%). Equal-update phase times differ and are not equal-compute evidence.

## ACDC TEST50 per-arm native patient Dice

| Regime | Arm | Steps | RV | MYO | LV | FG mean | Phase s | Pseudo accepted |
|---|---|---|---|---|---|---|---|---|
| equal-update | scalar_brier | 128 | 0.000 | 0.000 | 0.000 | 0.000 | 587.6 | 0 |
| fresh equal-time | scalar_brier | 131 | 0.000 | 0.000 | 0.000 | 0.000 | 602.2 | 0 |
| equal-update | joint_brier | 128 | 0.000 | 0.000 | 0.000 | 0.000 | 574.9 | 0 |
| fresh equal-time | joint_brier | 130 | 0.000 | 0.000 | 0.000 | 0.000 | 603.4 | 0 |
| equal-update | scalar_dice | 128 | 0.000 | 0.000 | 0.000 | 0.000 | 582.9 | 0 |
| fresh equal-time | scalar_dice | 131 | 0.000 | 0.000 | 0.000 | 0.000 | 604.2 | 0 |
| equal-update | joint_dice | 128 | 0.000 | 0.000 | 0.000 | 0.000 | 587.9 | 0 |
| fresh equal-time | joint_dice | 132 | 0.000 | 0.000 | 0.000 | 0.000 | 604.4 | 0 |
| equal-update | frozen | 128 | 0.000 | 0.000 | 0.000 | 0.000 | 175.2 | 0 |
| fresh equal-time | frozen | 461 | 0.000 | 0.000 | 0.000 | 0.000 | 601.2 | 0 |
| equal-update | support_only | 128 | 0.000 | 0.000 | 0.000 | 0.000 | 176.7 | 0 |
| fresh equal-time | support_only | 474 | 0.000 | 0.000 | 0.000 | 0.000 | 600.8 | 0 |
| equal-update | plain_head_u | 128 | 0.000 | 0.000 | 0.000 | 0.000 | 245.4 | 128 |
| fresh equal-time | plain_head_u | 326 | 0.000 | 0.000 | 0.000 | 0.000 | 601.9 | 326 |
| equal-update | extra_supervised | 128 | 0.000 | 0.000 | 0.000 | 0.000 | 174.0 | 0 |
| fresh equal-time | extra_supervised | 470 | 0.000 | 0.000 | 0.000 | 0.000 | 600.6 | 0 |
| equal-update | smaller_lr | 128 | 0.000 | 0.000 | 0.000 | 0.000 | 245.9 | 128 |
| fresh equal-time | smaller_lr | 324 | 0.000 | 0.000 | 0.000 | 0.000 | 600.4 | 324 |
| equal-update | full_ema | 128 | 0.417 | 0.408 | 0.619 | 0.481 | 524.1 | 128 |
| fresh equal-time | full_ema | 151 | 0.504 | 0.523 | 0.699 | 0.575 | 607.8 | 151 |

All 20 ACDC evaluations scored 50/50 patients, 100/100 annotated frames, with prediction and scoring coverage 1.0. Native predictions and manifest/checkpoint hashes are in `runs/pro4000/evaluation_*/` and `evaluation_index_*.json`; patient-level Dice is in `PATIENT_METRICS.csv`.

## Failure analysis

All four guarded arms accepted zero pseudo transactions in both regimes. Each head-only arm predicted at most one foreground voxel across all 50 ACDC query patients and produced zero RV/MYO/LV Dice; extra_supervised at equal-time predicted zero foreground voxels. This supports a near-total foreground-collapse diagnosis for the fixed-encoder/head-only pilot under this initialization and budget; the reported scores do not isolate which individual component caused it.

The stronger full_ema executable comparator trains encoder/body despite frozen-encoder wording in the runbook. It was the only nonzero ACDC predictor: equal-update FG 0.481, equal-time FG 0.575. For equal-time full_ema, RV/MYO/LV are 0.504/0.523/0.699. The worst five patient foreground means are recorded below; per-patient and per-class details remain in the CSV.

| Patient | FG Dice |
|---|---|
| patient124 | 0.325 |
| patient107 | 0.345 |
| patient131 | 0.346 |
| patient125 | 0.413 |
| patient130 | 0.418 |

## M&Ms VAL32 external generalization

| Arm | RV | MYO | LV | FG mean | Patients |
|---|---|---|---|---|---|
| scalar_dice | 0.000 | 0.000 | 0.000 | 0.000 | 32 |
| joint_dice | 0.000 | 0.000 | 0.000 | 0.000 | 32 |
| extra_supervised | 0.000 | 0.000 | 0.000 | 0.000 | 32 |
| smaller_lr | 0.000 | 0.000 | 0.000 | 0.000 | 32 |
| full_ema | 0.388 | 0.363 | 0.480 | 0.410 | 32 |

The four declared primary arms score 0 foreground Dice on VAL32. The full_ema row is a supplemental separate-dataset comparator, exported and scored with the same verified adapter; it does not alter the ACDC GO/NO-GO gate. M&Ms original TEST136 was not scored.

## Decision and limits

Offline equal-time gate: **NO_GO**; joint>scalar=False, guarded Dice>smaller LR=False, U>extra supervised=False. The equal-update gate is `INCONCLUSIVE_NOT_RUN` by design because its actual compute was unmatched.

Single label seed and one run per arm; no uncertainty interval. ACDC Drive TEST50 was previously used in this repository and is not a newly unseen confirmatory cohort; challenge provenance is not independently certified. M&Ms VAL32 is a verified but nonofficial filtered subset after duplicate quarantine. No M&Ms retraining or pooling. The implementation discrepancy in full_ema makes it a stronger comparator. W&B credentials were unavailable; tmux log, JSONL steps, utilization CSV, checkpoint manifests and native prediction receipts provide the audit trail.

## Reproduction and paths

Run `bash reports/automatic_pilot_pro4000/execute_B128.sh` from `/workspace/Self-Audit` in the existing `/venv/main` only when creating a new output tree; the original output directories are immutable. Per-arm config overrides are in `run_stage.py`, model source/weight hashes in `model.json`, numeric backend in `numeric_backend.json`, data checks in `acdc_integrity.json` and `mnm_integrity.json`. Checkpoints and native predictions: `/workspace/Self-Audit/runs/pro4000/`. Reports: `/workspace/Self-Audit/reports/automatic_pilot_pro4000/`. Tmux window: `ssh_tmux:self-audit-pilot`. Raw patient data and checkpoint binaries were not pushed to Git.

Post-run full test suite: `535 passed, 1 warning in 128.48s (0:02:08)`.

Independent output audit: PASS for 20 checkpoint manifests/payloads, 25 metric/frozen receipts and 1160 native prediction files. See `OUTPUT_AUDIT.json`.
