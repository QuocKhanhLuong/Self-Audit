# RTX 4070 training results — main 513ddda, batch 12

The run completed 50 epochs and 85,700 optimizer updates. Final foreground VAL
Dice is **3.1896%**, IoU **1.8036%**, recall **2.5163%**, and coverage **0.4960%**.
This is poor segmentation performance: **99.5040%** of voxels abstain as unknown.

## Run and model

- Source: `513ddda8cec3a166f13cfe68fe6379cab014a1a7`, with no scientific code modifications.
- Architecture: GuidanceStudent with a frozen 125,670,400-parameter CineMA MAE teacher and fixed guidance bank.
- Geometry: current-main `cardinal_v1`; this run does not use historical `world_left_v2` experimental code.
- Training: 50 epochs, batch 12, width 32, AdamW learning rate 0.0003, seed 42; teacher tile batch 1.
- Data: 80 TRAIN, 18 VAL, 50 official unlabeled TEST patients. Training and bank construction use no GT.
- Selection: fixed last epoch, no validation-based checkpoint selection.
- [W&B run k37v8mka](https://wandb.ai/nthne-hanoi-university-of-science-and-technology/self-audit-stable-guidance/runs/k37v8mka), verified finished with final metrics.

## Final validation metrics

Metrics cover 18 VAL patients / 36 labeled frames. Aggregation is frame mean,
then patient mean, then foreground-class mean (RV/MYO/LV).

| Metric | Result |
| --- | ---: |
| Foreground Dice | 3.1896% |
| Foreground IoU | 1.8036% |
| Foreground precision | 18.8578% |
| Foreground recall | 2.5163% |
| Coverage | 0.4960% |
| Unknown fraction | 99.5040% |
| RV Dice | 0.0063% |
| MYO Dice | 4.4446% |
| LV Dice | 5.1180% |

Undefined precision/recall entries are excluded from the macro mean; defined-patient
counts are preserved in `CLASS_METRICS.csv`. This differs from the zero-filled
precision convention in some historical reports. AUROC is unavailable because
this pipeline exports hard labels only. The official TEST cohort has exports,
not scored metrics, because its labels are unavailable.

## Training loss interpretation

| Quantity | Epoch 1 | Epoch 50 |
| --- | ---: | ---: |
| Feature loss | 0.361056 | 0.209630 |
| Semantic cross-entropy | 1.123137 | 1.085173 |
| Weighted target entropy | 1.046016 | 1.046319 |
| Estimated online KL (CE minus entropy) | 0.077121 | 0.038853 |

The TRAIN-only audit verified all 80 bank file hashes and replayed the exact
patient/slice RNG order, batch grouping, reliability weights, foreground gate and
all-step epoch averaging. Semantic updates reproduce exactly: 79,221 / 85,700.
The residual KL fell 49.62%; feature loss fell 41.94%. Mean KL decreased from
0.044009 (epochs 31–40) to 0.039950 (41–50), so improvement continued slowly.
This is an estimate from changing online training models, not fixed-checkpoint
validation KL. Soft-label entropy explains the nonzero CE floor; neither low KL
nor a plateau certifies correct segmentation or complete convergence.

## Batch handoff and VRAM probe

The original batch-8 run finished its 80-patient bank, then was interrupted during
epoch 1 for the requested VRAM optimization. Its last sampled update is step 825;
the exact final update count is unknown and no student checkpoint was saved.
The batch-12 run reused the immutable bank and initialized a fresh student with
seed 42. Only batch size changed in the recipe; all 50 epochs were executed.
The failed initial handoff receipt and successful retry are both preserved.

On the largest tested native TRAIN shape, the batch-12 microbenchmark reached
50.43 samples/s versus 49.13 for batch 8 (about +2.63%), with peak reserved memory
10.08 GiB. The probe had feature-active/semantic-inactive batches and is not an
end-to-end throughput or accuracy comparison. Completed full training recorded
about 8.75 GiB peak allocated memory. Larger batches changed the number of
optimizer steps per epoch, so this is not an equal-update comparison to batch 8.

## Evaluation status and provenance

The canonical pipeline completed training and froze both VAL and TEST predictions,
then exited with code 1 at strict VAL evaluation because of the known ACDC
frame/cine affine mismatch. The documented recovery verified exact voxel identity
for 14 affected frames without resampling, and evaluated the same frozen checkpoint,
predictions and thresholds. The worker final exit code is 0. Original `BLOCKED`
and tracker `failed` records are retained; `evaluation_recovery/RECOVERY_STATUS.json`
and `tracking/FINAL_WANDB_VERIFIED.json` record the completed recovery. Earlier
`READY_WAITING_FOR_4070` optimization receipts describe launch-time state only.

## Contents and integrity

- `completed_run/`: complete training/export/evaluation logs, loss history, effective split and recipe, preflight/test receipts, executed source, W&B receipts, and frozen-artifact manifests.
- `completed_run/analysis/semantic_convergence/`: executed entropy audit, report, and all 50 epoch rows.
- `interrupted_batch8/`: initial bank construction logs and interrupted-run/handoff evidence.
- `vram_probe/`: executed benchmark source, measurements and log.
- `summary.json`, `MAIN_METRICS.csv`, `CLASS_METRICS.csv`: machine-readable results.
- `publication_checks.json`: original source/hash/manifest checks.
- `MANIFEST.json`: per-file published hashes, original hashes and text transformations.

This publication contains text evidence. Checkpoints, bank arrays, NIfTI volumes,
source tarballs and W&B binary/cache files remain local; frozen manifests preserve
their identities. Absolute local paths are replaced by `${REPO_ROOT}`,
`${GUIDANCE_ENV_ROOT}`, `${HF_CACHE_ROOT}` and related placeholders; ANSI control
sequences and trailing log whitespace are stripped, and CSV line endings use LF.
Copied `FROZEN.json` files describe the original local
artifacts, so use `MANIFEST.json` to verify published copies. Archived scripts
record the executed code and require the original run layout and local binary
artifacts to run; placeholders are not automatically expanded.
