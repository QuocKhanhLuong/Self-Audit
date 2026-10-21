# Runbook: v2 only

Use a fresh worktree and an appropriate installed official PyTorch CUDA wheel.
Do not update a checkout used by an active job, reset local files, or run git clean.

```bash
git fetch origin main
git worktree add --detach ../Self-Audit-semi-v2 origin/main
cd ../Self-Audit-semi-v2
python train.py --describe
python scripts/check_repo.py
python -m pip install -r requirements.txt
python -m pytest -q
```

## Data and annotation budget

Use your locked patient split. The parser supports explicit train/val/test lists,
existing *_patients fields/nested patient lists, or frozen locked_splits metadata;
conflicting representations fail. This reuses split metadata, NOT historical Q.
The old dataset copy had duplicate content in two VAL IDs. Prefer the explicitly
reviewed locked split for that copy; this code will not silently drop patients.

```bash
python prepare_acdc.py --root /ABS/ACDC/training \
  --split /ABS/locked_patient_split.json --labels 8 --seed 42 \
  --out /ABS/cache/L8_S42
```

This reads GT only for the selected eight TRAIN patients. U gets complete cine
images without GT, ED/ES selection or reference-derived ROI. `train.json` lists
only image paths for U and pins every cached payload. `PREPARATION.json` records
source hashes and actual mask-access patients. A failure leaves an incomplete new
cache for inspection; it is not success. Do not overwrite an existing cache.

Add `--evaluation` only when separately authorizing development reference
preparation. This makes `development.json`; the trainer never consumes it.
NIfTI parsing uses nibabel. The output grid is native; no resizing/foreground
resampling is hidden. The frame/cine exact voxel bridge must hold for GT alignment.

## Backbone source and weights

Copy `configs/backbone.example.json` to a local file and fill absolute source repo,
exact git source commit, local DINOv3 ConvNeXt-Tiny pretrained weights and SHA256.
Source must have no tracked modifications; weights must match. No downloads,
external terms acceptance, supervised segmentation weights or random fallbacks are
performed. Satisfy the pinned official source's dependencies explicitly.

```bash
BANK_JSON=/ABS/cache/L8_S42/train.json
DATA_SHA=$(python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$BANK_JSON")
python train.py --config configs/pilot.json --model-config /ABS/BACKBONE.json \
  --data "$BANK_JSON" --data-sha256 "$DATA_SHA" \
  --out /ABS/runs/semi_v2_pilot --device cuda
```

Review STARTUP, REPORT and steps.jsonl. There must be real supervised native
gradients, distinct supervised/audit patient IDs, correct label budget, genuine
weight provenance and explicit accepted/rejected transactions. A successful
software pilot is not a quality claim. `configs/full.json` uses decoder scope;
its JVP/runtime must be profiled before the expensive run.

```bash
python train.py --config configs/full.json --model-config /ABS/BACKBONE.json \
  --data "$BANK_JSON" --data-sha256 "$DATA_SHA" \
  --out /ABS/runs/semi_v2_full --device cuda --allow-full-run
```

No exact resume is implemented. The checkpoint contains states for audit and
inspection, but resuming is not silently approximated. Preserve incomplete runs.
The same local pinned backbone and source are currently required on reload.

## Frozen evaluation

Use the final checkpoint manifest SHA printed in REPORT.json. Evaluation hashes
all full-cine predictions before opening any reference mask arrays. This does
not make repeatedly used VAL an unseen confirmation set.

```bash
python evaluate.py --checkpoint /ABS/runs/semi_v2_full/final \
  --checkpoint-sha256 VERIFIED_MANIFEST_SHA \
  --data /ABS/cache/L8_S42/development.json --data-sha256 VERIFIED_DATA_SHA \
  --out /ABS/runs/semi_v2_eval --device cuda
```

Default output is dense raw segmentation. `--threshold` is an optional separately
reported gate, never a way to discard false negatives from evaluation. No weights,
class mapping or thresholds are fitted during evaluation.

```bash
python experiments/plan.py --out /ABS/plans/semi_v2 --steps 10000
```

The resulting MATRIX.json contains templates. Resolve and verify hashes, review
the annotation/compute budget, and execute selected commands yourself/Astra. The
planner has no automatic execution path.

## Native slice batches (declared experimental factor)

Optional `batch_size` defaults to 1 for existing v2 recipes. At larger batch sizes,
all samples for each role come from its scheduled patient, preserving native
spatial dimensions without resizing or padding. Supervised and audit patients
remain different. Audit Brier weights give each sampled slice equal mass within
that patient; CE uses all valid batch pixels, Dice uses the batch-wide present
classes, and pseudo candidates retain the shared accepted-pixel denominator.
This is a different batch protocol from the original single-slice setting. All
matched arms must use the same batch size, schedules and sampling policy.
Metadata records every sampled frame/slice. Larger batches do not increase the
number of distinct patients visited per transaction.

`scripts/profile_acdc_gpu.py` measures gradients and one decoder JVP on a repeated
real TRAIN slice, with zero optimizer/EMA updates. It is a memory diagnostic,
not a transaction throughput benchmark or patient efficacy result. A successful
32-step actual training pilot with checkpoint saving remains necessary. Full
transaction memory includes optimizer states, multiple class directions and
functional trials, and can exceed this diagnostic substantially.

For repeated label subsets, `prepare_acdc.py --image-cache /ABS/PRIOR_CACHE`
can reuse normalized images from a completed cache. Each reused image is compared
exactly against normalization recomputed from the raw cine and hashed. Masks are
prepared separately for the new budget/seed; no masks or ED/ES identities are
borrowed from another label subset. Keep the prior image cache immutable and
available for every dependent run. The first cache may use `--evaluation` to
include VAL images/references in a separate development manifest.

## Fixed training crops for the 24-GB run

`crop_size: 256` selects independent random coordinates using image geometry
only, after frame/slice sampling. Supervised/audit image and reference use the
same crop; U's EMA and strong views share a crop. Small images use centered edge
padding; labels use IGNORE=255. An image-only validity mask excludes padded U
pixels from the confidence gate and the shared accepted-pixel denominator,
including at a zero threshold. No foreground bounding box or U GT is consulted.
`crop_size: 0` preserves legacy native slices. VAL/TEST remain full native cines.

The crop/batch setting is a declared new experimental factor: use the same fixed
batch52/crop256 recipe in all matched arms. Previous native batch16 checkpoints
remain in their pinned worktree. Stopped runs retain checkpoints/logs and are not
claimed as resumed. Full-transaction VRAM probes are separate diagnostics: they
may load a pinned v2 checkpoint to exercise realistic four-class proposals, but
full experiments initialize afresh from the original pretrained model.

## Package and upload final checkpoints

After the four retained runs have finished, use the already configured rclone
remote; no authentication/remote setup or sharing change is performed:

```bash
python scripts/package_final_checkpoints.py --root /workspace/self-audit-v2
python scripts/upload_checkpoint_archive.py --root /workspace/self-audit-v2 \
  --drive-root KhanhDrive:Self-Audit-v2-artifacts
```

The packager creates a new timestamped ZIP containing the four step10000 final
student/teacher/optimizer checkpoint payloads, original hashed manifests,
pretrained DINO weights, pinned current-v2 source and official DINO Git bundle,
configs and compact evaluation receipts. It requires matching checkpoint/source
hashes, tests ZIP CRC, and writes SHA256/MD5 receipts. Raw patient images, masks,
full prediction arrays, credentials, raw logs, intermediate/pilot checkpoints and
incomplete runs are excluded. They are not deleted locally. Exact resume remains
unsupported; the restore note describes absolute paths in the original manifests.

The uploader checks the archive SHA256, uses immutable rclone copies, retries
quota failures, compares four uploaded files, verifies remote MD5/size, and uploads
EXPORT_RECEIPT.json. Drive view URLs use existing permissions; files are not made
public. ZIPs and large checkpoint artifacts remain outside Git.
