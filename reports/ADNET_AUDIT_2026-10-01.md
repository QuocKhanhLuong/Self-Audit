# ADNet Baseline Audit - 2026-10-01

Scope: audit the newly staged `baseline/ADNet/` tree for benchmark eligibility.
This report does not implement a runner or modify ADNet. It classifies the
baseline contract needed before ADNet can be compared against the shared
cardiac benchmark.

## Verdict

ADNet is benchmarkable only as a **few-shot labeled-support baseline**.

It is not eligible for the existing strict image-only anonymous-producer track
used by CUTS/DFC/STEGO/PiCIE, because inference requires a foreground support
mask to construct a class prototype. A fair ADNet track must allow frozen
support image/mask evidence while firewalling query GT until evaluation.

Recommended status:

| Track | Status | Reason |
|---|---|---|
| strict image-only anonymous producer | incompatible | ADNet cannot infer a target class without support masks. |
| few-shot labeled-support, query-GT-firewalled | acceptable after wrapper | Matches ADNet method shape while preserving query-GT isolation. |
| current upstream scripts as-is | not acceptable | Script/runtime and protocol issues remain unresolved. |

## Provenance

- Local path: `baseline/ADNet/`
- Nested upstream remote: `https://github.com/sha168/ADNet`
- Nested upstream HEAD observed locally: `c6bba85 Update README.md`
- License: MIT (`baseline/ADNet/LICENSE`)
- Paper: "Anomaly Detection-Inspired Few-Shot Medical Image Segmentation
  Through Self-Supervision With Supervoxels"

The ADNet tree is currently untracked by the parent repository. Any benchmark
result must bind both the parent repository identity and the nested ADNet source
identity, or vendor ADNet into the parent tree before publishing receipts.

## Method Contract

ADNet inference is class-wise and support-supervised:

- The inference entrypoint loops over non-background labels and calls
  `getSupport(label=...)` for each class (`baseline/ADNet/main_inference.py:73-90`).
- Support images and support foreground masks are passed into the model
  (`baseline/ADNet/main_inference.py:117-119`).
- The model extracts a foreground prototype by masked average pooling
  (`baseline/ADNet/models/fewshot_anom.py:60-65`,
  `baseline/ADNet/models/fewshot_anom.py:117-121`).
- The model applies a learned scalar threshold to anomaly scores
  (`baseline/ADNet/models/fewshot_anom.py:70-78`,
  `baseline/ADNet/models/fewshot_anom.py:179-185`).
- The upstream script saves one binary prediction per class as
  `image_<id>_<label>.pt` (`baseline/ADNet/main_inference.py:155-157`).

Therefore a valid benchmark wrapper must expose ADNet as:

```text
ADNet-FewShot-Support-v1
allowed producer evidence:
  query images
  frozen support images
  frozen support masks
  checkpoint
  class mapping
  preprocessing/grid metadata
forbidden producer evidence:
  query masks/GT
  query Dice/IoU
  query-label-derived support selection
  post-hoc class/tie policies chosen from query performance
```

## Data and Label Audit

Upstream ADNet expects the Ouyang-style preprocessed layout, not this repo's
Self-Audit ACDC layout:

- CMR images: `cmr_MR_normalized/image*`
- CHAOST2 images: `chaos_MR_T2_normalized/image*`
- Labels inferred by replacing `image_` with `label_`
- Supervoxels under `supervoxels_<n_sv>/super*` for training

Evidence:

- Test loader glob and support/query split:
  `baseline/ADNet/dataloading/datasets.py:17-30`
- Query labels read inside upstream inference dataset:
  `baseline/ADNet/dataloading/datasets.py:46-51`
- Support labels read inside `getSupport`:
  `baseline/ADNet/dataloading/datasets.py:85-100`
- Training supervoxel dependency:
  `baseline/ADNet/dataloading/datasets.py:126-138`

CMR label names are:

```text
0 BG
1 LV-MYO
2 LV-BP
3 RV
```

For the Self-Audit/ACDC semantic schema, the required mapping is:

```text
LV-MYO -> MYO
LV-BP  -> LV
RV     -> RV
BG     -> Background
```

This mapping must be explicit in the support manifest and in every result
receipt.

## Benchmark Contract Required

The ADNet benchmark must freeze support evidence before execution:

- `support_manifest_schema`: `adnet.support_manifest.v1`
- support record fields:
  - `support_id`
  - `dataset`
  - `case_id` / patient id
  - `class_name`
  - `canonical_class_id`
  - `image_path` or image locator
  - `image_sha256`
  - `mask_path` or mask locator
  - `mask_sha256`
  - `slice_selection_policy`
  - `preprocessing_contract`
- query record fields:
  - image-only locator and hash
  - patient/case identity
  - split membership
  - shared grid / preprocessing identity
- run receipt fields:
  - parent repository identity
  - ADNet source identity
  - checkpoint SHA-256
  - support manifest SHA-256
  - class mapping SHA-256
  - mode (`ADNet-2D` or `ADNet-3D`)
  - fold, seed, n-shot, EP protocol
  - device/runtime versions

Producer execution must stage query images and support assets in allowlisted
roots. Query masks must be inaccessible to producer code. Evaluation may open
query GT only after sealed predictions exist.

## Output Contract Required

Upstream ADNet emits class-wise binary masks. The benchmark needs a deterministic
semantic merge wrapper:

1. Run ADNet per foreground class.
2. Map each class-wise binary mask to canonical cardiac ids.
3. Merge into one semantic map.
4. Write validity and metadata receipts.

The merge policy is not present upstream. It must be defined before reporting
multiclass metrics. Acceptable default for implementation:

```text
if no class predicts foreground: BG
if exactly one class predicts foreground: that class
if multiple classes predict foreground: fail closed unless the wrapper records
a deterministic pre-declared tie rule
```

Fail-closed overlap handling is preferred for the first audit-grade run because
the current upstream saved `.pt` masks do not preserve per-class probability
scores.

## Runtime and Protocol Blockers

These issues must be fixed or wrapped around before any scientific run is
accepted:

1. Query GT leak in upstream inference dataset.
   - `TestDataset.__getitem__` reads label files and returns `sample["label"]`
     during inference (`baseline/ADNet/dataloading/datasets.py:46-64`).
   - Wrapper must split producer from evaluator.

2. Support selection is implicit and order-dependent.
   - Upstream selects the last volume in the test fold as support
     (`baseline/ADNet/dataloading/datasets.py:24-30`).
   - Benchmark must use an explicit frozen support manifest instead.

3. 3D script mismatch.
   - Shell scripts call `main_inference_3D.py`
     (`baseline/ADNet/scripts/test_cmr_3D.sh:11-18`), but the repo file is
     `main_inference_patches3D.py`.

4. 3D inference missing import.
   - `main_inference_patches3D.py` calls `random.seed` without importing
     `random` (`baseline/ADNet/main_inference_patches3D.py:39-42`).

5. CUDA-only execution path.
   - Entrypoints call `.cuda()` and `nn.DataParallel(...)` directly
     (`baseline/ADNet/main_inference.py:53-56`,
     `baseline/ADNet/main_inference_patches3D.py:52-55`).
   - Benchmark wrapper should make device explicit and record it.

6. Logger is not idempotent.
   - `set_logger` calls `os.makedirs(log_path)` without `exist_ok=True`
     (`baseline/ADNet/utils.py:35-37`).

7. String identity comparisons.
   - Inference uses `label_name is 'BG'`, which compiles with a warning and
     should be `==` (`baseline/ADNet/main_inference.py:79-83`,
     `baseline/ADNet/main_inference_patches3D.py:78-82`).

8. Supervoxel generation is not production-ready.
   - `generate_supervoxels.py` has hard-coded `<path_to_data>` and commented
     write output. Treat training provenance as unresolved unless a separate
     supervoxel-generation receipt is introduced.

9. Dependency/runtime vintage is pinned only by Dockerfile.
   - Dockerfile uses PyTorch 1.9.0 CUDA 10.2 and older dependencies, including
     SimpleITK 1.2.3 and scikit-image 0.14.2
     (`baseline/ADNet/Dockerfile:1-19`).

## Validation Already Performed

Local static checks:

```bash
python3 -m py_compile \
  baseline/ADNet/main_train.py \
  baseline/ADNet/main_train_3D.py \
  baseline/ADNet/main_inference.py \
  baseline/ADNet/main_inference_patches3D.py \
  baseline/ADNet/models/fewshot_anom.py \
  baseline/ADNet/models/fewshot_anom_3D.py \
  baseline/ADNet/dataloading/datasets.py \
  baseline/ADNet/dataloading/datasets_3D.py \
  baseline/ADNet/dataloading/dataset_specifics.py \
  baseline/ADNet/utils.py
```

Result: syntax compilation passed. Python emitted warnings for `label_name is
'BG'` in both inference entrypoints.

No runtime inference was attempted because the required preprocessed ADNet data,
support manifest, checkpoint, and runtime environment are not present in the
current checkout.

## Acceptance Checklist for Implementation Phase

Before ADNet results can be reported:

- [ ] Parent repo vendors or pins the nested ADNet source identity.
- [ ] A frozen support manifest exists and is hashed into every run.
- [ ] Query producer receives no query GT paths or query label arrays.
- [ ] ADNet wrapper can run 2D inference from query image + support assets only.
- [ ] Class-wise predictions are sealed before evaluation.
- [ ] Semantic merge policy is deterministic and predeclared.
- [ ] Query GT evaluator is a separate phase.
- [ ] Result receipts include checkpoint, support manifest, class mapping, code
      identity, seed, fold, EP protocol, and device/runtime evidence.
- [ ] At least one synthetic fixture proves producer GT firewall behavior.
- [ ] At least one smoke fixture proves class-wise output sealing and semantic
      merge receipt creation.

## Bottom Line

Proceed with ADNet only under a named few-shot support track. Do not describe it
as image-only. The scientific claim should be:

```text
ADNet was evaluated with frozen labeled support evidence and query-GT-firewalled
producer execution.
```

That claim is accurate to the method and compatible with this repo's benchmark
discipline once a dedicated wrapper and support manifest are implemented.
