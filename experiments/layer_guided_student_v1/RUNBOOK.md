# Layer-guided student runbook

This is opt-in scaffolding for exact software reproduction and local preflight.
Existing baseline/training entrypoints are unchanged. No patient training,
download, push or merge is part of these commands. The task's execution boundary
is specified in [the exact prompt](../../docs/prompts/ASTRA_LAYER_UPGRADE_EXACT.md).

## Environment and verification

Recorded local environment: macOS arm64, Python 3.11.16, torch 2.14.0,
torchvision 0.29.0, NumPy 2.4.6, pytest 9.1.1, CPU only. These are observed
versions, not a promise of equivalent results on other runtimes. Use the exact
interpreter in the receipts to reproduce this run; no dependencies are downloaded
by the verifier.

From the worktree root:

```bash
repo="$PWD"
experiment="$repo/experiments/layer_guided_student_v1"
python_bin=/Users/alvinluong/miniforge3/bin/python

rtk proxy "$python_bin" scripts/materialize_layer_upgrade_reference.py \
  --out /private/tmp/layer-reference-NEW --run-checks
rtk proxy "$python_bin" "$experiment/verify_port.py" \
  --out /private/tmp/layer-port-NEW
```

Replace `NEW` with a fresh name: the scripts refuse existing output directories.
`verify_port.py` first checks all 19 archive files and the verbatim port files,
then checks the protected historical paths against the reference commit. It runs
the entire experimental test suite, full random ConvNeXt forward/backward and
operator probes in the fresh receipt directory. See `PARITY_REPORT.json` for the
exact executed argv, hashes, counts, device and omissions.

For direct integration imports, **both** the experiment and repository `src`
must be on `PYTHONPATH`. The bank adapter reuses the historical read-only IO
validator; importing from the experiment directory alone is insufficient.

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$experiment:$repo/src" \
  "$python_bin" -c 'from integration import bank, artifacts, geometry, steps'
```

`tests/test_entrypoints.py` runs this import from outside the checkout without
pytest's path injection, and checks that missing approval fails before creating
preflight output. Integration code and new tests use Ruff F/I checks; the copied
reference files are excluded from formatting to preserve their exact hashes.

## Approved backbone preflight

The checked-in `recipe.json` fixes `layer_guided_convnext_tiny_v1`, seed 42,
detail bound 1.0, per-cine normalization, guard off, proposal-only refiner and
semantic admission off. Unknown keys or incompatible recipes fail.

Supply a separately reviewed local JSON receipt with these fields:

```json
{
  "schema": "layer_guided.backbone_approval.v1",
  "approved": true,
  "model": "dinov3_convnext_tiny",
  "lineage": "official_self_supervised_dinov3",
  "repo": "/absolute/path/to/reviewed/dinov3",
  "weights": "/absolute/path/to/reviewed/weights.pth",
  "weights_sha256": "REVIEWED_64_HEX_DIGEST",
  "source_commit": "REVIEWED_40_HEX_COMMIT",
  "review_record": "external review record identifier"
}
```

This example is not an approval. Bind the actual receipt to its independently
reviewed SHA-256. The loader requires existing local weights, matching digest,
a clean source checkout at the reviewed commit and the official source origin.
It calls `torch.hub.load(source="local", pretrained=True, weights=...)` only
after these checks. Keep weights outside that source checkout to avoid making
it dirty. Missing inputs yield `BLOCKED_PRETRAINED`; there is no random fallback.
The trust boundary is a reviewed receipt, not cryptographic proof of ancestry.

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$experiment:$repo/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$python_bin" -m integration.cli \
  --recipe "$experiment/recipe.json" \
  --approval /absolute/path/approval.json \
  --expected-approval-sha256 REVIEWED_RECEIPT_SHA256 \
  --out /private/tmp/layer-approved-preflight-NEW --device cpu
```

This command was **not run with official weights** in this delivery. It executes
one manufactured `1x3x65x97` forward/backward, no optimizer step, and records
outputs, source/backbone digests, elapsed time and process peak RSS. A later
authorized `--device cuda` run also records peak allocated CUDA memory and fails
if CUDA is unavailable. The raw stage API and local weight argument were checked
against upstream [ConvNeXt source](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/convnext.py)
and [hub factories](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py);
this source inspection is not runtime validation of a particular release.

## Images, registration and permitted supervision

`normalize_cine(image_tzyx, image_sha256)` applies a single image-only min/max
transform to all frames and spatial slices of one cine. `slice_contexts` returns
`[previous_z, center_z, next_z]`, replicating endpoints. Each input is float
`B x 3 x H x W` in `[0,1]`; no mask or teacher tensor is an inference argument.

`load_training_context(bank, patient_id, indices, expected_bank_sha256=...)`
reads a verified frozen TRAIN bank using an externally locked `FROZEN.json`
digest. It rejects VAL/TEST overlap, unknown producer hashes, changed manifests
and mismatched native dimensions. Only image and 64-channel features are exposed;
historical Q, labels and reliability are not converted into conditional Z mass.
The reader verifies the historical bank as a whole; those stored arrays may be
loaded/validated by the shared IO helper but are not returned as supervision.

The source-side `TeacherRegistration` binds image digest, native/teacher grids,
affine, source manifest, producer hashes, `TZYXC` axis order and full-native-FOV
pixel-edge mapping. `GridGeometry` compares destination geometry to that receipt
and refuses implicit flips, crops or affine resampling. The historical adapter
pins `teacher.py`, `pipeline.py` and `io.py` from the reviewed `main@4d58f3a`
producer. Self-consistent forged metadata is outside this trust model.

The coordinate policy is:

| Grid | Mapping and support |
| --- | --- |
| Native | H x W image, XYZT affine retained as metadata; processing tensors use TZYX |
| Padded | Bottom/right replicate to multiples of 32 for images; no evidence added |
| Coarse | Padded H/4 x W/4; disjoint logit-cell D; positive pixel-overlap transport for source mass/support |
| Feature | Padded H/16 x W/16; transported teacher average; partial/padded cells excluded from masks |

For named coarse supervision, provide nonnegative `B x 4 x Ht x Wt` class-mass
density and `B x Ht x Wt` support equal to the class sum, together with a source
manifest hash matching that registration's source manifest, registration hash,
explicit ontology `[BG,RV,MYO,LV]`, verified
orientation and kind `fixed_anatomical_mass` or `conditional_z_mass`.
`coarse_evidence` transports both with the same linear map. UNKNOWN has zero
weight; the uniform storage value at zero support cannot become BG evidence.
`coarse_semantic_loss` uses `outputs["coarse_logits"]` directly. It never resizes
native logits while claiming direct detail-gradient separation.

`masked_feature_objective(model, images, teacher, geometry, mask)` requires a
boolean mask on fully native stride-16 cells. It masks the aligned 16x16 image
regions in all three input slices before calling the student, then uses detached
teacher targets. It provides representation learning without named labels.

## Checkpoints and inference

`construct_student(recipe, approval, device="cpu")` returns the approved model
in eval mode plus provenance. `save_checkpoint(model, new_folder, recipe,
lineage)` writes exactly `model.pt` and `MANIFEST.json`, returning the manifest
SHA-256. Required lineage includes the frozen bank hash, frozen teacher
`checkpoint_sha256` and source, disjoint locked train/val/test lists, actual
selected train patients and actual optimizer step count.

`load_checkpoint(folder, expected_manifest_sha256, approval, device="cpu")`
requires the external manifest digest, strict inventory, matching payload and
recipe digests, current implementation hashes and the same approved backbone.
It reconstructs the new architecture and strictly loads its state dictionary.
Historical `GuidanceStudent/last.pt` checkpoints are rejected.

The schema is `layer_guided.checkpoint.v1`. State keys preserve the reference
prefixes `encoder`, `input_adapter`, `lateral`, `up_channels`, `refine`,
`coarse_head`, `feature_project`, `detail_stem`, `detail_fuse`, `detail_head`,
and the recorded `detail_bound` buffer. Official native stages live under
`encoder.downsample_layers` and `encoder.stages`. Mean/std and bound must equal
the fixed v1 defaults. Zero-step artifacts are `INITIALIZED_ONLY`; positive
step counts are `REPRESENTATION_ONLY`. Semantic admission is always off.

Inference is `model(image_contexts)` with no teacher inputs. Outputs retain all
reference keys: native `logits`, padded `logits_padded`, `coarse_logits`, padded
`raw_detail`/`bounded_detail`, 64-channel stride-16 `features`, padded boolean
`native_valid`, and `native_hw`. These tensors are not automatically authorized
named segmentations. No named export/evaluation path is added.

## Isolated research factors

The control architecture ID is `layer_guided_convnext_tiny_conventional_v1`;
copy the recipe and change only that ID. It uses the same parameterization and
initialization with a bilinear coarse plus unprojected detail head. Its coarse
target is applied after D to native logits, so detail receives gradients. It
computes/discards the reference composition, adding overhead; parameter equality
does not imply equal FLOPs. No matched patient experiment has run.

`witness_plan` fixes TRAIN patient/class/source/sample metadata and requires all
selected patients to be represented. `guarded_decoder_step` accepts pure frozen
objective/witness closures in eval mode, limits selected parameters to explicit
`refine.0` through `refine.3` blocks, and returns actual displacement and loss
acceptance. It has no optimizer state, momentum or weight decay. Closure/sample
consistency is the caller's responsibility. An unrelated AdamW step after it is
outside the guarantee; an AdamW-compatible guard requires a separate version.

`image_proposal` clones the frozen energy buffers and returns a detached
distribution with `semantic_admission=False`. Unaries, anchors and image
affinities need separately justified provenance before any later supervision
policy. The recipe does not silently enable either primitive in training.

Before a separately authorized pilot: supply reviewed local weights/source and
registered TRAIN artifacts; run approved-backbone preflight and measure memory;
freeze splits, sampling/witness plan, actual step counts and equal schedule;
compare the split head with the conventional head first. Add the guard and then
the refiner as separate factors. Do not tune thresholds against exposed VAL/TEST.
