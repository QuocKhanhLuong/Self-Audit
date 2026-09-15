# Mask-free anatomical label generation — operator guide (150-epoch pipeline)

This document covers the launch, export and freeze layer (W7) of the mask-free
pipeline in `src/self_audit_maskfree/`. It is the operator's reference: what the
commands actually are, what they write, what is gated on what, and what this
pipeline does *not* claim.

## Status at the time of writing

**150-epoch training: NOT STARTED in the latest supplied log.** On 2026-09-15
the user reported an ACDC launch on an explicitly selected RTX 5070 Ti. Stage 2
inventory failed with `MixedStudyGeometryError` for `acdc:patient001`, before
preflight or training. That log establishes neither a passed batch-8 gate nor
a trained checkpoint. The implementation checks in this checkout use synthetic
CPU fixtures; the actual conflicting source headers have not been inspected here.

The launchers reflect that: without `RUN_FULL=1` they print a fully resolved
plan and exit 0 without touching the GPU, the data or the workspace.

The YAML `raw_images/ACDC` and `raw_images/MnMs` paths are **unverified
placeholders**. Set `ACDC_DATA_ROOT` and `MNMS_DATA_ROOT` to the actual inventoried
raw image roots on the rented host. The templates deliberately do not reuse
the supervised pipeline's `preprocessed_data` directory: its crop and cohort
provenance has not been established as mask independent. Any explicitly chosen
preprocessed images require documented preprocessing provenance before a
mask-free scientific claim; inventory cannot reconstruct missing history.

## What this pipeline is, and is not

It generates anatomically named *draft* labels from unlabeled cardiac images,
scores competing partitions by how well they predict observations they were not
fitted on, and trains two lockstep students — one on unaudited labels, one on
audited labels — inside a single 150-epoch timeline per dataset.

* No manual segmentation mask influences training, proposal generation,
  selection, thresholds, checkpoint choice or export. Hand-specified anatomical
  priors and class definitions *are* used and are enumerated by the ontology
  resolver; the claim is **no manual segmentation masks**, not the absence of
  all prior information.
* A predictive negative log-likelihood is not segmentation accuracy. An evidence
  margin is not a calibrated probability of correctness. A validated freeze
  manifest proves which bytes were compared, not that any of them is right.
* Runtime completion and scientific outcome are separate fields. A run can
  finish all 150 epochs and still have failed its label-generation objective
  (for example if useful coverage collapses); the launcher prints
  `scientific_outcome` at the end of the completion check for exactly this
  reason.

## Commands

### The documented entry point

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
RUN_FULL=1 TOTAL_EPOCHS=150 BATCH_SIZE=8 \
bash scripts/run_maskfree_acdc_mnms.sh
```

Runs full native ACDC, then full independent native M&Ms. A single numeric
`CUDA_VISIBLE_DEVICES` value is resolved against the physical `nvidia-smi`
inventory and replaced with the selected GPU UUID before Python starts. ACDC
must pass its own completion check before M&Ms starts. Neither run reads the
other's checkpoints.

```bash
scripts/run_maskfree_acdc_mnms.sh --help
scripts/run_maskfree_acdc_mnms.sh --datasets mnms      # one dataset, or reorder
```

### One dataset

```bash
scripts/run_maskfree_full.sh acdc
scripts/run_maskfree_full.sh --dataset mnms
scripts/run_maskfree_full.sh --verify-run <run_dir>    # re-check a finished run
scripts/run_maskfree_full.sh --help                    # full environment reference
```

Stages, each gated on the previous one:

| # | stage | command it runs |
|---|-------|-----------------|
| 1 | resolve | writes one unique read-only config through W5's loader |
| 2 | inventory | `scripts/prepare_maskfree_data.py --dataset … --root … --output …` |
| 3 | preflight | `scripts/train_maskfree.py --config … --preflight` |
| 4 | train | `scripts/train_maskfree.py --config …` |
| 5 | verify | the completion check below |

### Environment

`scripts/run_maskfree_full.sh --help` is the authoritative list. The ones that
matter most:

| variable | meaning |
|----------|---------|
| `RUN_FULL=1` | actually execute; anything else prints the plan |
| `WORKSPACE` | run root, default `/home/linhdang/workspace/quockhanh_workspace/SpecMamba` |
| `CUDA_VISIBLE_DEVICES` | optional single physical index or UUID; resolved to a real UUID on GPU |
| `MASKFREE_GPU_UUID` | explicit physical GPU UUID; takes precedence over the inherited selector |
| `MASKFREE_GPU_OVERRIDE` | explicit rented-GPU identity record when the product is not RTX 4070 |
| `MASKFREE_ALLOW_GPU_NAME` | legacy product-name override, also required for the unsupported list |
| `MASKFREE_FALLBACK` | `4x2` or `2x4`, resolved **before** the run |
| `ALLOW_CPU=1` | CPU software check; not the contracted experiment |
| `ACDC_DATA_ROOT` / `MNMS_DATA_ROOT` | per-dataset image roots (sequence script) |

## The GPU and batch gates

**GPU identity.** On GPU, the launcher clears the inherited visibility mask
while reading `nvidia-smi --query-gpu=index,uuid,name`. It accepts one explicit
`MASKFREE_GPU_UUID`, or resolves one numeric/UUID `CUDA_VISIBLE_DEVICES` value
to the corresponding physical row. Multiple selectors or an inherited selector
that disagrees with the explicit UUID are hard errors. The selected product
must contain `RTX 4070`, unless `MASKFREE_GPU_OVERRIDE` records the deliberate
rented-GPU exception. `MASKFREE_ALLOW_GPU_NAME` remains a product-name
override for compatibility and for the unsupported list (currently `5070 Ti`).
The decision, original selector, physical index, UUID, product name and
override record are written to `reports/gpu_identity.json`.

**Batch.** The requested shape is physical 8, accumulation 1. Neither has been
measured on the target hardware. If the bounded preflight probe OOMs, resolve
the fallback *before* the run:

```bash
MASKFREE_FALLBACK=4x2 RUN_FULL=1 bash scripts/run_maskfree_full.sh acdc
```

`4x2` and `2x4` both give effective batch 8. They change optimizer microbatch
grouping; W4 contrastive negatives remain within-image, and the launcher
records the fallback choice and its limitation in the run receipt. There is no
mid-run OOM retry, batch change, LR rescale or resolution change.

The full run is gated on `reports/gate_receipt.json` with `status: "pass"`, and
the launcher additionally refuses to proceed if the gate measured a different
physical/accumulation shape than the run requests. A receipt only licenses the
shape it actually measured.

## Live terminal metrics and load diagnostics

The Python entry points print flushed progress to stderr, captured by the
launcher's existing `tee` logs. The display is a plain terminal dashboard that
also works inside a GUI terminal and redirected logs, without cursor escapes.
It starts before importing torch. No extra UI package is required.

* `stage.start` / `stage.done`: operation, path or unit, and elapsed wall seconds.
  Inventory includes file index/count, shape, affine, orientation, source hash,
  current frame fingerprinting, duplicate decisions and grid comparisons.
* `heartbeat`: every 10 seconds, including the innermost active phase, nested
  phase path, current file/frame/slice or unit, and phase elapsed time. Set
  `MASKFREE_HEARTBEAT_SECONDS=5` for a shorter interval. This is liveness context,
  not a claim that the GPU is busy or that a blocked operation has progressed.
* `metrics`: at most every 5 seconds after a batch, plus the final batch. Shows
  global epoch/batch, physical batch, accumulation, LR, label ramp, actual
  component update counts, producer loss components/collapse diagnostics, both
  student losses and support, audit fit/select NLL and improvement, coverage,
  unresolved/accepted counts, class occupancy, stage times and CUDA memory bytes.
  Values are running epoch means with denominators per metric: loss counts are
  batch observations, while NLL counts are scored pixels (across candidate fits).
  Student `valid_support` sums weights; coverage counts positive-weight pixels.
  Loss is recorded once per batch. CPU memory fields remain unavailable.
* `epoch.summary`: completed/partial status, full epoch metrics and counters.
  `image_only.metric` prints computed aggregate results by split after freezing:
  verification NLL, paired comparisons, stability, coverage and availability
  reasons. Per-unit details and controls remain in the complete report files.

Diagnostic journals are `manifests/inventory_progress.jsonl`,
`reports/preflight_progress.jsonl`, and `reports/progress.jsonl`. They append on
resume and may contain repeated attempted work; checkpoints and
`reports/epoch_metrics.jsonl` remain authoritative. Repeated fast stage messages
are throttled, while the heartbeat always describes the current operation.
Stage times are inclusive and must not be summed into end-to-end latency.

### Recovering the ACDC inventory geometry failure

Discovery first proves duplicate image content using decoded frame shape, dtype
and bytes. An identical native NIfTI 3D frame re-export can be omitted in favor of its 4D
cine source even when its header differs; the manifest records both geometries
and the content proof. Different image content is retained.

Remaining sources must share a study's spatial grid and observation partition.
Only affine rounding with maximum displacement at the eight spatial corners
of **at most 0.0001 voxel**, in both voxel bases, is accepted. Original affines
are preserved for export. This does not enable temporal evidence or resampling.
Material shape, orientation, unit or affine conflicts still stop inventory,
with full source paths and geometry differences saved in `inventory_acdc.json`
as `FAILED` / `training_status: NOT_STARTED`.

Pull the fix on the rented host and relaunch with a **new run ID**, retaining
the chosen image root and the explicit GPU override. The failed inventory has
no training checkpoint to resume. Do not edit or reuse its read-only YAML.
The launcher generates a fresh ID when `RUN_ID` is unset; preflight must still
measure the requested physical batch 8 before the 150-epoch run starts.

## Completion check

`--verify-run <dir>` reads `reports/pipeline_report.json` and requires:

* `total_epochs == 150`, `epochs_completed == 150`, `last_completed_epoch == 149`;
* `completed == true` and `bounded_run` unset — a `--max-steps` or `--max-epochs`
  probe can never pass as a finished run;
* `resolved_config.json`, `source_provenance.json`, `data_manifest.json`,
  `reports/epoch_metrics.jsonl`, `checkpoints/last.pt`,
  `checkpoints/{producer,student_no_audit,student_audited}_final.pt`,
  `exports/freeze_manifest.json`,
  `exports/deployment/freeze_manifest.json`, and
  `reports/image_only_summary.{json,csv,md}` all present;
* `validate_freeze(..., require_complete=True)` passes over the freeze manifest;
* the primary freeze enumerates every method in its declared
  `REQUIRED_COMPARISONS` set;
* the deployment freeze validates its two full-input student methods;
* `finalization.available` and `hardware_qualified` are true, so a CPU 150-epoch
  software run cannot pass the contracted completion gate;
* verification rows cover exactly `finalization.units` unique units and every
  frozen comparison method is present in each row.

It then prints `scientific_outcome` and states explicitly that it checked
runtime completion only.

## Run layout

```
$WORKSPACE/runs/maskfree150/<dataset>/<run_id>/
  resolved_config.json  source_provenance.json  data_manifest.json
  manifests/            image-only discovery output
  checkpoints/          last.pt, producer_final.pt, student_*_final.pt
  reports/              epoch_metrics.jsonl, pipeline_report.json,
                        gate_receipt.json, preflight_report.json,
                        failure_report.json, experiment_rows.json,
                        verification.json, image_only_summary.{json,csv,md},
                        gpu_identity.json
  banks/                candidate-bank records and evidence traces
  exports/              predictions/, freeze_manifest.json, deployment/
$WORKSPACE/run_configs/maskfree150/<run_id>/<dataset>.yaml   (mode 0444)
$WORKSPACE/logs/maskfree150/<run_id>/                        (stage logs)
```

`output_dir` in the config is the **workspace root**; the trainer appends
`runs/maskfree150/<dataset>/<run_id>`.

## Export and freeze (`src/self_audit_maskfree/export.py`)

```python
export_prediction(output_dir, *, record, prediction_name, labels, probabilities,
                  validity, alternatives, checkpoint_id, version) -> dict
assemble_volume(output_dir, *, prediction_name, volume_id=None, study_id=None,
                write_nifti=True) -> dict
assemble_all(output_dir, *, prediction_name, write_nifti=True) -> list[dict]
freeze_predictions(output_dir, entries, checkpoint_paths, *, dataset, protocol,
                   epoch=149, required_methods=None, required_unit_ids=None,
                   required_checkpoints=None, required_nuisance_files=None) -> dict
validate_freeze(manifest, *, require_complete=False) -> None
verified_freeze_session(manifest, *, require_complete=False)
resolve_native_target(record) -> dict      # the geometry decision, on its own
comparison_completeness(present) -> dict
```

`export_prediction` writes one sampling unit on the stored model grid as an
`.npz` plus a JSON sidecar. `freeze_predictions` accepts either per-unit shards
or already-assembled volume descriptors: given unit shards — which is how the
trainer calls it — **it assembles the volumes itself at finalization**, so a
freeze can never enumerate units whose volume was never built. Both the unit
shards and the assembled volume files are hashed into the manifest.

### Geometry rules

* A volume is `grid: "native"` only when the source was a NIfTI whose affine
  survived the data layer, `stored_to_native_axes` is a real permutation of
  `(0,1,2)`, and permuting the reverse-interpolated stored shape by it
  reproduces `native_shape` exactly. That arithmetic is checked, not trusted.
* Reverse interpolation: probabilities bilinear then renormalized, labels
  nearest, validity bilinear (it is a weighted support). Then the axis
  permutation. Each choice is recorded in the volume sidecar.
* NIfTI affines stay in the source spatial units. The exporter preserves the
  input `xyzt_units`/`spatial_unit` when nibabel can represent it and records
  `spacing_mm` separately only when W3 marked the conversion valid; raw header
  zooms are never silently called millimetres.
* An `.npy` source has no native affine, so it exports `grid: "stored"` with
  `native_export_available: false` and a reason. It is never relabelled native
  because the numbers happen to fit.
* A slice no unit produced gets `validity = 0` and is listed in
  `missing_slice_indices` with `complete_volume: false`. It is not converted to
  background.
* Re-exporting a unit with different content under the same identity raises. A
  second, divergent freeze does not overwrite the first.

### Record keys consumed by the exporter

From W3's manifest records (`RECORD_KEYS` in `export.py` is the machine-readable
copy). Required for a *native* export — a missing one degrades that unit to a
stored-grid export with a reason rather than failing the run:

`study_id`, `patient_id`, `unit_id`, `dataset`, `split`, `volume_id`,
`slice_index`, `num_slices`, `source_format`, `native_shape`, `native_slice_shape`,
`depth_axis`, `stored_to_native_axes`, `native_geometry`
(`{available, affine, spacing, reason}`).

Optional, carried through into sidecars and freeze lineage: `source_path`,
`protocol`, `frame_index`, `partition_id`, `manifest_id`, `cohort_provenance`,
`spacing_mm`, `spatial_unit`, `xyzt_units`, `spacing_valid`.

Units are grouped by the W3 `volume_id` (with `acquisition_id` as a legacy
alias). A patient or study identifier is only a fallback when W3 did not emit a
volume identity; the exporter never merges distinct acquisitions merely because
they share a patient.

`stored_to_native_axes` is the argument to `np.transpose`: applied to the stored
volume `(num_slices, *native_slice_shape)` it must yield `native_shape`. A
record with no `slice_index`/`num_slices` at all (a full-input deployment
record) is treated as a single-slice volume, is marked
`slice_identity: "defaulted…"`, and cannot claim a native grid.

### What the freeze manifest contains

`manifest_id` (a canonical hash of the body), dataset/protocol/epoch, the
semantic order, the lineage every entry agreed on (`partition_id`,
`manifest_id`, pseudo-label `version`), the full `compared_methods` list, a
`completeness` block against `REQUIRED_COMPARISONS`, every prediction file with
its SHA-256 and size, and every checkpoint with its SHA-256.

`configs/maskfree_experiments.yaml` is the human-readable mirror of
`REQUIRED_COMPARISONS`; `tests/test_maskfree_export.py` asserts they cannot
drift apart.

`verified_freeze_session(manifest)` is the bounded verification-loop API. It
performs a complete physical hash pass on entry and exit. Calls to
`validate_freeze` inside the context may reuse the exact yielded manifest
object after checking its identity and body, so a large frozen dataset is not
rehash-scanned once per verification unit. A copied or reloaded mapping does
not receive that fast path, and an exit hash mismatch fails the session.

## Supervision ledger for this layer

Nothing in `export.py`, the two launchers or the three configs reads, writes,
resolves or accepts a path to a manual mask, a reference segmentation, an
annotation file, a pretrained weight, or a GT-derived threshold. There is no
selection logic here at all: the exporter records what it was handed. Ground
truth is read only by `src/self_audit_maskfree/evaluation/` through
`scripts/evaluate_maskfree_reference.py`, as a separate process, after a
validated freeze, and it cannot return a decision into training.

The configs contain only `MaskfreeConfig` fields, so W5's strict loader rejects
any legacy or pretrained option that is added to them by accident.

### Optional reference configuration

Reference evaluation is a separate post-freeze process. If it is enabled, each
dataset needs its own explicit config, including a fixed raw-label mapping and
an exact key for every reference volume:

```yaml
dataset: acdc
split: test
protocol: spatial_predictive
reference_root: /data/acdc/references
reference_label_map:
  0: BG
  1: RV
  2: MYO
  3: LV
cases:
  - patient_id: patient001
    volume_key: patient001_frame01
    mask_path: /data/acdc/references/patient001_frame01.nii.gz
    geometry_valid: true
```

Set `ACDC_REFERENCE_CONFIG` or `MNMS_REFERENCE_CONFIG` to an explicit file when
calling the sequential launcher. The hook runs only after both dataset freezes
pass; it does not feed a reference metric back into training or selection.

### Exact resume

An interrupted dataset run resumes in its existing immutable run directory with
the same resolved config, checkpoint, source manifest, runtime settings and
physical device. For example:

```bash
PYTHON=python3
WORKSPACE=/home/linhdang/workspace/quockhanh_workspace/SpecMamba
RUN_ID=<existing-acdc-run-id>
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<same-resolved-gpu-uuid> \
  "$PYTHON" scripts/train_maskfree.py \
  --config "$WORKSPACE/run_configs/maskfree150/$RUN_ID/acdc.yaml" \
  --resume "$WORKSPACE/runs/maskfree150/acdc/$RUN_ID/checkpoints/last.pt"
bash scripts/run_maskfree_full.sh --verify-run \
  "$WORKSPACE/runs/maskfree150/acdc/$RUN_ID"
```

The resume guard refuses a changed scientific config, workspace, source or
device identity. After ACDC passes verification, start M&Ms as a fresh
dataset-global run; the sequential wrapper creates independent ids and never
resumes ACDC automatically.

## Known limitations of this layer

* The local implementation checks did not touch a GPU or real data. The reverse geometry transform has been
  checked against synthetic asymmetric volumes only; whether ACDC and M&Ms
  records actually carry the metadata it needs is unverified, and the
  preprocessed `.npy` roots are expected to resolve to stored-grid exports with
  no native claim.
* v1 implements `spatial_predictive` only. The requested data's frame count,
  timing and native geometry are unconfirmed until image-only inventory runs;
  no cine claim is made by these templates or this exporter.
* `validate_freeze` proves byte integrity. It says nothing about the quality,
  anatomical correctness or calibration of the predictions it protects.
