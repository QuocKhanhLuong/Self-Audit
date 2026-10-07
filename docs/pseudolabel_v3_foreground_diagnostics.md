# Foreground support diagnosis, 07-Oct-2026

Integration base: `a7a34ae0cdffee4bd14f030666ec6a1f410927fb`, preserving the
unknown-spatial-units opt-in. The initial diagnostic review used `50999b0`.
This page describes the reporting-only part of the work, which adds image-only
support diagnostics without changing the recipe. The separately opt-in
[experimental bootstrap](pseudolabel_v3_bootstrap_experimental.md) introduces a
new declared training recipe. Neither establishes real-data foreground recovery
or certifies quality, and no student safeguard is bypassed.

## Distinguish three outcomes

1. The teacher can freeze predictions containing only BG or UNKNOWN. This is
   valid output data, although it is unsuitable foreground supervision.
2. Independent evaluation can score those predictions. A missing prediction
   against present reference anatomy has Dice 0. A class empty on both sides is
   excluded and represented as JSON `null`; UNKNOWN does not hide missed anatomy.
3. Student training deliberately refuses a frozen **training** split with zero
   foreground. Validation-only foreground cannot satisfy this guard. The failure
   is `NO_FOREGROUND_SEEDS`, not a failure to calculate Dice.

A failed pipeline now preserves `PIPELINE_SUMMARY.json` with the failed stage,
completed stages, existing completed evaluation, and a non-success status.
The original exception and nonzero exit are preserved; W&B receives a failed
exit code. No student inference/evaluation is launched after a student failure.

## What the new diagnostics mean

Teacher `train_metrics.json` records support for every processed batch, even
with `--log-every 0`; sampled records and W&B receive the same diagnostic fields.
Counts are region observations across batches, not unique anatomy or accuracy.

- `enclosure_pairs`: connected internal wall/cavity pairs found by the existing
  image-only topology rules
- `raw_seed_{bg,rv,myo,lv}_regions`: raw evidence passing confidence, margin and
  minimum region-size checks, before prototype history
- `accepted_{bg,rv,myo,lv}_regions`: surviving prototype class-agreement checks;
  these are the regions used by the semantic seed loss
- `decoded_{bg,rv,myo,lv}_regions`: surviving guided-evidence and neural-agreement
  checks, before dense pixel support/confidence gates; available in sampled
  metrics and export reports
- Existing `decoded_*_pixels` metrics: dense support before temporal consistency

Each epoch prints accepted support by class. A background-only seed loss can
fall toward zero without the teacher having learned any foreground. A warning
therefore accompanies an epoch with no accepted foreground; training is not
prematurely stopped because later image-only learning could change topology.

`export_report.json` reports per-patient `seed_region_counts`,
`pre_consistency_class_pixels`, existing final `class_pixels`, and
`consistency_rejected_class_pixels`. These isolate missing raw seeds from
prototype/neural/dense rejection and later temporal rejection.
`run_summary.json` additionally reports `split_support` separately for train
and val. Its `NO_FOREGROUND_SEEDS` means zero final exported foreground
supervision; it does not establish that raw anatomical seeds were absent.
`FOREGROUND_OBSERVED` is support only; `teacher_ready` stays
`NOT_EVALUATED`.

## Verified bootstrap limitation, not a real-run diagnosis

Foreground semantics currently require suitable connected ring/cavity topology
in the hard anonymous clustering. With no enclosure pairs and default gates,
RV/LV receive no positive vote and MYO's boundary-only logit is at most 0.75.
Its best possible softmax confidence is below 0.414, below the unchanged 0.70
acceptance threshold. The prototype bank cannot create unsupported seeds.

Reconstruction and registration have no direct gradient to region-head
parameters, but do update upstream features and can therefore change clustering
indirectly. The clustering information objective is invariant to arbitrary
spatial pixel permutation; it gives no explicit connected-wall/cavity incentive.
This makes foreground bootstrap possible to miss; longer training is not a
proven remedy.

Two independent synthetic CPU probes illustrate this limitation: a fixed
64x64 radial-intensity image, seed 42, produced zero enclosures and foreground
after 40 default-model updates in one probe and 100 reduced-width updates in
another. The reduced-width probe's semantic seed loss fell from about 1.367 to
0.002 using background-only seeds. Conversely, a synthetic valid ring+RV
assignment produced 295 foreground pixels at unchanged gates and retained them
under consistent temporal evidence. None of these is an ACDC/M&Ms result.

A defensible algorithmic repair requires matched image-only experiments and a
new declared recipe, rather than relaxing connectedness or inventing masks.
A zero-foreground log establishes unusable exported supervision, but the saved
configuration and stage-level export support are still needed to distinguish
missing raw candidates from later rejection. A real-data rerun must verify recovery.

## Inspect an existing run safely

Do not overwrite an active checkout, environment, output directory, or frozen
artifact. Existing checkpoints bind source identity. New diagnostic fields
require a fresh run in a separate checkout; old runs remain inspectable without
retraining. This read-only command prints selected small JSON fields, excluding
checkpoints, images, W&B files, credentials, and arbitrary log text. Replace the
path with the full pipeline run directory:

```bash
RUN=/absolute/path/to/full_run python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['RUN'])
def read(relative):
    path = root / relative
    return json.loads(path.read_text()) if path.is_file() else None
def select(value, keys):
    return {k: value[k] for k in keys if k in value}
result = {}
summary = read('teacher/run_summary.json')
if isinstance(summary, dict):
    result['teacher'] = select(summary, ('status','optimizer_steps','exports',
        'valid_foreground_pixels','teacher_ready','split_support'))
history = read('teacher/train_metrics.json')
if isinstance(history, list):
    names = ('epoch','step','accepted_regions','semantic_seed','enclosure_pairs')
    names += tuple(f'{p}_{c}_regions' for p in ('raw_seed','accepted')
                   for c in ('bg','rv','myo','lv'))
    result['last_teacher_batches'] = [select(r, names) for r in history[-10:]]
report = read('teacher/export_report.json')
if isinstance(report, dict):
    names = ('patient_id','split','class_pixels','pre_consistency_class_pixels',
             'consistency_rejected_class_pixels','seed_region_counts','support_status')
    result['export_support'] = [select(r, names) for r in report.get('patient_rows', [])]
for name in ('evaluation_val.json','PIPELINE_SUMMARY.json'):
    value = read(name)
    if isinstance(value, dict):
        result[name] = select(value, ('status','failed_stage','error_type','return_code',
            'completed_stages','foreground_mean','rv','myo','lv','known_fraction','pseudo_metrics'))
print(json.dumps(result, indent=2, allow_nan=False))
PY
```

This is a diagnostic readout, not freeze integrity verification. Preserve the
full original `pipeline.log` for the exact exception. Do not use the readout to
re-sign artifacts or select thresholds from evaluation labels.
