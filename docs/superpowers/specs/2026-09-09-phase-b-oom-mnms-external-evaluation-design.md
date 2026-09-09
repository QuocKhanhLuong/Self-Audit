# Phase-B OOM and M&Ms External Evaluation Design

**Date:** 2026-09-09

**Status:** Approved design, pending implementation plan

## Objective

Deliver two related, independently testable improvements to the Self-Audit
pipeline:

1. Make Phase-B auditor validation use bounded GPU memory without changing the
   metric contract or checkpoint-selection semantics.
2. Evaluate a frozen ACDC-trained Phase-C checkpoint on the existing four-class
   M&Ms `testing` cohort as an independent external test, without allowing M&Ms
   to influence training, checkpoint selection, or threshold calibration.

The server already contains both datasets. ACDC raw data is under `data/ACDC`.
The usable M&Ms data is under `preprocessed_data/mnm`; the separate
`preprocessed_data/mnm_binary` tree is outside this design because the current
model and metrics use the four-class `BG/RV/MYO/LV` contract.

## Verified Deployment Facts

The server-side M&Ms tree has this layout:

```text
preprocessed_data/mnm/
|-- training/{volumes,masks,metadata.json}
|-- validation/{volumes,masks}
`-- testing/{volumes,masks}
```

The observed split counts are 300/300 training volume-mask pairs, 68/68
validation pairs, and 272/272 testing pairs. A sampled testing pair had matching
shape `(224, 224, 14)`, a `float32` volume, a `uint8` mask, and labels
`[0, 1, 2, 3]`. The implementation must discover and report counts rather than
hard-code these observations.

Case identifiers have the form `<patient>_tXX`. The inspected training metadata
records `num_slices` and `target_size` but does not provide an authoritative
ED/ES label; metadata files for the other splits are not assumed to exist. The
evaluator must therefore report the full cohort and individual timepoint cases.
It must not infer ED or ES from the numeric timepoint or its ordering.

## Non-Goals

- Do not put ACDC or M&Ms data in Git or Git LFS.
- Do not use the binary M&Ms derivative.
- Do not retrain or fine-tune on any M&Ms split.
- Do not select a checkpoint, epoch, threshold, or other hyperparameter from
  M&Ms results.
- Do not reconstruct raw M&Ms NIfTI or claim native-geometry metrics from the
  already resized `224 x 224` arrays.
- Do not guess ED/ES phase labels.
- Do not change Phase-B training loss composition or the existing provenance
  namespaces.

## Current Problems

### Phase-B validation retains dense GPU state

`_auditor_batch(..., collect=True)` currently stores detached local logits,
dense local targets, predicted quality deltas, and target Dice deltas. It
materializes separate tensors for `on_policy`, `synthetic`, and `combined`.
`validate_auditor_epoch` retains these tensors for the whole validation loader
and concatenates them on the GPU at epoch end.

The observed failure occurred while concatenating `local_target`: the process
needed another 1.27 GiB with only 1.05 GiB free. Fragmentation settings cannot
solve the live-tensor retention or the full-cohort concatenation.

### The declared M&Ms protocol is not executable

`build_patient_dataset` always constructs `ACDCDataset`, even when the config
declares `dataset: mnms`. `validate_dataset_splits` returns an unvalidated
external-dataset marker for M&Ms. The full training pipeline only creates ACDC
train and validation loaders, and no entrypoint consumes the nested
`test: dataset: mnms` block in `configs/self_audit_acdc_to_mnms.yaml`.

The existing checkpoint audit command is validation-oriented and describes its
evidence as diagnostic rather than an independent external test. Reusing its
cohort evaluation logic directly would also carry ACDC-specific wording and
phase grouping into M&Ms reports.

## Architecture

Implementation is divided into two workstreams. They share no mutable runtime
state and will have separate implementation plans and review gates.

### Workstream A: Streaming transition metrics

Introduce a focused transition-metric accumulator in
`src/self_audit/evaluation/transition_accumulator.py`.

For each transition group, the auditor batch will reduce dense local outputs
before the batch can escape:

```text
local logits [N,3,H,W]
        |
        +--> argmax [N,H,W]
        +--> bincount(target * 3 + prediction, minlength=9)
        +--> CPU int64 confusion matrix [3,3]

delta_q / delta_dice
        +--> detached CPU float32 vectors [N]
```

The accumulator interface will be:

```python
class TransitionMetricAccumulator:
    def update(
        self,
        *,
        local_confusion: torch.Tensor | np.ndarray,
        delta_pred: torch.Tensor | np.ndarray,
        delta_target: torch.Tensor | np.ndarray,
    ) -> None: ...

    def merge(self, other: "TransitionMetricAccumulator") -> None: ...

    def finalize(self, *, neutral_margin: float) -> dict[str, float]: ...
```

The accumulator owns only a `3 x 3` integer confusion matrix and CPU delta
vectors. `local_fix_f1` and `local_regress_f1` are derived exactly from global
TP/FP/FN counts. AUROC, AUPRC, accuracy, correlation, and transition counts are
computed from the concatenated CPU delta vectors. Empty populations preserve
the current `nan` and empty-class behavior.

`_auditor_batch` will return a per-provenance batch summary instead of dense
`transition_data`. Validation will update `on_policy` and `synthetic`
accumulators immediately. `combined` will be finalized by merging those two
accumulators, not by storing a third copy of per-transition tensors.

The public result keys and checkpoint rule remain unchanged:

- `audit/on_policy/*`
- `audit/synthetic/*`
- `audit/combined/*`
- legacy flat keys sourced from `combined`
- `primary_metric` sourced only from on-policy AUROC, with the existing
  on-policy accuracy fallback

The training path uses `_auditor_batch(..., collect=False)` and remains
unchanged except for internal return-field naming when no collection is
requested.

### Workstream B: Independent M&Ms evaluation

#### Dataset dispatch and validation

`build_patient_dataset` will dispatch on the normalized `dataset` value:

```text
acdc -> ACDCDataset
mnms -> MNMSDataset
other -> hard error
```

The M&Ms branch consumes `data_root`, split, image size, depth axis, cache
settings, and the explicit `raw_to_acdc` mapping. It must reject a non-four-class
configuration and must never silently use `mnm_binary`.

M&Ms validation will inspect the requested split and return a real validation
record containing dataset name, split, case count, patient count, volume-mask
pairing status, label set, mapping, and a deterministic cohort membership
signature. It fails closed for an absent root, an empty split, unmatched pairs,
shape mismatch, unknown labels, or an invalid mapping.

The default semantic mapping is retained and recorded in every report:

```text
M&Ms 0 BG  -> ACDC 0 BG
M&Ms 1 LV  -> ACDC 3 LV
M&Ms 2 MYO -> ACDC 2 MYO
M&Ms 3 RV  -> ACDC 1 RV
```

#### Protocol configuration

`configs/self_audit_acdc_to_mnms.yaml` remains a protocol-level config rather
than a Phase A/B/C training config. It will explicitly separate source training
identity from external-test settings:

```yaml
protocol: acdc_to_mnms_domain_shift

source_training:
  dataset: acdc
  phases: [annotation, auditor, joint]

external_test:
  dataset: mnms
  split: testing
  data_root: preprocessed_data/mnm
  raw_to_acdc: {0: 0, 1: 3, 2: 2, 3: 1}

image_size: 256
num_classes: 4

model:
  encoder_name: convnext_tiny
  shared_channels: 96
  num_classes: 4
  window_k: 8
  max_turns: 3

audit:
  tau_accept: 0.0
  t_max: 3
  neutral_margin: 0.005
```

A CLI `--data-root` override takes precedence so deployments can keep data
outside the repository. Absolute server paths are never committed.

#### External evaluator

Add `scripts/evaluate_external_mnms.py`. It is an evaluation-only entrypoint
with this required flow:

1. Load and validate the external protocol config.
2. Apply the explicit data-root and split overrides.
3. Validate the M&Ms cohort before loading it for inference.
4. Build the four-class model and bind the requested checkpoint.
5. Verify that architecture and live state match the checkpoint binding.
6. Resolve `tau_accept` only from an explicit CLI value or the predeclared
   config value. Do not expose threshold search or M&Ms calibration options.
7. Run all comparison modes on the frozen M&Ms testing cohort.
8. Emit one versioned JSON report.

Reusable cohort evaluation code currently embedded in
`scripts/audit_checkpoint.py` will move to
`src/self_audit/evaluation/cohort.py`. Both the existing ACDC diagnostic command
and the new external evaluator will import it. Dataset-specific statements,
phase grouping, and evidence classification remain in their callers.

The external report schema must include:

- schema version and generation timestamp
- `evidence_class: independent_external_evaluation`
- protocol name and dataset `mnms`
- split `testing`
- actual case and patient counts
- cohort membership signature
- checkpoint path, file hash, live-state digest, and model identity
- exact label mapping
- stored array grid `224 x 224` when confirmed by metadata
- network input grid `256 x 256`
- `metric_space: volume_resized`
- `native_dice_available: false`
- `phase_partition: unavailable_without_authoritative_metadata`
- resolved `tau_accept` and its source
- per-case, per-patient, and full-cohort metrics
- per-class RV/MYO/LV Dice and macro foreground Dice
- comparison modes `initial_only`, `always_accept_refinement`, `self_audit`,
  and `oracle_accept`; `oracle_accept` is explicitly labeled a non-deployable
  upper-bound diagnostic
- acceptance/rescue/regression audit metrics already supported by the shared
  evaluation stack

The evaluator groups cases into patients using the prefix before `_t`, but it
does not name either timepoint ED or ES. If authoritative phase metadata is
added later, that is a separate protocol revision with a new report schema or
explicit phase-mapping artifact.

## Operational Flow

Training remains ACDC-only and continues to use the existing A/B/C configs:

```bash
python scripts/train_self_audit.py \
  --config_a configs/self_audit_annotation.yaml \
  --config_b configs/self_audit_auditor.yaml \
  --config_c configs/self_audit_joint.yaml \
  --output_dir weights/self_audit_full
```

After training and ACDC-only threshold selection are complete, external testing
is a separate command:

```bash
python scripts/evaluate_external_mnms.py \
  --config configs/self_audit_acdc_to_mnms.yaml \
  --checkpoint weights/self_audit_full/phase_c_best.pt \
  --data-root preprocessed_data/mnm \
  --split testing \
  --tau-accept 0.0 \
  --device cuda \
  --output reports/external_mnms.json
```

Nothing in `train_self_audit.py` imports, opens, validates, or reports M&Ms.
This separation makes accidental external-test tuning visible and prevents an
external-data failure from corrupting an otherwise completed training run.

## Error Handling

Both workstreams fail explicitly rather than silently degrading:

- A malformed transition summary raises with the expected and actual shape.
- A GPU tensor reaching the long-lived accumulator is copied to CPU at the
  update boundary; no dense local tensor is accepted by the accumulator.
- Missing or non-finite delta values follow the existing metric validation
  policy.
- Missing M&Ms directories list the resolved absolute path.
- Missing volume-mask partners list up to five offending case identifiers.
- Unexpected mask labels report the label set and configured mapping.
- Use of a two-class config or `mnm_binary` is rejected.
- A checkpoint/config architecture mismatch fails before inference.
- An empty testing cohort does not produce a report with `nan` headline
  metrics; it fails.
- ED/ES fields are absent or explicitly unavailable, never fabricated.

## Testing Strategy

### Streaming metric tests

- Compare streaming and legacy metric values on deterministic synthetic
  predictions, including neutral transitions and empty ranking classes.
- Verify exact on-policy, synthetic, combined, and legacy-flat namespace
  behavior.
- Verify combined metrics equal a merge of on-policy and synthetic summaries.
- Process many synthetic validation batches and assert that retained local
  state remains a fixed `3 x 3` matrix rather than growing with image count.
- On CUDA when available, assert that full validation completes and that live
  allocated memory after batch warm-up does not grow linearly with loader
  length. This is an optional hardware test, not the sole regression test.

### M&Ms flow tests

- Dispatch ACDC and M&Ms configs to the correct dataset classes.
- Discover paired `.npy` data in `training`, `validation`, and `testing`
  aliases.
- Reject missing roots, empty splits, pair mismatch, shape mismatch, unknown
  labels, bad mappings, and two-class configs.
- Verify patient identity extraction from `<patient>_tXX` without inventing a
  phase.
- Verify the evaluator loads `testing` only and never constructs an optimizer,
  scheduler, training loader, or threshold sweep.
- Verify the report records independent evidence, checkpoint binding, cohort
  signature, label mapping, metric space, and unavailable phase partition.
- Run a CPU smoke test on a tiny temporary four-class M&Ms fixture.
- Run the current project test suite after both workstreams land.

## Acceptance Criteria

1. Phase-B full validation completes on the target 11.61 GiB GPU without the
   reported concatenation OOM and without a batch cap.
2. Existing Phase-B metrics and checkpoint selection match the pre-change
   implementation on a small deterministic fixture.
3. GPU-resident validation state is bounded by the current batch rather than
   the full validation cohort.
4. A `dataset: mnms` config creates `MNMSDataset`, never `ACDCDataset`.
5. The server testing split is validated as 272 paired four-class cases at run
   time; different deployments report their discovered count without a code
   change.
6. External evaluation loads only the frozen Phase-C checkpoint and the M&Ms
   testing split.
7. No M&Ms metric changes checkpoint selection or `tau_accept`.
8. The external report makes no native-geometry or ED/ES claim unsupported by
   the available preprocessed metadata.
9. No dataset file is added to Git.
