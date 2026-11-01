# Adjacent-slice diagnostic on historical VAL18

Status: implemented diagnostic, **patient inference NOT RUN**. The checked-in
configuration is an asset inventory/template: the historical payloads and
prepared VAL18 arrays are missing locally; `/REQUIRED/` runtime paths must be
resolved before inference. No backbone, loss, gradient bridge or adaptive encoder
is trained by these scripts.

The question is whether adjacent slices let the teacher correct errors that the
compact student still makes on the same pixel. The main table conditions on both
models being wrong at lambda=0, then reports teacher-only, student-only, both and
neither corrections. Report raw counts, fractions of all GT class voxels, and
fractions of common failures separately. A student already correct at lambda=0
is excluded from this table. Its context-induced regressions are reported in a
separate supplemental opportunity table, never added to the common-failure rate.

## Fixed protocol

- Historical B8 **raw student** payload `0ae593ec…1281f1b6`, historical wide
  compact export `32a35c1f…ab00380`. Full SHA-256 identities are in the config.
  The parent verifies the compact export's teacher identity and TRAIN/L8 lineage.
  The reproduced B8 payload is not an allowed substitute in this protocol.
- Exact native VAL18 manifest and locked split; no TEST50. Absolute image paths
  inside the pinned manifest must resolve, e.g. through a read-only data mount.
  Do not edit the manifest to silently alter normalization, masks or membership.
- Prepared NPY cine in TZYX, finite floating values in [0,1], no renormalization.
  Native spatial size, batch one, FP32, eval mode, no autocast, TTA or postprocess.
  Channels are adjacent axial slices at the same time, clamped at volume edges.
- Lambda=1 clones the original tensor exactly. Before any counterfactual output,
  each worker compares canonical context -> model with interpolated context ->
  model: exact input/argmax, logits atol=1e-6 and rtol=1e-5. This establishes path
  parity against the pinned model/context implementation, not equivalence to all
  external deployment wrappers or a cross-device reproducibility guarantee.
- Only after this pass, generate all lambda in {0,0.5,1}, where side channels are
  center + lambda*(side-center). Lambda=.5 is a sensitivity analysis; no monotonic
  response is assumed. Repeated-center inputs can be out of distribution.
- Freeze every prediction from both models, hash all outputs, then open the GT
  masks. Mask metadata selects the same predefined ED/ES reference frames for all
  methods; mask payloads never enter the prediction workers.
- Match the historical wide lambda=1 Macro4 of 0.777413392930114 to absolute
  tolerance 1e-4 (0.01 percentage point), fixed before scoring. On mismatch,
  write `REPRODUCTION_MISMATCH_DO_NOT_INTERPRET`, investigate the evaluator/data/
  runtime difference, and do not use context findings to justify training.

## Metrics and interpretation

`self_audit/context_metrics.py` emits frame records, pooled counts, patient records,
equal-patient rate summaries and leave-one-patient-out sensitivity. Primary Dice
matches the historical reduction: full frame-volume class Dice -> mean of
GT-present frames per patient/class -> mean patients per class -> Macro4/FG.
Absent GT class Dice is omitted even if FP exists; FP counts and rates remain
explicit. Ignored label 255 is excluded. Slice-based standard errors are invalid.

Report RV/MYO/LV rescues AND harms, confusion matrices, FP introduced/removed,
full-frame Dice, and edge/interior strata. The first/last axial slices form one
union (one slice is counted once). Leave-one-patient-out analysis is a stability
check, not independent validation or a confidence interval. Report lambda=.5 and
lambda=1 without choosing a winner using VAL.

The diagnostic is descriptive on exposed development patients. Teacher-only fixes
identify a potential transfer opportunity; they do not show that response KD can
recover it. No automatic clinical or training gate is inferred from a positive
table. If subsequent response-KD is warranted, compare two-view KD against exactly
the same two-view KD plus response loss, before adding architectural changes.

## Runtime and execution

The current source tree differs from the historical checkpoint source digest.
Do not bypass that check. The parent launches each model in a separate process
whose import path selects its pinned historical runtime. The teacher retains the
strict checkpoint loader; compact retains its export/model-file hash check.
Required third-party backbone source and paths stored in the teacher checkpoint
must also exist; an asset-only preflight does not prove successful model loading.
Pin config bytes with SHA-256 after resolving paths/device. Relative paths are
relative to the repository working directory, not the config's directory.
Before inference, the parent also freezes `PROTOCOL.json`: hashes of the worker,
orchestrator and scorer plus the current Python/PyTorch/NumPy versions, executable,
device properties and numerical settings. Both workers must match this digest,
and scoring verifies it again. This records and locks the diagnostic runtime; it
does not claim to reconstruct the historical hardware/software environment.

```bash
.venv/bin/python scripts/run_context_diagnostic.py \
  --config configs/context_diagnostic_historical_val18.json \
  --config-sha256 <actual-config-sha256> \
  --preflight-only --out <new-preflight.json>
```

Exit 2 means assets are missing, with an explicit list and no inference. After a
successful preflight and compute setup, the tracked diagnostic run uses the same
command without `--preflight-only`, with `--out` naming a new output directory.
It runs teacher then compact serially, so both do not occupy device memory
simultaneously. A run must use an approved backend and pinned dependencies; do
not start an ad-hoc patient job to bypass the project's run tracking.

Outputs: `PROTOCOL.json`, worker logs/inventories, global `PREDICTIONS_FROZEN.json`,
`FRAME_RECORDS.json`, `RESULT.json`, `frame_transitions.csv`, `frame_metrics.csv`,
`patient_transitions.csv` and `summary.csv`. The result contains per-patient and leave-one-patient-out
summaries. Validation failure stops the process rather than silently dropping
patients. Existing output directories are never overwritten.

Dependencies follow `requirements.txt`; use a private worktree environment.
Synthetic software checks are not patient results. Tests, source SHA and missing
asset evidence belong with the eventual diagnostic handoff.
