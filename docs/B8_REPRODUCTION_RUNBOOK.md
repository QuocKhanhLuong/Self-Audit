# Historical full-model B8 reproduction and backup

This run is a new reproduction, because the historical payload
`0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6`
is unavailable. It does not inherit that payload's quality or identity.

The historical recipe is committed under
`results/compact_acdc_l8_20261029/experiment_20261028/b8_recovery/full/CONFIG.json`.
Its execution source is `0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca`, closure
`3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7`.
An isolated branch adds checkpoint atomicity and same-run resume to that
historical `train.py` / `self_audit.engine`; it must not use current compact
trainer defaults. The patched source commit and closure are separate provenance.

The budget is one seed-42 ACDC L8/U72 run, 10,000 total steps including 1,000
supervised warm-up steps, batch 8, crop 256, window none. Both supervised and
pseudo steps update the full model. The config's `update_scope: head` is not the
scope of `ema_full`: body AdamW owns all non-head trainable parameters, and head
SGD owns the head. No scheduler or mixed precision is introduced. Decoder,
detail, fuse and head use the historical constructor initialization. The same
optimizers perform supervised and accepted pseudo steps; EMA updates once after
the final student state. Original augmentation, sampling and loss denominators
are retained, including historical audit sampling RNG draws.

Labels are exactly patient007, patient010, patient011, patient024, patient026,
patient041, patient077 and patient099. All other locked TRAIN patients provide
images only. No M&Ms assets are required for this run.

`scripts/run_b8_reproduction.py` takes a JSON plan and its SHA256. The plan pins
the recipe, source commit, train manifest, pretrained files, evaluation assets,
and backup provenance. Only the pretrained source and weight paths may differ
from the archived recipe. Run its preflight before the training launch:

```bash
python scripts/run_b8_reproduction.py --plan /absolute/PLAN.json \
  --plan-sha256 ACTUAL_PLAN_SHA256 --preflight-only
python scripts/run_b8_reproduction.py --plan /absolute/PLAN.json \
  --plan-sha256 ACTUAL_PLAN_SHA256
```

Use the exact plan recorded in the server run, not the placeholders above. Run
the second command inside a dedicated tmux session. The runner holds a process
lock, refuses to restart a previously started run from scratch, and records
`TRAIN_STARTED.json`. After interruption, the same command with `--resume`
continues the latest verified `step_*` checkpoint in the same output directory.
Never use a new output directory, additional seed, batch, or step count to hide
an interruption or a poor result.

The executable ordering is training → atomic final checkpoint → raw-student
export/reload → SHA256 → weights/provenance upload → full remote-byte SHA256 and
size verification → VAL18 and TEST50 final-raw-student evaluation → report
upload and HANDOVER update. Failed upload gets bounded retries and
`UPLOAD_PENDING`; weights remain local. Retrying backup must not retrain.
The raw teacher export includes architecture/preprocessing metadata and pinned
source assets; it is not the independent compact deployment package.

Evaluation uses native inputs, FP32, batch 1, no threshold/TTA/postprocessing,
and frozen argmax predictions before opening evaluation references. Primary
Macro4 is the mean of the historical frame → patient/class → class means;
foreground mean and BG/RV/MYO/LV are separately reported. TEST50 was previously
exposed development data, and is used once for this final checkpoint comparison,
not tuning. Reference Macro4 is 0.8800492397526454, foreground 0.8412102397257094;
the already declared quality gate remains Macro4 ≥ 0.87, not a promised outcome.

After verified backup and teacher evaluation, the authorized next stage is the
four locked CPTU arms. They must explicitly bind the new teacher and cache
hashes and preserve the 2,000-step protocol; they cannot silently satisfy the
old immutable teacher pins. A failed teacher gate stops this expansion. Resume
reproducibility and CUDA verification remain required before patient CPTU runs.
No joint sweep is authorized.
