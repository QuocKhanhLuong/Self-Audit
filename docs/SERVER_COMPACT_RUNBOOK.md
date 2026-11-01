# Server compact S0 runner

This is a new, opt-in experiment using `self_audit.compact_training.train_compact`.
It leaves the existing fresh S0 recipe unchanged. The runner supports one
10,000-step supervised S0 per invocation, with random Kaiming initialization,
seed 42, widths `[32, 64, 128, 256]`, crop 160, and no pretrained weights. It
does not launch a seed or arm sweep.

Use the repository runtime/test dependencies plus `nibabel`, `psutil` and
`tqdm`; `rclone` must be installed on the server and configured outside Git.
Record the actual Python/PyTorch/CUDA versions and `pip freeze` with each run.
The progress dependency is checked before training begins.

## Profile and lock

Profile the canonical trainer at B8/B32/B64 on the exact TRAIN fixture intended
for the run. The short profile excludes its configured warmup steps from latency
statistics. It reports median/p95 step latency, labeled and sampled L+U slices
per second, and peak allocated/reserved CUDA bytes. A batch OOM is retained as an
`OOM` result; it cannot be selected by the lock.

```bash
python scripts/run_server_compact.py profile \
  --train-manifest /workspace/data/train.json \
  --train-sha256 TRAIN_SHA256 \
  --device cuda --warmup-steps 3 --measured-steps 10 \
  --out /workspace/server_compact/profile
```

For software-only setup, `--synthetic` without a manifest generates a retained
manufactured TRAIN fixture. That profile is not patient, GPU, or quality
evidence.

After preparing exact development manifests, lock B32 or B64 once. The lock
includes the profile receipt, Git commit, every `self_audit/**/*.py` file,
runtime scripts/configs, base recipe, exact config, and TRAIN/VAL/optional TEST
paths and hashes. Training uses the selected B32/B64; native VAL and optional
TEST use a separate pinned `eval_batch_size=8` across all arms to bound full-resolution
inference memory. This is pinned in both the runner config and protocol lock.
TRAIN and VAL patient IDs must be
disjoint; optional TEST IDs must be disjoint from both. No manifest can be
silently substituted after the lock is written.

```bash
python scripts/run_server_compact.py lock \
  --profile /workspace/server_compact/profile/PROFILE.json \
  --profile-sha256 PROFILE_SHA256 \
  --train-manifest /workspace/data/train.json --train-sha256 TRAIN_SHA256 \
  --val-manifest /workspace/data/val.json --val-sha256 VAL_SHA256 \
  --test-manifest /workspace/data/test50.json --test-sha256 TEST_SHA256 \
  --batch-size 64 \
  --out /workspace/server_compact/protocol_lock.json
```

Omit both TEST arguments to leave TEST unconfigured. The selected batch is shared
by every later comparison using this lock; this CLI itself runs only supervised
S0. It does not expose loss, crop, seed, step-budget, or batch overrides after
locking.

## Run and resume

Use an existing, listable rclone parent path. At step 10,000 the runner writes
the terminal full checkpoint, exports and reloads its raw student, captures the
current best raw export and `best.pt`, and backs those weights up before doing
the final VAL evaluation. Each upload uses `rclone copyto`, then streams the
remote bytes through `rclone cat` and compares byte count and SHA-256. Rclone
operations also have a hard 900-second total process deadline; `--timeout`
remains an inactivity setting. Stderr and command arguments are omitted from
logs so credential-bearing error text is not persisted. Local files are kept.
Retries are bounded; `UPLOAD_PENDING` blocks deferred evaluation until backup
is verified.

```bash
python scripts/run_server_compact.py run \
  --lock /workspace/server_compact/protocol_lock.json \
  --lock-sha256 LOCK_SHA256 \
  --out /workspace/server_compact/run_s0 \
  --device cuda \
  --rclone-destination drive:server-compact-runs
```

The canonical trainer writes a per-step `TRAIN_LOG.jsonl` and exact-resume full
checkpoints every 1,000 steps. A read-only `tqdm` watcher displays ETA plus
`L-patient coverage epoch`: one coverage cycle completes only after every locked
TRAIN-L patient has appeared in the supervised sample schedule. This is a
patient-coverage equivalent, not a fixed-size image epoch. The watcher does not
read labels or consume training RNG.

If interrupted after a saved checkpoint, use the exact path and SHA emitted in
`RUN_STATUS.json`; the runner accepts only the latest checkpoint and requires a
new output segment. It never starts a replacement run over an existing output.

```bash
python scripts/run_server_compact.py run \
  --lock /workspace/server_compact/protocol_lock.json --lock-sha256 LOCK_SHA256 \
  --out /workspace/server_compact/run_s0 --device cuda \
  --rclone-destination drive:server-compact-runs \
  --resume-checkpoint /workspace/server_compact/run_s0/segments/segment_0001/step_004000 \
  --resume-sha256 CHECKPOINT_MANIFEST_SHA256
```

Each periodic checkpoint is exported and reloaded before canonical two-phase
VAL evaluation. The terminal raw export and pre-evaluation weights are backed
up before the final terminal VAL evaluation. Predictions are frozen before VAL
masks are read. Highest VAL
`class_first_macro4` wins; an exact tie keeps the earliest step. VAL is used only
for development checkpoint selection, with no acceptance threshold or training
gate. Each improved selection atomically publishes a real `best.pt` and
`BEST_SELECTION.json` with architecture and preprocessing metadata. Completion
also writes `best_export/`, `BEST.json`, and `FULL_RESUME_CHECKPOINT.json`.

Optional TEST scoring uses only the frozen selected export after best selection,
verified VAL prediction freeze, core backup, and successful post-VAL backup.
The BEST receipt stores the exact original VAL prediction path and SHA-256; the
runner verifies the frozen receipt, VAL manifest, selected export, and complete
patient coverage before TEST. TEST metrics are reported after selection and do
not alter the chosen model or any gate. After TEST the runner updates
`TEST_RESULT.json`, `SUPPLEMENTAL_EVALUATION_REPORT.json`, and `SHA256SUMS.txt`,
then verifies those reports, logs, config, provenance, and evaluation artifacts
in a final upload phase.

If the run reports `UPLOAD_PENDING`, keep the run directory and retry the same
remote base:

```bash
python scripts/run_server_compact.py backup \
  --run-root /workspace/server_compact/run_s0 \
  --rclone-destination drive:server-compact-runs
```

The backup plan pins the unique remote child and exact local file hashes. A retry
reuses that same destination and refuses changed local inputs or a different
remote base. If the initial core upload is pending, retry uses the exact
`final_checkpoint` path and `final_checkpoint_manifest_sha256` from
`RUN_STATUS.json`, finishes deferred final VAL, then continues post-VAL backup
and optional TEST without retraining. If a later phase is pending, retry resumes
that phase and preserves the scored TEST artifacts.

```bash
python scripts/run_server_compact.py backup \
  --run-root /workspace/server_compact/run_s0 \
  --rclone-destination drive:server-compact-runs \
  --device cuda
```

## Evidence boundary

`PROFILE.json` describes only the bounded profile. `TRAIN_LOG.jsonl`, checkpoint
manifests, VAL frozen-prediction receipts, `PROVENANCE.json`, `SHA256SUMS.txt`,
and run status bind the actual run artifacts. Synthetic fixtures verify
software behavior only. A server profile
does not establish patient quality, and VAL is development data rather than
confirmation. No GPU or patient training was launched as part of implementing
this runner.
