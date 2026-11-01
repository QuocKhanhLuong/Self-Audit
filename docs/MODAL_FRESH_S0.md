# Fresh ACDC S0, independent of historical teacher recovery

**Current transfer route:** [direct Drive API runbook](MODAL_DRIVE_API_RUNBOOK.md).
It supersedes the rclone execution commands below for the current request.
The earlier commands remain historical documentation; do not rerun them or
rewrite the historical STAGING receipt. The preparation/S0 contract below is
shared by both routes.

This opt-in execution shell starts at commit `0605636`. It does not modify
`prepare_acdc.py`, the compact trainer, the historical diagnostic pins, or
`/assets/receipts/STAGING.json`. S1/S2 and historical diagnostics remain blocked
until the exact historical raw B8 payload and required runtime/cache are verified.
Their blockers never gate supervised S0.

## Data contract

`scripts/modal_fresh_s0.py` uses private Volume `selfaudit-assets` at `/assets`.
Only its CPU `prepare_data` function receives `selfaudit-rclone`. A temporary
0600 config outside the Volume is deleted on exit; subprocess logs never expose
credentials. CPU/GPU execution functions do not receive the Drive Secret.

The locked split SHA256 is
`b826eaf494760079aa559b0c0c12f7444951d779ebbb5ed291c0398f20bf835f`.
TRAIN80, VAL18, seed42, L8 = patient007/010/011/024/026/041/077/099.
The allowlist contains 202 files: 98 cine images, 52 frame images, 52 masks.
Only L8 and VAL18 reference frames are copied. U72 masks, patient055/076,
M&Ms and TEST50 payloads are excluded. Existing TEST IDs in manifest exclusion
metadata remain byte-preserved; they do not cause file reads.

The only download source is the observed `Dataset/ACDC/training` below
`RCLONE_REMOTE_NAME`. `--observed-inventory` uses the previously captured
`/assets/receipts/discovery/e7f5be8897f16d63.json` directory inventory and its
observed training folder ID, avoiding another recursive Drive listing. It still
downloads actual payloads with `rclone copy --files-from-raw` and verifies sizes
and full SHA256 afterward. It never substitutes raw data for prepared arrays.
Rate-limit retries are bounded. A failed namespace is retained as evidence;
choose a new namespace for a new preparation attempt, never overwrite receipts.

The unchanged `prepare_acdc.prepare(..., 8, 42, evaluation=True)` creates
fresh `train.json` and `development.json`. Shape/axes/affine, canonical labels,
frame-to-cine equality, split/payload overlap and hashes are checked.
`val18_fresh_native.json` is a separate geometry-enriched artifact. Its 18 image
and reference hashes must match the historical VAL18 payloads. No claim is made
that its manifest bytes equal historical `val18_native.json`.

`configs/fresh_s0/acdc_l8_historical_train.json` restores exact historical bytes:
its object comes from `data_provenance.manifest` in
`results/compact_acdc_l8_20261029/experiment_20261028/smoke_scalar_guard_omp2/final/MANIFEST.json`;
the key order is the one written by canonical `prepare_acdc.py`, with indent2
and a final newline. Its full SHA256 is exactly
`e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`.
This is metadata recovery, not recovery of any teacher weights.

`DATA_PREP.json` records exact equality or a complete parsed-manifest comparison
that permits only image/mask parent paths and serialization to differ. Payload
hashes, IDs, split metadata and all other fields must agree. No expected hash is
changed. A second, single-use CPU container reloads the committed Volume,
rehashes all prepared arrays, loads a TRAIN batch and checks VAL geometry before
issuing PASS. The historical STAGING receipt SHA remains checked and unchanged.

## Locked S0 recipe and cost boundary

`configs/fresh_s0/supervised_wide_seed42.json` explicitly locks the canonical
compact mode: random Kaiming initialization seed42, widths32/64/128/256,
crop160, labeled batch8, 10,000 supervised CE+foreground-Dice AdamW steps,
LR0.0003, weight decay0.0001. Encoder, decoder and head all update. Eight initial
steps carry the trainer's warm-start telemetry; method=supervised applies
supervision on **every** step. The existing image-only U sampling is retained
for schedule compatibility but receives no loss, labels, teacher or audit.
No pretrained/KD export is loaded. No compact EMA, DINO or cache is required.
Default FP32 and canonical deterministic-algorithms=false are explicit; CUDA
bitwise replay is not claimed. No augmentation/optimizer rule is replaced.

The CPU smoke uses this same wide architecture, crop and batch for two updates,
then verifies gradients **and actual weight changes** in all three modules.
Its weights never initialize S0. GPU initialization must match the locked random
initial tensor hash; source/config/data/Git identities must match the smoke.

One A100-40GB container has CPU hard limit2 and RAM hard limit16GiB, retries0,
max_containers1, idle timeout2s. The conservative rate bound is $3/hour versus
the observed whole-container rate below $2.33/hour. Hard function timeout is
at most7,200s, capped further by `min($15 - spent_this_round, credit_remaining)`
less a $2 CPU/storage/egress reserve. The user reported $30 credit on 2026-11-01.
Check current usage and other jobs before launch/resume; do not assume that
this balance persists or that independent jobs cannot consume it.

Every1,000 steps the I/O wrapper writes a hidden partial directory, verifies
payload/manifest SHA256, fsyncs, atomically renames, and commits the Volume.
The training code and update rule are unchanged. A read-only watcher prints
step/10000 and ETA. If the measured cost envelope cannot fit another checkpoint
segment, stop at the persisted boundary with `STOPPED_COST_BOUNDARY`.
The hard Modal timeout remains a backstop; an abrupt timeout can lose only the
work after the last committed checkpoint. It does not authorize a restart.

At step10,000: export, reload, copy full checkpoint and deploy export into
`s0_backup`, write `SHA256SUMS.txt`, and **commit before evaluating VAL**.
Then `run_two_phase_evaluation` freezes native predictions before opening VAL
references and reports Macro4, FG and class Dice. Backup here means a second
verified payload copy on the persistent private Volume, not a claimed Drive
backup. The A100 never receives Drive credentials. TEST50 is never evaluated.

## Commands (conda selfaudit-modal)

Run from a clean, committed isolated worktree. No data passes through the Mac.

```bash
conda activate selfaudit-modal
export S0_NS=/assets/fresh_s0/acdc_l8_s42_20261101_r4

# New preparation only: use a new namespace if this attempt already exists.
python -m modal run --detach scripts/modal_fresh_s0.py \
  --action prepare --namespace "$S0_NS" --observed-inventory

# Read/reverify prepared data in a fresh CPU container; no GPU allocation.
python -m modal run scripts/modal_fresh_s0.py --action verify --namespace "$S0_NS"

# These commands fail before GPU submission unless DATA_PREP and smoke pass.
python -m modal run --detach scripts/modal_fresh_s0.py \
  --action smoke --namespace "$S0_NS" --credit-usd 30 --spent-this-round-usd 0
python -m modal run --detach scripts/modal_fresh_s0.py \
  --action train --namespace "$S0_NS" --credit-usd 30 --spent-this-round-usd 0

python -m modal app list
python -m modal app logs APP_ID
python -m modal volume ls selfaudit-assets "${S0_NS#/assets}"
python -m modal volume get selfaudit-assets \
  "${S0_NS#/assets}/DATA_PREP.json" ./DATA_PREP.json
```

Supply actual remaining credit and spent-this-round on a resume; do not reuse
the launch balance. Only explicit checkpoint resume is supported, with full
optimizer/RNG/data-sampler state and total steps still10,000:

```bash
python -m modal run --detach scripts/modal_fresh_s0.py --action train \
  --namespace "$S0_NS" --credit-usd ACTUAL_REMAINING --spent-this-round-usd ACTUAL_SPENT \
  --resume "$S0_NS/s0_train/step_001000" --resume-sha256 FULL_MANIFEST_SHA256
```

Evidence lives under the namespace: `DATA_PREP.json`, `RAW_FILES.json`,
`CPU_SMOKE.json`, `S0_LOCK.json`, `S0_STATUS.json`, `s0_train/TRAIN_LOG.jsonl`,
`s0_train/step_*/`, `s0_train/final/`, `s0_export/`, `s0_backup/`, `s0_val18/`.
Treat a submit as RUNNING; only final persisted metrics justify COMPLETED.
Software/fixture checks never establish patient Dice or efficiency.
