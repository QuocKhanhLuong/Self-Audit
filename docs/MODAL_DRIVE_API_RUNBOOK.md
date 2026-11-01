# Direct Drive API → canonical ACDC preparation → fresh S0

This is the current execution route. It uses no rclone executable or browser
cookies. MRI payloads travel directly from Drive into the private Modal Volume
`selfaudit-assets`, mounted at `/assets`. Source and receipts can be read on the
Mac; no patient image is downloaded to the Mac. The historical staging receipt
and its 17 verified files are rehashed and preserved, never rewritten as READY.

## Credentials and first real smoke

Use conda `selfaudit-modal`. Only `stage_smoke` and `stage_data` receive a Drive
Secret. Verification, the controller, CPU training smoke and A100 receive none.
`SELFAUDIT_DRIVE_AUTHORIZED_USER_JSON` takes precedence over the previously
granted token in `selfaudit-rclone`; the latter is parsed in memory without
calling rclone. Refresh is allowed only with its matching OAuth client. Missing
clients are never guessed. HTTP status/reason enums are logged, not response
messages, token strings or credential JSON.

When a new grant is necessary, obtain an OAuth **Desktop** client JSON for the
user's own Google Cloud project with Drive API enabled. The authorization helper
requests only `drive.readonly`, receives the loopback callback, writes credentials
outside the repository with mode0600 and creates a uniquely named Modal Secret.
The browser consent is the only interactive step; never paste a token into chat.
No MRI is handled by this local helper.

```bash
conda activate selfaudit-modal
python scripts/authorize_drive_readonly.py --client-json /absolute/path/client.json
# Read the non-secret `modal_secret` value printed by the helper.
export DRIVE_SECRET=selfaudit-drive-readonly-ACTUAL_TIMESTAMP
export S0_NS=/assets/fresh_s0/acdc_api_20261101_r2

python -m modal run scripts/modal_drive_api.py --action smoke \
  --namespace "$S0_NS" --secret-name "$DRIVE_SECRET"
```

The smoke pins `patient007_4d.nii`, file ID
`1KYy4ktn-9WtkCMpYHJwGIp0K0D2BTkWV`, to15,913,312 bytes. It uses
Drive API v3 `files.get`/`get_media` and `MediaIoBaseDownload`, writes `.part`,
checks source checksums when supplied, computes full SHA256, fsyncs, atomically
renames and commits. A separate single-use CPU container reads and rehashes it.
Check `API_SMOKE.json`: only PASS with `persistence_verified=true` permits bulk
transfer. Existing verified files are rehashed before reuse, without redownload.
Sidecar proofs are under `.verified/` so they cannot match the canonical NIfTI glob.
Per-file180s deadlines, HTTP25s timeouts and zero automatic API retries bound
quota/permission failures. A failure retains the partial evidence and blocks GPU.

Actual attempt on2026-11-01: app `ap-cprm3FlylRcbp5iDRNTXaT`, namespace
`/assets/fresh_s0/acdc_api_20261101_r1`, returned HTTP401 `authError` with the
previous access-only grant. All17 historical hashes matched. No data transfer,
preparation or GPU training completed. See `reports/modal_drive_api_20261101/`.

## One detached pipeline after authorization

Run from a clean committed isolated worktree. Check current credit, total spend
for this task and active Modal apps first. The earlier user report was$30 credit;
it is evidence of that time, not a live balance. Use actual remaining amounts.
The budget helper rejects insufficient credit and caps GPU timeout using
`min($15 - task_spend, remaining_credit)` with a$2 CPU/storage reserve and
conservative$3/hour whole-GPU-container bound. No new seed, extra arm or duplicate
training submission is authorized by a failed transfer.

```bash
python -m modal app list --json
python -m modal run --detach scripts/modal_drive_api.py --action pipeline \
  --namespace "$S0_NS" --secret-name "$DRIVE_SECRET" \
  --credit-usd ACTUAL_REMAINING_CREDIT --spent-this-round-usd ACTUAL_TASK_SPEND
```

The CPU controller remains remote after the client disconnects. It chains:
smoke → fresh-container readback → allowlisted transfer → canonical prepare →
fresh-container TRAIN batch/VAL geometry verification → CPU S0 smoke → one
A100-40GB S0. The namespace launch file refuses repeated launches; do not submit
concurrent apps against one namespace. Metadata failures never launch training.
The returned app ID/function call ID and `PIPELINE_STATUS.json` are execution
status, not a claim of completion.

Drive discovery is scoped to training folder
`15BKahUSkHyJv0tctCcavnoLS30YsvNY6`. For U patients the file query asks only for
the cine name. The202-file plan contains98cine images plus52reference frame
images and52masks, exclusively for L8 and VAL18. No U72 masks,055/076, M&Ms or
TEST50 are transferred/read. The complete split, preparation and comparison
contract is in [MODAL_FRESH_S0.md](MODAL_FRESH_S0.md). This route calls the same
`prepare_acdc.py`; it does not replace normalization, frame/axis handling or
canonical class mapping. `development.json` and the geometry-enriched
`val18_fresh_native.json` remain distinct artifacts.

`DATA_PREP.json` keeps the historical train hash
`e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4` fixed.
Only complete parsed-manifest equality after removing image/mask parent paths
and serialization is an allowed fresh-S0 witness; all payload hashes and other
fields must agree. An unexplained difference blocks GPU. The new receipt does
not modify `/assets/receipts/STAGING.json`.

## S0, eligible KD arms and backup

S0 is randomseed42, wide32/64/128/256, crop160, Lbatch8, exactly10,000 supervised
CE+foreground-Dice updates. The CPU smoke verifies encoder/decoder/head gradients
and weight changes, and locks initialization/source/config/data hashes. Its
weights are discarded. The production trainer and learning rule are unchanged.
Logs print every100 steps; atomic resume checkpoints and Volume commits occur
every1,000. A cost stop retains a checkpoint and requires explicit resume, never
a new initialization. At10,000, exported/reloaded weights and full checkpoint
are backed up with SHA256 and committed **before** final exported-student VAL18
evaluation. No TEST50 reference is opened. Report Macro4, FG, per-class Dice and
RV precision/recall from the persisted metrics, not from console estimates.

The exact S1/S2 configurations are `configs/fresh_s0/s1_plain_kd.json` and
`s2_plain_kd.json`:10,000 total steps including1,000 supervised warm-up,
Lbatch8/Ubatch8, with crop160/256 respectively. They preserve historical KD
loss, temperature, threshold, optimizer and teacher identity. They are **BLOCKED**
until the raw B8 payload `0ae593ec…`, its L8 lineage and the complete compatible
TRAIN-U cache are physically verified. The controller currently reports their
missing prerequisites; it does not launch KD. Even three matching top-level
files produce `REQUIRES_FULL_CACHE_AND_LINEAGE_PREFLIGHT`, never READY. A KD
concurrency profile/launcher is NOT_RUN and requires those assets first. The wide
KD export and reproduced B8 are never substitutes or S0 initializations.

The read-only grant cannot upload backup weights. The current route records
`DRIVE_BACKUP_PENDING` and keeps full weights on Volume. A separate CPU upload
job with an explicitly granted write credential and remote checksum verification
is required before claiming Drive backup; it is not silently implemented by a
successful Volume commit.

```bash
python -m modal app logs APP_ID
python -m modal app stop APP_ID
python -m modal volume ls selfaudit-assets "${S0_NS#/assets}" --json
python -m modal volume get selfaudit-assets \
  "${S0_NS#/assets}/PIPELINE_STATUS.json" ./PIPELINE_STATUS.json

# Only after an interrupted run, with current remaining credit/spend:
python -m modal run --detach scripts/modal_drive_api.py --action resume \
  --namespace "$S0_NS" --credit-usd ACTUAL_REMAINING_CREDIT \
  --spent-this-round-usd ACTUAL_TASK_SPEND \
  --resume "$S0_NS/s0_train/step_001000" \
  --resume-sha256 ACTUAL_CHECKPOINT_MANIFEST_SHA256
```

Checkpoints: `$S0_NS/s0_train/step_*/` and `final/`; resume output is under
`s0_resume_STEP/`. Export: `s0_export/`; persistent backup: `s0_backup/`;
VAL predictions/metrics: `s0_val18/`. Patient results remain NOT_RUN until real
training and exported-student evaluation finish. Fixture tests establish only
software behavior, not historical payload equivalence, Dice or GPU throughput.
