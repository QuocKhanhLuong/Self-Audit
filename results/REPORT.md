# ACDC v2 execution report — real GPU diagnostics, patient outcomes pending

This is semi-supervised v2. Dense audit labels belong to the same TRAIN-patient
budget; no U/VAL/TEST mask supplies training targets, subset selection or gates.
No patient Dice or improvement is reported yet. All unexecuted runs remain
explicit in `EXPERIMENT_STATUS.csv`; header-only metric CSVs contain no results.

## Source and initialization

Fetched main: `637f38ced217fe1db2347a87924695533056d8f0`.
Isolated worktree: `/root/Self-Audit-acdc-v2-run2`.
Current training-source SHA256:
`858d8b31f473ef2362a3f8e977f540d3e9c93afe36f602ef916d4ce09356c30a`.
The GPU diagnostic receipts retain their actual earlier source digest; they
must not be relabeled as measurements from a later version.

User-authorized mirror:
https://huggingface.co/jaychempan/dinov3/blob/main/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth

Downloaded weight SHA256:
`21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`
(112,235,916 bytes). It matches the mirror LFS digest and the official source's
short hash `21b726bb`. A full independent official upstream digest is unavailable;
the official HF resolve endpoint returned 401 GatedRepo. This is a hash-pinned,
user-authorized mirror, not independent upstream certification. No random
fallback or license acceptance was performed.

Official source: `/workspace/self-audit-v2/external/dinov3`, clean at
`6876159a11b4df116f30f667f8c9888617df0751`. Strict official state-dict loading
succeeded. The resulting native model has 32,865,530 parameters. Local provenance
config: `/workspace/self-audit-v2/BACKBONE.json`.

## Data and locked split

Drive: `KhanhDrive:Dataset/ACDC` (extracted files, no ZIP).
Local raw data: `/workspace/self-audit-v2/data/raw/ACDC`.
Copy completed: **1,053 files, 7.269 GiB**, exit 0, using `--immutable`.
Raw log: `/workspace/self-audit-v2/logs/acdc-download.log`.
Separate remote checksum comparison: **PASS, all 1,053 files matched**.
Compact receipt: `logs/TRANSPORT.json`. Google Drive returned quota errors during discovery/transfer.

The original TRAIN80/VAL20 split fails local integrity: patient074 (TRAIN) and
patient076 (VAL) have identical raw cine bytes and float32 voxel hashes. The
complete inventory checked all 100 training-folder cines, exact ED/ES image/cine
bridges, shapes/affines and all 50 TEST folder IDs, opening **zero mask arrays**.
An inventory is not physical HD95 certification or complete GT validation.

`docs/RUNBOOK.md` explicitly prefers the existing reviewed locked split for this
copy. Recovered **metadata only** from archive commit
`924265f7df604d3dca84dda49578b1a0cf28d821`, path
`reports/astra_dinov3_4070_20261019/full50/checkpoints/epoch_0001/MANIFEST.json`:
`metadata.lineage.locked_splits`. No historical checkpoint weights were loaded.

The restored split is `splits/acdc_reviewed_locked_train80_val18.json`, SHA256
`b826eaf494760079aa559b0c0c12f7444951d779ebbb5ed291c0398f20bf835f`.
TRAIN IDs are exactly unchanged; VAL18 excludes patient055/patient076 as already
recorded in that reviewed metadata. TEST IDs are patient101..150. The original
split and all raw patients remain untouched. The reviewed inventory passed before preparation; see
`logs/DATA_VERIFICATION_REVIEWED.json`. This reuses the established exposed VAL18 protocol; it is
not an unseen confirmation cohort.

The Drive `testing/` folder contains `_gt` filenames. Their provenance and
permission for scoring are unverified. TEST inference code never discovers or
opens these masks, and never computes TEST Dice.

## Measured GPU diagnostics and batch factor

RTX PRO 4000 Blackwell: 24,467 MiB; driver 595.71.05. Python 3.12.15,
PyTorch 2.14.1+cu132. CUDA tensor execution passed. Existing jobs were preserved;
initial utilization alone did not establish exclusive access.

On a real native ACDC slice from patient007, inside L8 subset seed42, supervised
encoder/detail gradients were nonzero and a decoder JVP was finite. No optimizer
or EMA update was performed. The repeated slice was 224×222; these are single
memory probes, not actual training throughput or patient efficacy measurements.

| Batch | Peak allocated GiB | Native supervised gradient + one decoder JVP |
|---:|---:|---|
| 1 | 0.42 | PASS |
| 16 | 4.34 | PASS |
| 24 | 6.46 | PASS |
| 32 | 8.52 | PASS |
| 48 | 12.75 | PASS |

Receipts: `logs/GPU_B*_JVP.json`, `GPU_BATCH_PROFILE.csv`.
Figure: `figures/gpu_batch_probe.png` / `.pdf`.
Actual transaction memory additionally includes optimizer states, teacher,
multiple candidate directions, and functional trials; larger patient shapes can
increase it. **Batch16 is the initial real pilot/full-run recipe, not a claimed
24-GB optimum.** OOM is recorded without silently changing the backbone/scope,
threshold, image grid or batch.

Batching is now supported; legacy configs default to batch1. Each scheduled
supervised/audit/U patient supplies its own batch of native slices. Supervised
and audit patients remain distinct. Metadata retains every sampled frame/slice.
Audit Brier gives equal mass to sampled slices within its patient, CE averages
valid batch pixels, Dice uses batch-wide present classes, and pseudo candidates
share the accepted-pixel denominator. This is a declared batch-size factor;
all matched configurations use batch16. It is not pooled with old batch1 results.
These are ordinary tensor batches, not gradient accumulation.
The guard timeline and direct audited commit remain unchanged.

## Preparation, training and evaluation queues

Preparation waits for all 1,053 remote checks plus reviewed local inventory PASS,
then creates nested 4/8/16-patient caches for seeds 42/123/2026, with L8 first.
Audit labels count inside each budget. Only L8_S42 prepares separately authorized
VAL references into `development.json`; the trainer consumes only `train.json`.
Additional caches reuse exactly verified normalized image arrays through
`--image-cache`; each subset writes only its own selected TRAIN masks. A prior
cache's masks are never borrowed. Failures preserve incomplete/new artifacts.
**All nine caches completed** at 2026-10-20 08:39:40 UTC. Actual labeled
frame/slice counts and manifest hashes are in `DATA_PREPARATION.csv`. L8/S42
has 16 labeled frames and 154 labeled slices; all nine preparations report zero
unlabeled-mask reads. No patient data were deleted or overwritten.

Queue order:
1. Real 32-step head and 32-step decoder pilots, batch16; verify gradients,
   patient separation, label access and final checkpoint receipts.
2. L8 subset seed42, 10,000 steps each: supervised, supervised_extra, ema_full,
   ema_block, joint_guard, independent_guard, sign, joint_no_class, joint_no_guard.
3. Freeze final checkpoint hash, score exposed VAL18, export figures after each
   completed run. No VAL-based checkpoint selection or threshold tuning.
4. Image-only TEST inference is queued after all nine priority-arm training runs,
   preserving NIfTI prediction geometry and exporting overlays. No TEST metric.

The 81-config matrix retains all other budgets/subset seeds as pending; the
current execution queue covers nine L8/seed42 arms only. All use training seed42;
three subset seeds are not three training initialization seeds. Exact resume is
not supported. Existing run directories are never silently reused. Disk checks
fail closed before storing full frozen probability volumes. No baseline latency
advantage is claimed from instrumented timings. `wall_gpu_hours` is timed
training-loop/final-save wall time; it excludes startup/data hashing and is not
a full billed GPU-reservation cost measurement.

`collect_acdc_results.py` updates compact CSVs from completed receipts only;
with one completed subset seed, across-seed std remains blank. Patient Dice is
mean of available reference-frame Dice; IoU/precision/recall and confusion use
pooled ED/ES frame counts, explicitly named in the CSV. Absent GT classes keep
undefined Dice/IoU and explicit FP counts. HD95 remains unimplemented/uncertified.
The figure renderer exports overlays, probability/confidence/entropy maps,
patient Dice boxplots, confusion, TRAIN loss and acceptance/cost curves. No VAL
learning curve exists under final-only checkpoint scoring.

## Checks and monitoring

`python scripts/check_repo.py`: PASS.
`python -m pytest -q`: **56 passed**, one PyTorch TorchScript deprecation warning.
Manufactured tests cover batch sampling, all nine batched decoder arms, guard
risk checks, matched schedules, U-mask isolation, shared-image cache verification,
explicit duplicate quarantine and IoU. They are software evidence only.

Existing session: `ssh_tmux`; selected live display: **`v2-live`** (step, loss,
gradient norm, peak VRAM and pseudo status). Relevant new windows: `self-audit-v2` (download),
`reviewed-data`, `v2-pilots-B16`, `v2-L8-full`, `v2-results`, `v2-test-pending`.
Local logs: `/workspace/self-audit-v2/logs/`.
Local patient caches/runs/evaluation/TEST exports remain outside Git; only compact
reports, source changes and non-patient diagnostic figures are committed.
Both real patient 32-step pilots completed. Head: 12.09 s, 5.70 GiB peak;
decoder: 14.46 s, 11.76 GiB peak, **13 actual training decoder JVPs**.
Both used all eight supervised/audit patients, 32/72 U patients, 32 supervised
updates and 32 EMA updates, and read no labels outside L8. A 32-step pilot does
not certify full U-patient coverage. Both saved hash-verified final checkpoints.
Head manifest SHA256: `c9d2baa1a40ac0abad996d3d4c3ca5918b97ac07cecf622e56a7ea71cce42006`.
Decoder manifest SHA256: `2d09dc1dddd514830c5bc2c8158cc73ed90bfbd70818dfb5f4ffa9ebc3161a31`.
Both had **zero accepted pseudo transactions**: 11/24 post-warmup transactions
had no candidates; the remaining candidates received no applied update. This
is a stall diagnostic, not evidence of medical benefit. No finite trial was
rejected because no positive-coefficient displacement reached a trial.
Receipts: `PILOT_SUMMARY.csv`, `logs/pilot_B16_*_REPORT.json`; TRAIN curves:
`figures/real_patient_pilots.*`.

The real supervised L8/S42 batch16 10,000-step baseline started at
**2026-10-20 08:37:57 UTC** in `v2-L8-full`; it is RUNNING, not completed.
Remaining priority arms are queued. No frozen full-run VAL score, across-seed
mean±std or efficacy conclusion exists yet. Pending background actions are not
successes.

ACDC required citation: O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.,
“Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation
and Diagnosis: Is the Problem Solved?”, IEEE TMI 37(11), 2514–2525 (2018),
doi:10.1109/TMI.2018.2837502.
