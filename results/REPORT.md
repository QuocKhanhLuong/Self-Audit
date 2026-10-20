# ACDC v2 — crop256/batch52, 24-GB execution

The active training protocol is now **random crop 256×256, ordinary batch52**,
FP32, identical across all matched methods. This replaces native-size batch16
as a new declared experimental factor. VAL/TEST inference remains full native
cine; no evaluation crop, resizing or TEST tuning is introduced.

## Measured execution

A real 72-step decoder-scope pilot completed with original pretrained encoder
initialization, seeded native decoder and fresh optimizer/EMA state:

- 72 supervised updates and exactly 72 EMA updates; all eight labeled/audit and
  all 72 U patients visited. No label access outside L8.
- 158.85 seconds timed training/final-save wall time; peak allocated
  23,915,954,176 bytes (22.27 GiB). This excludes startup/data hashing from timing.
- 64 actual decoder JVPs, four accepted pseudo transactions; 14 post-warmup
  transactions had no candidates. This is TRAIN optimization evidence, not
  patient accuracy or medical benefit.
- Final checkpoint manifest SHA256:
  `fc86faa41930a7a00754713b51269539568a35850fc0e5fe8d3bd6b349e3a625`.
- Receipt: `logs/PILOT_C256_B52_72_REPORT.json`.

During the real pilot, `nvidia-smi` measured **23,752/24,467 MiB (~97%)** with
100% GPU utilization. Memory varies by method and phase; the same batch is held
fixed for the comparison. No dummy allocation fills unused VRAM.

The fresh supervised L8/S42 crop256/batch52 **10,000-step run started at
2026-10-20 08:54:24 UTC** and is RUNNING. The eight other L8/S42 arms are queued
behind it. No full-run checkpoint, frozen VAL score or TEST Dice exists yet.
The other 72 matrix configurations (budgets/subset seeds) remain unexecuted.

The earlier native batch16 baseline was stopped at logged step1515 in response
to the requested VRAM/protocol change. Its step1000 checkpoint and all logs
remain at `/workspace/self-audit-v2/runs/L8_S42_supervised_B16_10000`, with pinned
source worktree `/root/Self-Audit-acdc-v2-run2` at `970f142`. See
`logs/STOPPED_NATIVE_B16.json`. It was not completed or exactly resumed.
Previous native pilot receipts/figures are retained with their original source
hashes, separate from the new crop experiment. Their steps, batches and scopes
must not be used to claim a crop-related efficacy improvement.

## Crop and padding contract

Coordinates are drawn only from image dimensions. Each supervised/audit image
and its budgeted reference share the crop. Each U crop is fixed before the clean
EMA and stronger intensity-only view are made. No U mask, ED/ES identity, label
boundary or reference-derived ROI selects a crop.

Images smaller than 256 use centered edge padding. Reference padding is IGNORE
255; CE, Dice and audit Brier exclude it. A separate image-geometry validity
mask excludes padded U pixels from confidence acceptance and the shared pseudo
class denominator, even at a zero confidence threshold. The positive/negative
audit weighting and absent-positive reporting remain explicit. This crop/padding
objective is a new factor, applied equally to all arms. All original dense TRAIN
patient annotations, including audit use, still count in the same 4/8/16 budgets.
Legacy `crop_size:0` keeps native behavior; no implicit resize was added.

## Capacity measurements and source

`CROP_VRAM_PROFILE.csv`, `logs/transaction_C*.json` and
`figures/crop256_vram.*` contain real full-transaction memory diagnostics. These
include a supervised AdamW/SGD step, real EMA targets, candidate gradients,
decoder JVPs/functional trials and one EMA update, with real ACDC inputs.

Four-class joint-guard crop256 probe results:

| Batch | Peak allocated GiB | Peak reserved GiB |
|---:|---:|---:|
| 48 | 20.62 | 21.02 |
| 52 | 22.32 | 22.79 |

Batch52 also passed full-EMA and a small-image/padding case. Batch54 passed
a four-candidate probe too (21.96 GiB allocated, 22.55 GiB reserved); observed
peaks were not monotonic across these individual minibatch/process probes.
Batch52 was retained after its successful 72-step actual training pilot, using
nearly all available VRAM with a small margin. This is not a proof of the
absolute maximum batch on every future state or a throughput comparison.

Capacity probes load the hash-pinned native-v2 step1000 checkpoint only to
exercise realistic learned four-class EMA candidates, with fresh optimizer
states. They are diagnostics, not an exact resume or a patient outcome. The
actual 72-step pilot and all full experimental arms initialize afresh from the
original pretrained encoder and seeded native decoder, never those probe states.

Current worktree: `/root/Self-Audit-vram24`.
Training-source SHA256:
`6e12885e44b725c132a01437dc6179cbf524b94c51dd5a5f5254701eb78cf442`.
Official DINO source clean at `6876159a11b4df116f30f667f8c9888617df0751`.
User-authorized mirror weight SHA256:
`21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`.
The full digest matches mirror LFS metadata and the official short hash; an
independent full official upstream digest remains unavailable. Strict official
state-dict loading passed. No random fallback or license acceptance occurred.

## Data and evaluation boundaries

Raw Drive: `KhanhDrive:Dataset/ACDC`.
Local data: `/workspace/self-audit-v2/data/raw/ACDC`.
All **1,053 files / 7.269 GiB** copied and matched the remote checksum check.
All nine nested caches (4/8/16 × subset seeds42/123/2026) completed, with actual
labeled frames/slices and hashes in `DATA_PREPARATION.csv`. L8/S42 has 16 labeled
frames and 154 labeled slices; U-mask array reads during preparation were zero.
Full normalized images are shared after exact verification; masks remain specific
to each budget/seed. Crops are made at training time, not by altering caches.

The original TRAIN80/VAL20 split fails due to identical patient074 (TRAIN) /
patient076 (VAL) cines. Following the current runbook, the existing reviewed
TRAIN80/VAL18/TEST50 metadata was restored from archive commit
`924265f7df604d3dca84dda49578b1a0cf28d821`, manifest path
`reports/astra_dinov3_4070_20261019/full50/checkpoints/epoch_0001/MANIFEST.json`,
field `metadata.lineage.locked_splits`. Only metadata was reused; no historical
trainer or checkpoint weights were loaded. TRAIN is unchanged, and reviewed
excluded IDs patient055/patient076 remain untouched on disk.

Reviewed split: `splits/acdc_reviewed_locked_train80_val18.json`, SHA256
`b826eaf494760079aa559b0c0c12f7444951d779ebbb5ed291c0398f20bf835f`.
Reviewed local inventory passed; all 100 cines and exact image-frame bridges were
checked without opening reference masks. This does not certify physical HD95.
VAL18 is development-exposed, not independent confirmation.

The Drive testing folder includes GT filenames of unverified provenance. TEST
inference never discovers/opens these masks and never computes TEST Dice. It is
queued after all nine priority training arms finish. Frozen final checkpoint
scoring and figure export follow each completed full run; no checkpoint/threshold
is selected using VAL/TEST. HD95 remains unimplemented/uncertified.

## Monitoring, tables and checks

Active session/window: **`ssh_tmux:v2-live-crop256`**. Other new windows:
`crop256-pilot`, `crop256-L8-full`, `crop256-results`, `crop256-TEST-pending`.
Outputs/logs: `/workspace/self-audit-v2/vram24/`.
Exact queued commands are in `configs/PILOT_QUEUE.sh`, `PRIORITY_QUEUE.sh` and
`TEST_INFERENCE_QUEUE.sh`. Existing unrelated sessions/jobs were not interrupted.

The compact CSV collector is configured for suffix `_C256_B52_10000`. It reports
only measured receipts; mean/std across subset seeds remain unavailable until
those seeds actually complete, and one-seed std stays blank. Equal optimizer
steps do not imply equal compute; timings exclude initialization and data hashing.
All arms currently use initialization seed42; three label-subset seeds are not
three initialization seeds. No patient accuracy is inferred from tests.

`python scripts/check_repo.py`: PASS.
`python -m pytest -q`: **58 passed**, one TorchScript deprecation warning.
Tests cover aligned/GT-independent crops, ignored padding in pseudo normalization,
all nine cropped batched arms, matching schedules, guard checks and no U mask
access. Patient data, large checkpoints, credentials and raw logs stay outside Git.

ACDC citation: O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al., IEEE TMI
37(11), 2514–2525 (2018), doi:10.1109/TMI.2018.2837502.
