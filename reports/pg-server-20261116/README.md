# PG K8 server setup, 2026-11-16

User-authorized scope: download ACDC and M&Ms from the supplied Drive folder,
prepare M&Ms separately, and run complete ACDC PG K8 shared S1/S2/S3 followed
by A4, A3, A2, A1. Budgets, fixed patient split and L8 remain the PG request's
values. The original code-only delivery restriction is superseded by this
explicit server-training request. M&Ms training is not part of this run.

## Machine and paths

- SSH alias `vast-gpu`, host `ssh8.vast.ai`, port `10937`, hostname
  `ae8a6d2632f0` (live checks on 2026-11-16).
- GPU NVIDIA RTX PRO 4000 Blackwell, 24,467 MiB; PyTorch 2.14.1+cu132.
- New checkout `/root/Self-Audit-pg-20261116`; inspected base
  `0798b46738e335bb7e8075a966de2362bf1e16f4`.
- Dataset root `/root/dataset`; download/geometry/preparation receipts are
  separate. Raw NIfTI and original ZIP containers remain unchanged.
- Run `/root/mount_data/runs/pg-k8-20261116`.
- One tmux session `pg-k8`, one window `pipeline`, with training, ACDC and
  M&Ms log panes. Attach using `ssh -t vast-gpu tmux attach -t pg-k8`.

## Integrity and source preparation

Drive inventory: ACDC 1,053 files / 7,805,292,667 bytes; M&Ms 3 ZIP files /
15,380,572,339 bytes. Downloads require size and Drive MD5, also SHA256 where
present; local SHA256 is recorded for every file. ZIP members pass CRC.
Rclone's shared API client was throttled; direct public-file transport and
bounded HTTP ranges recover transfers by the same inventoried Drive IDs.
Interrupted bytes are preserved separately, never treated as complete data.

Official sources are pinned in `SOURCE_SETUP.md`. CineMA source GitHub SHA
and Hugging Face model SHA are distinct. A strict full checkpoint load and
one actual CUDA forward passed on the server. This is GPU software evidence,
not training or patient efficacy.

The fresh ACDC materialization keeps historical manifest hashes as ancestry,
records actual new manifest/source-lock hashes, and independently checks the
exact split, L8 identity, original mask hashes, native geometry and raw Drive
inventory. No TRAIN non-L8 mask is a training target.

## S1 source-view implementation resolution

Before training, resolve CineMA's source-only 192×192×16 view with policy
`image_center_no_gt_v1`: after source spacing conversion, centre each axis
when larger than the target, otherwise end-pad. Record the crop start and
size; the existing backprojection computes its affine and support mask.
The centre uses image dimensions and spacing only, never an LV mask or a
pseudo-label. Student input remains the complete native frame. This makes
the PG request's explicit CineMA-view preprocessing executable for U frames.

## Stage-boundary code update and recovery

The user notified us of another team modifying main. S1 uses a checked code
snapshot. At the S1/S2 boundary, `pg_refresh_main.py` fetches main, accepts only
a fast-forward from a clean checkout, runs repository and PG checks, and
records old/new SHA plus validation output. It never updates a running stage.
The queue writes stage identity and logs before launching each child and
retains completed stages on recovery. Code errors require a reviewed fix and
an explicit recorded `--resume-failed` reason, not a blind retry loop.

`pg_health.py` is the bounded health precheck. Network errors wake the repair
agent; they cannot silently count as a healthy run. A stopped process is not
completion. Completion requires finalization and export/TEST receipts.

ACDC and M&Ms downloads both reached DOWNLOADED_VERIFIED. Real ACDC loader
confirmed 80 TRAIN, 18 VAL, 50 TEST, 16 labeled frames and 2,132 image-only U
frames. The final materialization is pg-k8-20261116-v2, with a recorded
prelaunch metadata-only repair moving raw CINE provenance into manifest
lineage (FUSE record fields and array bytes unchanged).

Largest TRAIN frame patient057, native 8×512×428, passed the S1 CUDA resource
probe without OOM: 0.476 s/update at four CPU threads, peak allocated 7,089,144,832
bytes. Two/four/eight threads were almost equal; no claim of significant speedup.
The separate canonical all-stage two-update GPU profile and full campaign
launch are recorded in their subsequent receipts. The profile is warmup-only
for S2/arms, not a measurement of active consistency peak memory.

Main update e71b6c1 was integrated before launch; it includes the other team's
single Claude review and Linux CPU acceptance receipts. Local PG suite: 151
passed, then 12 focused relocation/audit checks passed (overlap between runs).
No patient efficacy result is claimed by this setup report.
