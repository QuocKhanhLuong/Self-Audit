# PG K8 S2 spatial roundoff recovery

## Measured failure

The existing S2 attempt exited at **2026-11-16 14:02:07 UTC** with
`ValueError: image must be finite and normalized to [0,1]` in
`StatefulViewGenerator.joint_views -> strong -> _validate_image`.
This is a new augmentation error, separate from the preserved OOM attempts.
The last durable checkpoint is update **1950**, SHA-256
`f1d49002107479263e1456e7e0157a6c3dcae302b39aaefa8fbacf5388949dd6`;
the last committed metrics are epoch 24/update 1920.

An augmentation-only CUDA replay restored the original sampler and labeled/U
private view RNG states from that checkpoint. It performed no model forward,
optimizer step, evaluation, training invocation or checkpoint write. The replay
reproduced the error at update **1989**, U partner **patient011/frame 5**;
the labeled key was patient077/frame 8 and U anchor patient094/frame 13.

The native `[1,1,9,256,216]` input and permitted partner crop/pad canvas
`[1,1,10,174,208]` were finite FP32 with exact range `[0,1]`. Bilinear rotation
using matrix
`[[0.7102487087249756,0.703950822353363],[-0.703950822353363,0.7102487087249756]]`
produced exactly one voxel at **1.0000001192092896**. A subsequent quarter-turn
preserved that value. The excess is one FP32 representable step above 1.
There were no negative or nonfinite values. The diagnostic peaked at
139,099,136 allocated CUDA bytes; this is augmentation-only memory, not training
fit evidence. The original checkpoint SHA was unchanged after replay.

## Narrow correction

Only `candidate_pg/views.py` changes production behavior. After the existing
spatial primitive returns, PG accepts finite nonnegative output up to
`1 + float32.eps` and projects that single representable upper overshoot to 1.
Negative/nonfinite values and excursions of two ULPs or more fail with measured
min/max context. Native inputs and direct strong-view inputs remain strictly
validated. In-range weak tensors are returned unchanged. Targets, validity,
native geometry, random draws and state formats are unchanged. Historical FUSE
augmentation and all other historical tracks remain untouched.

The 13 focused tests exercise a real CPU bilinear overshoot, labeled/unlabeled
and batch paths, bounded rejection, strict native/direct-strong checks,
in-range output and private/global RNG parity, target/valid/geometry preservation,
and disabled-augmentation identity. All 13 pass; repository checks and all 187 PG tests pass (70.18 seconds).
Full PG and repository test results and recovery execution are recorded below
when measured. This report does not claim campaign completion or all-stage fit.

## Recovery status

**CPU_VERIFIED_GPU_REPLAY_RESUME_PENDING**. The PG code coordinator confirmed
sole recovery ownership and the one-ULP-only scope. Original S1, checkpoint
1950, configuration, manifests, completed epochs and all previous failures must
be preserved before the stopped-boundary deployment. Recovery requires reviewed
source, passing checks, one recorded full-budget resume, and a newly verified
durable checkpoint beyond the reproduced failing update 1989.

All four arm exports, finalization and committed final TEST receipts are still
outstanding. Keep automation `c851dc66-2854-459a-9ef3-9ca6a4ae6e1e` enabled.

The exact source and test bytes passed code-team review. The two-update PG
CPU smoke passed with its explicit software-only overrides. The Q3D CPU smoke
completed with its explicit synthetic override; its tokenizer gate remains
NOT_READY/passed=false. Full repository tests are still running at publication.
Original S2 and failure evidence are preserved under
`code_updates/spatial-roundoff-20261116T141644Z/`, including a hash-verified copy
of checkpoint 1950. No training resume has yet occurred.

Coordinator relay of user steering at 14:16:12 UTC: finish this reviewed repair
and resume promptly; subsequent code fixes must use Luna only.
