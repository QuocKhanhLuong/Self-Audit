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

## Verified recovery

**RECOVERED_DURABLE_PROGRESS_VERIFIED_CAMPAIGN_RUNNING**, observed at
2026-11-16 14:30:11 UTC. Reviewed source
`dfeee8806daf471e6c8b045a42cd5ad791da487d` was published with a four-path
allowlist, remote-SHA verified and deployed at the stopped boundary under
`QUEUE.lock`. The running checkout remains pinned to this tested source.

- Local focused tests: **13 passed**. Full PG suite: **187 passed in 70.18s**.
- Full local suite: **1,178 passed, 4 skipped, 3 subtests passed in 457.74s**;
  two expected duplicate-ZIP warnings came from corruption fixtures.
- Repository checks pass locally and on the server. Server PG suite:
  **187 passed in 206.04s**.
- The two-update PG CPU smoke passes with its explicit software-only overrides.
  Q3D CPU smoke completed with its explicit synthetic override; its tokenizer
  gate remains NOT_READY/passed=false. These are software checks only.
- Original checkpoint 1950 strictly loaded the real teacher's 541 state keys
  and AdamW step 1950 on CPU without initializing CUDA.
- Fixed augmentation-only replay passed through updates 1951–2000. It corrected
  exactly the reproduced update-1989 voxel, with the original private RNG hash,
  and left checkpoint 1950 unchanged. No model forward or optimizer ran in replay.

The existing service resumed **once at 14:25:07 UTC** from update 1950 with
the unchanged full 8,000-update budget, two strong views, and zero automatic
retries. Durable checkpoint **2000** then loaded on CPU and matched its sidecar:
`96818d071dfb51c537cc2336e6c1410f1c01e37ced927eba5419cc7b9c089386`.
Its AdamW step is 2000; epoch 25 has 75 committed rows. The original resolved
config SHA is unchanged. Finite supervised/U/total diagnostic losses are
`0.7456923723220825 / 0.0272216796875 / 0.7729140520095825`.

Measured peaks through diagnostic 2000 are **7,909,834,752 allocated bytes
(7.37 GiB)** and **11,947,474,944 reserved bytes (11.13 GiB)**. One GPU trainer
PID 534662, existing supervisor PID 534633, and one `pg-k8` window were verified.
These observations establish recovery past the reproduced failure, not
whole-stage or all-stage GPU fit or scientific efficacy.

Original checkpoint 1950, S2 metrics and failed logs/queue/service/boundary
receipts are hash-preserved under
`code_updates/spatial-roundoff-20261116T141644Z/`. Completed S1, original data,
manifest/split/config identities and all earlier attempts remain preserved.
The directory also contains the server validation, strict CPU load, bounded
replay, explicit resume request, start result and verified-progress receipts.
The matching JSON report binds these observations to hashes and source.

All four arm exports, finalization and committed final TEST receipts remain
outstanding. Automation `c851dc66-2854-459a-9ef3-9ca6a4ae6e1e` was verified
enabled and remains enabled. No additional deployment or restart is needed
while the stage is healthy. M&Ms remains preparation-only; historical tracks
and runs were not changed.

Coordinator relay of user steering at 14:16:12 UTC: finish this reviewed repair
and resume promptly; **subsequent code fixes must use Luna only**. This
restriction is also preserved in `MONITOR_INSTRUCTIONS.md`.
