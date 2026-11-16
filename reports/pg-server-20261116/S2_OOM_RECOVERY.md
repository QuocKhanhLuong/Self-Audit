# PG K8 S2 post-warmup OOM recovery

At 2026-11-16 11:58:09 UTC the current PG queue failed during the second
strong-view forward of update 801, before backward or an optimizer step.
CUDA could not allocate 208 MiB with 22.63 GiB already allocated by PyTorch.
S2's last valid checkpoint is update 800, epoch 10, with 30 committed rows;
its SHA-256 is
`09564256071f9811bc3e4521968284ad0b281da7258afda6c36c0d69b28c0138`.
The checkpoint loads on CPU and matches its sidecar. S1 remains complete at
10,000 updates with artifact SHA
`f596a68a692799326aac736fdff085543a495ad13f0d8c82270b4724782b1b8f`.

The failure held the labeled and earlier U autograd graphs while building
later streams. The narrow S2 fix backpropagates each weighted stream before
constructing the next, then clips and steps the optimizer once. Backward is
outside separate autocast contexts. Only detached losses are retained for
logging, and unused weak-target feature references are released.

Both strong views and FP, original stream weights, forward/RNG order,
threshold update, configuration, data, split, model and budgets are unchanged.
No one-view recipe adaptation or checkpoint identity waiver was applied.
The gradient sum is mathematically equivalent; bitwise equivalence to the
old floating-point summation order is not claimed.

Both coordinators confirmed this monitor's sole recovery ownership through
Orca, and the code coordinator reviewed the scheduling constraints. Other
monitors stood down. Original queue/config/lock/receipt/log files and the
complete S2 directory are preserved at:
`/root/mount_data/runs/pg-k8-20261116/code_updates/oom-S2-20261116T115809Z/`.
`S2_OOM_PRESERVATION.json` records the original artifact and data hashes.

Local software verification:

- Repository checks: PASS.
- All PG tests: **169 passed in 67.02 seconds**.
- Four active-U comparisons cover one/two strong views and CPU FP32/BF16
  fixture forwards. Compared with retaining all weighted stream graphs and
  one combined backward, losses and RNG/sampler/threshold state match
  exactly. Maximum FP32 gradient absolute difference is `5.96046448e-8`;
  maximum post-AdamW parameter difference is `1.16415322e-10`. CPU BF16
  fixture differences were zero. These are tiny-model software measurements.
- Gradient tolerances: FP32 `atol=1e-6, rtol=1e-5`; CPU BF16
  `atol=2e-3, rtol=1e-2`. Post-step parameter tolerance:
  `atol=1e-6, rtol=1e-5`; optimizer moments are also compared.
- A fixture autograd marker measures four concurrent training graphs in the
  combined two-view reference and one with the fix.
- An 800-to-802 resume reproduces uninterrupted model, optimizer, RNG,
  sampler, augmentation and threshold state exactly, retaining 30 unique
  epoch rows. Nonfinite stream losses fail before optimizer mutation.

Real recovery outcome: **BLOCKED_OOM_AFTER_TESTED_STREAM_FIX**.
Fix commit `a0a88879994efd83ee9d86619886f5fd92f4b311` was pushed to main,
remote-SHA verified and deployed at the stopped S2 boundary. Server repository
checks and all **169 PG tests passed in 206.69 seconds**. One explicitly
recorded resume used the existing service, with automatic retries disabled.

At **12:19:39.983561 UTC**, that attempt failed in an individual U-stream
forward at `teacher_training.py:702`, through MedNeXt `dec_block_1`/`conv2`.
The failed allocation was 216 MiB; the exception reported 21.77 GiB allocated
by PyTorch, 1.20 GiB reserved but unallocated, and 96.56 MiB device-free out
of 23.43 GiB. These exception values are not a measured whole-stage peak.
The exact stream name, patient and failing update were not logged; do not
infer them. A prior allocator warning was followed by continued process
activity, but this later traceback and queue failure confirm termination.

No new diagnostic/checkpoint beyond update 800 was committed. The original
checkpoint hash and 30 epoch rows still match, so post-800 optimizer progress
is **NOT_VERIFIED**, not zero and not successful recovery. S1 and all
configuration/data hashes remain unchanged. There is no GPU writer, and no
further invocation was launched. The one-view adaptation remains untested
and unapplied; it must not be presented as a proven solution to this failure.

The second failed attempt, queue, traceback log and server validation are
archived at
`/root/mount_data/runs/pg-k8-20261116/code_updates/oom-after-stream-fix-20261116T121939Z/`.
`S2_OOM_RECOVERY_RECEIPT.json` binds the outcome and unchanged checkpoint.
Both coordinators were informed. The named PG monitor remains enabled;
future recovery requires a concrete reviewed mitigation, not an identical
retry. The original GPU profile is still not an all-stage PASS. Campaign
completion still requires S3, all four arm exports, finalization and committed
final TEST receipts.
