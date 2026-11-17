# PG K8 A4 padding recovery

Status: **RECOVERY_VERIFIED_CAMPAIGN_RUNNING**, observed 2026-11-17
00:18:31 UTC. This recovery is complete; the campaign is not complete.

Luna implemented the PG-only fix, reviewed by the PG code coordinator, in
`2f7aade2cf39ee69e5f47e25dea040fd85d39b59`. Main publication and the stopped
server deployment were verified at that SHA. The existing supervised service
resumed once at 00:08:57 UTC from original production update 800. The source
remains pinned while training; this report update does not refresh the server.

The original failure at 2026-11-16 23:27:08 UTC was
`ValueError: probabilities must have positive class mass at every cell`.
It occurred on the first post-warmup A4 step. An isolated reproduction from
checkpoint 800 sampled L patient041/frame10, U anchor patient099/frame11,
and U partner patient090/frame2. Its first mixed probability packet had shape
`[1,4,16,224,154]` in FP32. Each fusion/teacher/clean-student packet had 115,200
zero-mass cells, all invalid padding, and no zero-mass valid cell. The second
packet had 92,160 such invalid cells. The old validator rejected padding before
applying validity support.

The fix requires positive class mass on valid support and preserves zero-mass
invalid cells as zero. Finite/nonnegative checks remain global. It does not
change the architecture, data, splits, recipe, budgets, thresholds or losses on
valid voxels. Regression tests cover all three probability inputs, nonfinite
values inside and outside support, valid-zero rejection, and loss/gradient
parity with zero invalid gradients and detached targets.

Validation passed 28 focused reliability tests and 211 PG tests locally.
Repository checks and all 211 PG tests also passed on the stopped deployed
server, with CUDA hidden during those tests. A fresh two-update CPU smoke
captured the final source hashes before/after execution, exited zero, and
produced a new COMPLETED result with unique epoch rows. These are software
checks. Separately, an isolated GPU replay reached update 801 while retaining
the original production checkpoint; that temporary checkpoint was not used
for production resume.

Production checkpoint 850 was archived, hash-checked and loaded on CPU:
`83675dad32107ffc6de69eee652f739af6f08a37645bf3aef9b152a6c87e90eb`.
Its student tensors were finite, sampler advanced 800 to 850, and resolved
configuration hash was unchanged. All 111 existing non-fusion optimizer states
advanced 50 steps; 20 conditional fusion states advanced 47 (769 to 816),
consistent with labeled teacher dropout bypassing that branch. Six newly active
code-branch states reached 50. Their actual keep draws were not replayed on GPU.
At update 850, PG loss was 0.3441212773, CBKD loss 54.7873840332 and total loss
27.5939960480. The latest checkpoint sidecar at final verification was 960;
72 unique committed epoch rows and the sole GPU trainer PID 847983 were present.
There is one `pg-k8:0` window and approximately 43 GiB disk available.

S1/S2/S3 remain completed at 10,000/8,000/4,000 updates. Their artifact and
receipt hashes, configuration and manifest hashes match the preserved values.
All arms retain the original 8,000-update budget. The original checkpoint 800,
failure/queue/logs, 60 epoch rows and lock files are hash-preserved under
`code_updates/a4-probability-20261116T233344Z/`. The queue retained the A4
failure as attempt-history entry 6. Original epoch metric/content values remain
unchanged; the live historical `best_val` marker follows later VAL/student-only
selection, and the live experiment lock records the newly deployed source SHA.

Diagnostic errors were investigated and retained. Two initial instrumented
replays lacked checkout PYTHONPATH. A worker shell message accidentally executed
smoke through backtick substitution in an already-used output directory; its
duplicate-row failure is not used as PASS evidence. Three progress-verifier
assertions incorrectly assumed an unchanged source lock, identical per-parameter
step counts, or byte-identical mutable selection rows. The final verifier checks
the exact source-only lock change, conditional parameter counters, and unchanged
metric rows with only legitimate VAL selection-marker changes. None caused a
production retry or additional GPU writer.

At update 850, measured code perplexity was 1 and dead-code count was 127.
This receipt establishes resumed finite optimization, not useful codebook
support, patient efficacy or all-stage GPU fit. All four arm exports,
finalization and committed final TEST receipts remain outstanding. Automation
`c851dc66-2854-459a-9ef3-9ca6a4ae6e1e` remains enabled. M&Ms remains
preparation-only; historical tracks, runs and datasets are unchanged.
