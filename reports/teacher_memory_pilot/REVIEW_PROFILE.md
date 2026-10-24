# Independent manufactured native-cost profiler review

Date: 2026-10-24. Reviewer: `astra_novelty_review`; separate read-only CLI-dispatch
check by `dual_teacher_reference_check`. No source or tests were edited.

**Verdict: ACCEPT for the explicitly manufactured operator-cost run.** No blocking
issue was found for all seven arms, shared batch 4/crop 256, one warmup transaction
and two measured transactions. This verdict does not establish real teacher
readiness, patient efficacy, novelty, or real-data A1/A5 evidence.

Reviewed script SHA-256:
`11b89f20b1369872f6c295848e9b4a89e8704e7524ce2247b48c7f3be6997bdd`.
Reviewed four-test file SHA-256:
`df82fbb929a422e4ce991d7b485c7ee8b6b58e9dd5c5a9279261d1375691978f`.
The local worktree base remains `9341cc81fc046d61135b928f5ac51ebed19c19ed`;
`11b89f20` identifies the profiler file digest, not that Git commit.

## Source findings

| Requirement | Finding |
| --- | --- |
| Standalone operator path | `run()` immediately dispatches the native-cost flag. This path calls existing supervised/pseudo/EMA helpers and never calls `engine.train` or checkpoint saving. A new output directory is required; its only intended persistent artifact is `PROFILE.json`. |
| Real model, manufactured inputs | Requires a real DINO model config with pinned source/weights; no device fallback. Images, labels, bank vectors and external targets are explicitly manufactured. A5 is explicitly not a student-derived bank; A1 measures only the soft-target branch operator cost. |
| CUDA timing | Device synchronization occurs immediately before the timer and after the complete supervised, pseudo and EMA transaction. The first row is excluded from measured throughput under the requested one-warmup setting. CUDA runtime correctness remains to be observed on the actual run. |
| Counter definitions | At B=4 with two measured transactions, measured input exposure is 8 L + 8 U = 16 logical examples. The teacher also processes U, but the same U examples are not counted twice. Counts across all three transactions are 3 supervised, 3 pseudo, 6 body optimizer steps, 6 head optimizer steps and 3 EMA updates. Reused manufactured samples are processed exposures, not unique patients. |
| Same initial state and work | Every arm receives the same images/targets and seed. Seed, optimizer settings, EMA decay and window config are checked against A0. All seven shipped configs use `threads=2`, fixed windows, full EMA and tau=0.2. The native cost path forces pseudo support to include every pixel, ensuring the same number of complete transactions. |
| Trainable budget | A0/A1 have the common native student. A2–A6 add exactly W plus six alpha parameters: `32*96+6=3078` trainable scalars for the default real NativeStudent. Banks are buffers. The profiler does not emit total parameter counts; 3078 is a source-derived count, not a measured receipt field. |
| Per-arm cleanup | Model, teacher, optimizers, body parameter list, bank, memory alias and teacher outputs are deleted; GC and CUDA cache cleanup precede the next arm's peak reset. The loop variable `tensor` retains only the last 6x8 boolean bank-valid buffer (48 logical bytes), not the model or optimizer. This tiny residual is nonblocking and should not be described as literally zero retained CUDA references. |
| OOM/failure | Per-arm exceptions produce `FAILED_OOM` or `FAILED`, with `successful_profile_claim=false`, and are re-raised. No batch/device fallback occurs. Setup failures before the per-arm try block can exit before a failure receipt is written; absence of a completed report is not success. |

Peak allocated/reserved memory is reset before arm construction, so the reported
peak includes construction, warmup and measured execution, as well as the common
resident manufactured input tensors. It is not a measured-iterations-only peak.
Memory-arm gradient norm diagnostics occur inside the timed region; measured
throughput includes their overhead. With only two measured transactions, these
numbers are a bounded feasibility/cost probe, not a stable throughput benchmark.

`PROFILE.json` uses direct `write_text`, not atomic replacement. A process crash
during a rewrite could leave an incomplete JSON file; only a valid completed
receipt supports a completed-run claim.

## Executed independent checks

```text
rtk proxy python -m pytest -q tests/test_teacher_memory_profile.py
4 passed in 0.47s
```

An additional in-process CPU orchestration probe exercised the actual
`_manufactured_native_cost` helper while substituting a tiny manufactured model
at `build_model`. The report writer was intercepted so no artifact could falsely
claim real DINO execution. Traps forbade calls to `engine.train` and `torch.save`.
The probe found:

- Seven builds, 21 supervised helper calls, 21 pseudo helper calls and 21 EMA calls.
- Exact equality of initial common backbone/decoder tensors across all seven arms.
- Exactly two CPU threads; tiny-seam memory parameter counts
  `[0,0,262,262,262,262,262]` (`32*8+6` for its eight-channel query).
- Three rows per arm, one excluded warmup, two measured rows and correct counters.
- Zero full-trainer or checkpoint-save calls; no files persisted in the mocked
  successful output directory.
- An injected `CUDA out of memory` exception produced `FAILED_OOM` with no success
  claim and was re-raised.

This last probe is **CPU manufactured orchestration evidence only**. It did not
load real DINO weights, use CUDA, measure target-GPU speed, or execute patient data.
