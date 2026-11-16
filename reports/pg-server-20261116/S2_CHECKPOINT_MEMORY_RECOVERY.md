# PG K8 S2 activation recomputation recovery

## Reviewed mitigation

After the separately archived 12:19:39 UTC OOM, the run remains stopped at
durable update 800. The stream-accumulation fix alone did not establish GPU
fit. This second mitigation checkpoints each native MedNeXt block using
non-reentrant activation recomputation with RNG preservation. Sequential
stacks are checkpointed block by block. Parameters, model layout, state-dict
keys, native geometry, both strong views, FP, losses, optimizer schedule,
data/splits and full budgets are unchanged. FP generator draws remain outside
the recomputed blocks. Evaluation, no-grad and frozen no-grad-input paths
bypass recomputation.

The pinned real teacher was built and inspected on CPU only. Its core has
54 MedNeXt blocks, 62 GroupNorm modules and zero buffers. Inspection of the
pinned block forwards found no mutable running statistics. Official source
commit is `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`; `blocks.py` SHA-256 is
`2af27e80a743c84020df50c775e56f83a97a9f4be9d3e461c53b023f2a7da2fe`.

OOM handling now appends update, resume update, labeled/U frame keys, stream,
native/padded shape and CUDA memory counters to `S2/failures.jsonl`, then
rethrows the original exception. It performs no retry or optimizer step.

## CPU software evidence

- Repository checks: PASS. All PG tests: **174 passed in 69.73 seconds**.
- Five new tests plus the existing 17 teacher/stream tests pass, including
  the 800-to-802 resume regression.
- A representative eight-block module retains **3,539,456 bytes** of unique
  non-parameter saved storage without recomputation and **131,072 bytes**
  with it. This is a CPU fixture measurement, not GPU peak or fit evidence.
- A native anisotropic adapter fixture with multiple decoder blocks and FP
  compares region/SDF/boundary outputs, features, all parameter gradients,
  AdamW parameters/moments, global RNG and explicit FP RNG. FP32 and CPU
  BF16 comparisons are exact (`atol=0`, `rtol=0`). State keys are unchanged
  and the original state loads strictly. This is software parity evidence,
  not a claim of bitwise CUDA equivalence.
- Bypass tests cover eval, no-grad and frozen modules. Injected second-strong
  OOM verifies exact context, original exception identity, no optimizer step
  and no checkpoint publication.

The PG code coordinator and setup coordinator independently reviewed the
three implementation/test files at these SHA-256 values and reported no
blocking finding:

| File | SHA-256 |
| --- | --- |
| `teacher_mednext.py` | `02941fd4e33fafbd28d58e261adef2bc2af451c9304f5ec68b3c64975b6f592a` |
| `teacher_training.py` | `1f3295a1a27963e91f2f93d57799f338c37ee594b9fc8f7edacdd6e451ce8a38` |
| `test_candidate_pg_teacher_checkpoint_memory.py` | `e2359986d245be47e267a7177f15074c3fd4073545adcfeef366db1986d05a17` |

## Deployment gate and current outcome

Status at publication: **CPU_VERIFIED_GPU_RESUME_PENDING**. Both coordinators
confirmed sole recovery ownership and authorized one recorded resume only
after stopped-boundary server CPU checks pass. The existing service will use
the unchanged full configuration and update-800 checkpoint; no capped run or
second GPU writer is authorized. Success requires a new durable checkpoint
above 800, finite post-warmup losses and measured memory. Another OOM must
stop and retain its detailed context, without an identical retry.

Original checkpoint SHA-256:
`09564256071f9811bc3e4521968284ad0b281da7258afda6c36c0d69b28c0138`.
Completed S1 artifact SHA-256:
`f596a68a692799326aac736fdff085543a495ad13f0d8c82270b4724782b1b8f`.
All prior failures, boundary checks and receipts remain preserved as described
in `S2_OOM_RECOVERY.md`. The automation remains enabled until the four-arm,
finalization and committed TEST requirements are met. This repair does not
establish all-stage GPU fit, scientific efficacy or campaign completion.
