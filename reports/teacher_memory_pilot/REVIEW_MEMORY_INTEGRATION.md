# Independent memory and integration review

Reviewer: `astra_novelty_review`, with independent read-only A1/cache/timeline
inspection delegated to `dual_teacher_reference_check`. Implementation owners are
the Luna workers. The reviewer changed no source or tests.

Base commit: `9341cc81fc046d61135b928f5ac51ebed19c19ed`.
Review date: 2026-10-24.

**Status: ACCEPT for the frozen pilot software path at the hashes below.**
No blocking formula or integration defects remain in this review scope. Repository
release checks and the separate teacher-provenance gate remain coordinator-owned.
This is a software/formula review. No real teacher checkpoint, patient training,
GPU efficacy, anatomical correctness or scientific novelty is certified here.

## Verified core behavior

| Check | Evidence | Outcome |
| --- | --- | --- |
| A3 formula | Independent NumPy scalar loops over endpoint scores, reliability, softmax, tanh and class deltas; no call to the production pooling helpers for the oracle | Maximum float64 absolute error `1.1102230246251565e-16` |
| A2 formula | Independent scalar double sum under endpoint marginal attentions | Maximum float64 absolute error `1.1102230246251565e-16` |
| A2 removes correspondence | Compare A2 before/after A4 endpoint-and-confidence permutation, with identical W and alpha | Maximum difference `8.673617379884035e-19` |
| Capacity and input information | A2/A3 share the exact endpoint tensors, one projection and six gains; only the pooling calculation changes | PASS |
| Gradients and fixed buffers | Backpropagate the residual and run one Adam update in a manufactured probe | W and alpha receive nonzero gradients; detached base logits receive none; all bank buffers remain bitwise unchanged |
| Class/entry contract | Canonical BG/RV/MYO/LV and all six unordered pairs; finite normalized vectors, confidence threshold and zero padding validated | PASS for core bank |
| Missing groups | Invalid entries mask attention; entirely empty groups return zero without fabricated features | PASS |
| Algebraic bounds | Scalar oracle plus existing tests check zero-sum class residual and default 0.5 cap | PASS for default pilot cap |
| Native alignment | Retain full padded stride-4 h, pad probabilities/validity to 4h, pool exact 4x4 cells, upsample to 4h and crop native output | PASS; non-divisible 9x13 geometry covered |
| Disabled path against actual base commit | Import base-commit model source via `git show`; compare with new model at identical seed on 31x35 manufactured input | Exact state keys/values, RNG, outputs and all parameter gradients |
| Safe bank storage | NPZ with numeric arrays and JSON bytes, `allow_pickle=False`, SHA-256 checked before parse | PASS |

The independent probe used seed 67211 for a seven-dimensional manufactured bank,
seed 123 for W, nonuniform endpoint confidences, one missing class-pair, and
nonuniform alpha. The disabled-path probe used seed 971. A reviewer assertion
converted a requires-grad tensor to a scalar and emitted a PyTorch warning; the
probe completed with the errors reported above. This warning was in the review
harness, not a training failure.

Initial focused execution:

```text
rtk proxy python -m pytest -q tests/test_teacher_memory_core.py tests/test_teacher_memory_integration.py
17 passed in 1.44s
```

## Integration findings and resolved contract interpretation

- The native model reads `refine[0]` before final native interpolation, adds memory
  only after base logits, and leaves the disabled forward path unchanged.
- A1 combines internal CE and external soft CE in one backward/clip/optimizer
  transaction. The teacher EMA update is outside that transaction and occurs once
  after the final student state.
- The coordinator explicitly chose **restricted A1**: external CE attaches only
  to an existing EMA-eligible U transaction. If EMA support is empty, external-only
  support cannot create another optimizer step. Candidate external support and
  applied external loss support must be logged separately. The initial review
  flagged this as a possible bug; after the explicit protocol clarification it is
  an intentional design restriction, not an implementation defect.
- Checkpoint reconstruction obtains bank arrays from the student state, not an
  external bank file or online teacher. Existing manufactured tests delete the
  original bank and compare exported/restored student predictions exactly.
- A0–A6 configs share full EMA, fixed windows, schedule, crop, optimizer and batch
  choices. Missing patient assets remain a preflight failure rather than a fallback.

## Final-review items

- [x] A1 writer/loader explicitly declares and validates canonical class channel
  order; reject missing or permuted maps. A four-channel normalized tensor alone
  cannot establish RV/LV identity.
- [x] Opposite-support A1 regression freezes the coordinator's intended no-extra-U
  behavior: external candidate support positive, EMA support empty, applied
  external support zero, no pseudo transaction and one final EMA.
- [x] Confirm final geometry and cache integrity changes with the focused suite.
- [x] Record final reviewed source/test/config hashes before acceptance.

Final independent focused execution:

```text
rtk proxy python -m pytest -q tests/test_teacher_memory_core.py tests/test_teacher_memory_integration.py tests/test_memory_bank_builder.py
30 passed in 1.72s
```

The A1 cache now requires canonical class order and a separately hashed boolean
native support tensor. It validates simplex probabilities on supported pixels and
exact all-zero vectors elsewhere; the same crop and padding are applied to both
probabilities and support. Regression coverage includes a support hole under
padded cropping, swapped RV/LV rejection, and exclusion of the unsupported pixel
from external acceptance. These are image-geometry checks, not anatomical masks.

An intermediate moving-source run returned `16 passed, 2 failed` because the new
cache API returned `(probabilities, support)` while engine/test callers still
expected a tensor. The integration owner updated those callers before the final
30-pass run; the intermediate failure is retained here rather than hidden.

Acceptance applies to the shipped `tau=0.2`, default `gamma_max=0.5` pilot recipes.
The lower-level `TeacherMemory` constructor permits another positive temperature
or gain cap when called directly; those configurations are outside this frozen
formula review. The API's gain-cap option is not exposed by the pilot config.

The native-source/teacher lineage in a cache must not be confused with verified
patient-level provenance. Prepared-image hashes and probability hashes bind the
runtime inputs; they do not prove external teacher training disjointness. That
separate gate belongs to the teacher-provenance audit.

A4 may shuffle across patients and therefore also change patient association;
record this limitation when interpreting a patient experiment. Identity groups
with fewer than two entries cannot test pairing. No such limitation invalidates
the manufactured implementation checks above.

## Original reviewed snapshot SHA-256

| File | SHA-256 |
| --- | --- |
| `self_audit/teacher_memory.py` | `d491ec83f2b3e466eeca8e937bbaf3de09f4c9fcedbc2e30fd2c348d1257a5a8` |
| `self_audit/model.py` | `f0180ba618e53f613267698a043b4295c0b721bb262979178d4d520fbafa454f` |
| `self_audit/engine.py` | `e40757b248811b4525205320d1a756ca9da1c1fd1260f86d977c5aa0124c3f32` |
| `self_audit/external_pseudo.py` | `c817a18a4e76ce058fdbbd9a5970f12d7ec77f8c6077df8290ce8ec420dd6d2e` |
| `self_audit/memory_bank.py` | `a7acff5931782fad839553e7c9c14b6b348922d7b73e0e5ec0fb999b3ae0262b` |
| `scripts/export_teacher_memory_student.py` | `feb5cd8cd5abe2416033e32710dba0955ec885e8d185f09d70da1c98c86bb469` |
| `tests/test_teacher_memory_core.py` | `797b7cc06da228522705a4141654d5477d8689272afe4ba8a6dcbbb31f5a2992` |
| `tests/test_teacher_memory_integration.py` | `6be8de00425338c3ea09f978acca09d131f9ce3b5f23fe198b4cec7b4ac1e20f` |
| `tests/test_memory_bank_builder.py` | `d9f34542a9951a64ea14e4eae8f77516950f2891891aa341d2b719317c707d59` |
| `configs/teacher_memory_pilot/A0.json` | `aa9fc760b6c5d2588aef9e4d76528238c1f10fad95e1a282979da3b947754ad6` |
| `configs/teacher_memory_pilot/A1.json` | `973664d6469d9873cdb8015c6d7c24643ea74abfd94430adb7741c46d4765e24` |
| `configs/teacher_memory_pilot/A2.json` | `6f3c44530c1db168855446c136e82a1e3d4832745404cc9bfdc9e9f6c68a6931` |
| `configs/teacher_memory_pilot/A3.json` | `74dce694f55721b1c4774772cf8672e5e62c14224811bb862bf2e7287eaf4690` |
| `configs/teacher_memory_pilot/A4.json` | `61d64c8eb0662817920ecd25486c5ae2864c46b5cf3a81fdf9e2e5534e7a76fd` |
| `configs/teacher_memory_pilot/A5.json` | `6c5b267cf1f4113fe0658555f50c1259b7aeea8fa5244e63eb1a51d6f3a42dbe` |
| `configs/teacher_memory_pilot/A6.json` | `1b8049c6e719f6aef0abfc5780c0c9fc922d8209be930e4c0bb86d1f965e8ebc` |

## Export CLI defect and verified closure

The coordinator subsequently reproduced `ModuleNotFoundError: self_audit` when
calling the original export script directly. The earlier export test imported the
module in the pytest process and therefore did not cover this CLI entry point.
That was a review coverage gap; the prior export claim covered the imported
function, not direct-script usability.

The integration owner added the repository root to `sys.path` before the
`self_audit` imports. A new regression runs the absolute script path from a
separate temporary directory with `PYTHONPATH` removed.

Independent verification after the fix:

```text
# cwd=/tmp; no PYTHONPATH
rtk proxy env -u PYTHONPATH python /Users/alvinluong/Self-Audit-teacher-memory-pilot/scripts/export_teacher_memory_student.py --help
exit 0; checkpoint/hash/output arguments displayed

rtk proxy python -m pytest -q tests/test_teacher_memory_integration.py
8 passed in 2.48s
```

**Closure: ACCEPT for the export CLI fix.** Updated reviewed digests:

| File | SHA-256 |
| --- | --- |
| `scripts/export_teacher_memory_student.py` | `fa44bf04f040f6d92d39923c68c0bb21a19013005dd466a04d2ae2d6fd03bea2` |
| `tests/test_teacher_memory_integration.py` | `ab6896d7d4ab6960b57557d85b13e77f27ef22e42c6be2f1d8a8b21233238803` |

The engine, model and core memory hashes remain identical to the original review.
The bank builder and its tests changed under the separate ongoing bank review;
this CLI closure does not certify those newer builder revisions. The original
snapshot table above is retained as historical evidence rather than silently
relabeling old checks as checks of changed files. GPU operator-cost measurements
reported by the coordinator are outside this CPU CLI closure and do not establish
segmentation quality.
