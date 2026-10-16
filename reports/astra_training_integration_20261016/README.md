# Astra student training integration — 2026-10-16

**PASS_CPU_SOFTWARE_ONLY.** The new explicit runner, frozen-bank adapter,
configuration, checkpoint loader and image-only prediction API are implemented.
Use [the current 4070 handoff](../../docs/prompts/ASTRA_TRAIN_4070_LATEST.md).
Official DINO loading, real patient training, GPU execution and VAL/TEST quality
evaluation remain **NOT RUN** in this delivery.

## Code and preserved reference

Worktree branch: `codex/astra-student-training-20261016`. Training integration
starts from `ae71ad789175476295048db66ec5a24d741ce910`, which brings the previously
verified Astra port `2c5796084e4412ccab70830afca0adbb6a3e1d23` into an isolated
branch based on main `92a60744efef85558d7c0226c84b7c49d47217ec`. Main was not merged
or edited by this task. Test receipts record the execution base plus file hashes
of the implementation under review; those hashes identify the tested snapshot.

The immutable reference is `f4b0fe7d9e49ee398354e20ff0981c818840f5d4`. All 19
reference archive files verify against manifest
`3f442aa35872bea5db00eb8880c002b1049a1e53da2a88accfbf8e6e5cec2576`.
The six layer modules, original operator tests, probe scripts and pytest config
remain byte-identical. Original `integration/*.py` is unchanged.

No changes were made to baseline, historical source/configs/splits/runs or
original evidence. All **236 pre-existing file fingerprints** in the user's
original worktree still match the starting snapshot. The only existing files
edited by the training extension are two experiment documentation entry pages,
which now point to the latest handoff. Historical receipts were not overwritten.

## Verification

| Check | Result and evidence |
| --- | --- |
| Extracted original reference tests | **47 passed**, zero failures/errors/skips; [raw log](receipts/reference/pytest.log), [XML](receipts/reference/TEST_RESULTS.xml) |
| Complete experimental suite | **125 passed**, zero failures/errors/skips; [raw log](receipts/tests.log), [XML](receipts/TEST_RESULTS.xml) |
| Differential reference tests | 17 cases, including all output keys, input gradients and parameter gradients at four native sizes; strict `rtol=0, atol=0` |
| New training suite | 37 cases: configs, provenance/geometry, targets, runner, checkpoint, prediction and end-to-end |
| Random full-size model smoke | **PASS**, 32,890,558 parameters; [receipt](receipts/FULL_MODEL_CHECK.json) |
| Coarse projection discrepancy in that float32 smoke | Maximum `4.76837158203125e-7`; direct coarse-loss detail-head gradient sum **0** |
| Native-loss detail gradient in isolated smoke | Norm sum `0.009821066558996672`; this smoke objective is not enabled in the training recipe |
| Integration/source preservation | [parity receipt](receipts/PARITY_REPORT.json) and [delivery receipt](DELIVERY.json) |
| Ruff F/I checks on new implementation/tests/CLI | PASS; reference files excluded from formatting |
| Handoff command syntax | All six Bash blocks parse; embedded Python reload snippet compiles |
| CLI outside checkout | Help succeeds; missing approval exits 1 before output creation; [receipt](receipts/CLI_CHECK.json) |

The 17 differential cases use independent imported reference modules, cloned
states and identical inputs with a small manufactured encoder. They cover
forward/backward, input transform/losses/detach behavior, degenerate projection,
accepted/rejected/exception rollback and frozen-energy/proposal behavior. The
four forward/backward sizes are 32×32, 47×53, 64×96 and 65×97. This is exact
software parity, not an official pretrained-weight runtime comparison.

The new end-to-end test uses a manufactured frozen bank and real adapter,
AdamW loop, serialization, strict reload and native prediction. Only the
external DINO constructor is replaced with a small encoder. It performs five
actual optimizer steps, asserts coarse weights change and semantic updates are
positive, tracks actual versus planned patients and partial epoch counts, then
checks bit-identical same-input logits after reload. It also asserts the exact
split-head detail output head stays zero and has no gradient under these losses.

Prediction batch-size comparisons require exact dense/gated labels, with
`rtol=1e-6, atol=1e-7` for continuous confidence/entropy. A prior assertion of
bitwise continuous equality across batch sizes failed at about 1.19e-7; only that
new test contract was corrected. The original differential reference tolerance
remains exactly zero.

Observed local runtime: macOS arm64, Python 3.11.16, torch 2.14.0,
torchvision 0.29.0, NumPy 2.4.6, pytest 9.1.1, CPU. These are recorded versions,
not a GPU dependency lock. The target host must record its own environment and
run the official-backbone/patient pilot described in the handoff.

## Review and limitations

[Independent review disposition](INDEPENDENT_REVIEW.md) records the concrete
findings and fixes. No remaining P1/P2 code defect was reproduced in the final
review. Three implementation agents produced partial code then encountered a
usage limit; the root agent completed integration, corrections, tests and docs.
Those interrupted workers are not represented as completed independent reviews.

The explicit [policy/deviation record](../../experiments/layer_guided_student_v1/training_v1/README.md)
documents historical full-Q supervision, aligned full-view feature loss,
ordinary AdamW, inactive guard/refiner and the untrained native-detail branch.
No claim is made that the new architecture fixes noisy semantic sources,
converges on patients, improves anatomy or beats a matched baseline.
