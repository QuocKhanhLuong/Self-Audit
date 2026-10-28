# Fully automatic head pilot: locked implementation contract

Base: `bdb1c399d29c63ae49865af8db78954f79f9ad9d`, fetched 2026-10-27.
Source review: `FULLY_AUTOMATIC_REVIEW_VI.md` and its proposal/source snapshots
under `/Users/alvinluong/.local/share/openresearch/files/self-audit/novelty-audit-20261027/`.
Status: implementation contract; patient and PRO4000 results NOT RUN.

## State and information

- Deployment is raw NativeStudent with the fixed-window feature path. No teacher
  certificate, query-patient guarantee, novelty, or efficacy follows from checks.
- Guard reference is theta_sup, after the ordinary support-supervised update.
  Only the extra U displacement is guarded. Rejected functional trials cannot
  mutate student, optimizer, teacher, buffers, or RNG. Commit directly; EMA once
  after final state. Each trajectory owns its own teacher.
- All feature-producing parameters and mutable feature state are frozen during
  head adaptation, including window and decoder. Full-EMA is a declared control.
- Fixed support consists of complete labeled TRAIN-L frame volumes selected by
  image/manifest metadata before adaptation. All support and checkpoint-lineage
  target labels count in the declared TRAIN budget. No U/query/VAL/TEST masks
  reach learner, support selection, or guards. Unknown label lineage fails closed.
- Same initialization, support-only warm-start, schedules, inputs, class candidate
  IDs, common accepted-pixel normalization, label and compute budgets across arms.
  K=1 support reuse must be explicit; it is optimization data.

## Metrics and controller

- Multiclass argmax; sum TP/FP/FN over all valid voxels/slices of each frame
  volume before ratios. Mean frames within patient, then mean patients per
  foreground class, matching the evaluator's absent-class policy. Primary guard
  is patient-macro per foreground class, not strict per-patient nonregression.
  Absent-frame FP is additionally constrained with a fixed observed-voxel
  denominator. Ignore 255 and geometry padding contribute no counts.
- Brier proposer and finite Brier predicate remain. Hard-Dice is an additional
  finite predicate, using identical fixed support and theta_sup reference.
- Common scalar alpha=s*1 is a distinct arm: a_c=1^T A_c 1,
  b_c'=1^T b_c, ridge_scalar=m*ridge. Keep all cross terms.
- Trials use cached base logits plus candidate logit responses. Verify parity
  against the actual head and end-to-end fixed-window forward; reject stale
  caches after feature/input/view/geometry/state changes. No event sweep,
  soft-Dice proposer, temporal/3D candidates, or teacher-memory additions.

## Interfaces and ownership

- Controller worker owns `self_audit/automatic_control.py` and
  `tests/test_automatic_control.py`. Public entrypoint
  `controlled_head_step(head, features, labels, directions, source_ids, *,
  patient_ids, frame_ids, valid_mask=None, allocation='joint', dice_guard=False,
  ridge=1e-6, budget=0., backtracks=8, tolerance=1e-8, dice_tolerance=0.,
  fp_tolerance=0.)`. Features are concatenated BCHW slices, metadata has B
  entries, labels BHW, common padded H/W allowed with boolean validity. Return
  JSON-safe control diagnostics. Existing control.py stays compatible.
- Engine worker owns `self_audit/automatic.py`,
  model.py changes if needed, `train.py` router,
  and `tests/test_automatic_engine.py`. New opt-in config schema
  `self_audit.automatic.v1`; expose `run(config, output, *, synthetic=False,
  resume=None)` and strict checkpoint loader. The CLI remains train.py.
  Implement bounded runs, frozen support cache with fingerprints, strict
  initialization/lineage and exact resume. Coordinate schema with runner worker.
- Dedicated data/cache worker owns `self_audit/automatic_data.py` and
  `tests/test_automatic_data.py`, including complete support enumeration,
  native extraction, cache invalidation, memory bounds and all-shape parity.
- Runner worker owns `scripts/run_automatic_pilot.py`,
  `configs/automatic_pilot/`, `docs/AUTOMATIC_PILOT_RUNBOOK.md`, and
  `tests/test_automatic_runner.py`. Call train.py, never create a second trainer.
  Arms: scalar_brier, joint_brier, scalar_dice, joint_dice, frozen,
  support_only, plain_head_u, extra_supervised, smaller_lr, full_ema.
- Evaluation worker owns `self_audit/automatic_evaluation.py`,
  `scripts/evaluate_automatic.py`, and `tests/test_automatic_evaluation.py`.
  Export full native raw-student masks before scoring; separate image-only
  inference from protected query-reference reading. Reuse existing class metric
  definitions and record coverage, hashes, latency, throughput and VRAM.
- Coordinator owns this contract, integration review, final receipts and Git
  publication. Workers are sharing the worktree: never revert others' edits,
  never stage/commit/push. Report integration needs to coordinator.

## Acceptance and resource accounting

Test common-scalar restriction and cross terms, cached/actual parity and
invalidation, Brier-Dice and EMA counterexamples, volume/frame/patient reduction,
absent and ignored voxels, atomic rejection, EMA timing, lineage and resume.
Run `python scripts/check_repo.py` and `python -m pytest -q` at integration.
Only bounded synthetic smoke/profile locally; no long patient training,
downloads/licenses, active-job changes, or dataset deletion.

Runner prepares both equal-update and equal-time comparisons on PRO4000 24GB.
Report cold total cost (warm-start, extraction, adaptation, full output) and
marginal adaptation cost separately. Measure latency p50/p95, peak allocated and
reserved VRAM, throughput, acceptance, nonzero displacement and margin changes.
Missing components remain NOT_MEASURED, never invented zeros.

Research GO requires independent matched-compute evidence: joint beats scalar;
Dice guard beats smaller LR; U beats extra-supervised. Unrun quality endpoints
remain NOT_RUN and cannot yield GO. Final publication is separate branch only.
