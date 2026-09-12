# Joint training and predicted feedback from epoch 1 — integration review

Date: 2026-09-12. Reviewed base HEAD and fetched `origin/main`:
`4d1900c132fb55370ab04856cb8e83069e663cad`.

**Decision: GREEN for executing the new curriculum as a controlled experiment.**
This is a software/configuration verdict, not evidence of useful early Auditor
feedback, better segmentation, faster RTX 4070 epochs, or scientific novelty.
The user explicitly selected joint training and predicted feedback from epoch 1,
accepting that the initial Auditor is untrained. No curriculum warmup was retained.

## Implemented behavior

The new ACDC and M&Ms profiles use one interval `[0,130)` with `trainable=all`,
`objective=retained_final_annotation`, and `rollout=threshold_gate`. Human epochs
1–130 all execute the existing joint loss. The native Candidate C runner defaults
to these profiles; `CURRICULUM=staged` explicitly selects the previous curriculum.
The canonical `self_audit_full*.yaml` files remain staged baseline configurations.
The generic `run_full_pipeline.sh` still requires `--config` to select a new profile.

Base learning rates are encoder `3e-5`, annotation `3e-4`, Auditor `3e-4`.
These are provisional early-training rates, not the old epoch-121 fine-tuning
rates. The existing five-epoch learning-rate ramp remains: it does not freeze
either network or defer feedback. Its first-step multiplier is `1e-8`, so unit
tests using a larger learning rate do not establish visible weight movement at
that first scheduled step. Best selection opens at global epoch 0, subject to
the unchanged completed-validation and gated-metric requirements.

The objective remains

\[
L=L_{\mathrm{annotation}}(A_{\mathrm{retained}},Y)
 + L_{\mathrm{auditor}}(\mathrm{stopgrad}(A_{\mathrm{previous}},
 A_{\mathrm{candidate}},F),Y).
\]

Auditor loss updates the Auditor; its inputs are detached. Annotation loss
updates the retained prediction path. The discrete acceptance decision is not
differentiated. If the first proposal is rejected, the row HALTs and the
refinement expert receives no annotation-loss gradient for that row; A0/encoder
and Auditor can still receive their respective gradients. An all-reject collapse
therefore remains possible. No forced acceptance, extra candidate supervision,
GT query input, or softened HALT was added.

Predicted accepted evidence is consumed on later turns of the same image.
Candidate C still requires a previously accepted ordinary replay record: enabling
the mode at epoch 1 does not create history at turn 0. Records retain their
existing optimizer-update invalidation. Model-core, loss, solver, acceptance,
encoder and data implementations are unchanged.

`predicted_history_exposure=false` in these profiles only leaves the auxiliary
rollout unrequested. The joint branch ignores that flag under both values and
performs one `infer`/encoder pass per batch. New JSON/W&B telemetry explicitly
reports the interval's own `feedback_gates` and `joint_trainable` status.

## Ownership and review

Orca run: `run_26ef57c4af40`. One Astra coordinator reviewed all diffs. Two Claude
Opus workers were launched through Orca; launch receipts confirmed `opus/high`.

| Task | Ownership | Final decision |
|---|---|---|
| Gradient/feedback contract and integration tests | `unified_trainer.py`, `test_joint_from_start.py`, core report | PASS after revision: deterministic score fixtures, per-loss gradient assertions, zero weight decay in movement test, real Candidate C record timing |
| ACDC/M&Ms profiles and native runner | Two new YAML profiles, native runner, generic-runner help, config tests, usage report | PASS after revision: fail-closed generated-config validation and correct auxiliary-flag semantics |

The provisional warmup objective/helper was removed after the user's explicit
choice. `schedule.py` and all model-core files have no final diff. Production
trainer changes comprise 21 additive telemetry lines. No worker changes were
integrated solely on the basis of its reported test totals.

## Independently reproduced gates

Environment: macOS CPU, PyTorch `2.14.0`; CUDA unavailable. Target PyTorch 2.4.1
and RTX 4070 execution were not reproduced here.

Compile/import of the modified trainer and new tests, strict loading of both
profiles, and `bash -n` on both changed launchers passed.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_joint_from_start.py tests/test_joint_from_start_configs.py \
  tests/test_unified_console_progress.py \
  tests/test_runtime_checkpoint_commit_integration.py \
  tests/test_candidate_c_training.py
```

Result: **107 passed in 55.94 s**. One existing test emitted a warning about
converting a differentiable tensor to a Python scalar; there were no failures.

Two additional real-CLI smokes used separate synthetic four-class cohorts,
32×32×2 volumes, batch 2, two refinement turns, and `window_mode=candidate_c`.
Each copied its new dataset-specific profile and reduced it to one full epoch
with two optimizer steps and full tiny-cohort validation. Pretrained loading,
AMP, post-training calibration and extra decomposition passes were disabled
only in these temporary test configurations. Production profiles keep their
pretrained encoder and existing settings.

| Gate | ACDC | M&Ms native |
|---|---:|---:|
| CLI exited successfully; completed report | PASS | PASS |
| Optimizer steps / joint rollout batches | 2 / 2 | 2 / 2 |
| Auxiliary rollout batches | 0 | 0 |
| Reported accepted-history evidence attempts | 4 | 4 |
| Candidate C attempts / feasible / fallback | 4 / 0 / 4 | 4 / 0 / 4 |
| Reported replay failures | 0 | 0 |
| `feedback_gates=active`, `joint_trainable=1` | PASS | PASS |
| `best.pt`, `last.pt`, committed selection | PASS | PASS |

These smokes exercised record-consuming attempts and fallbacks. They did not
produce a feasible restitution and do not establish successful correction or
numerical factual-replay identity by themselves. The deterministic unit test
separately verifies no record at turn 0 and consumption of the accepted turn-0
record at turn 1, without assuming solver improvement.

Temporary evidence directory:
`/var/folders/zk/7qfmzgnd1c9fqnzv4lfsm8t00000gn/T/self-audit-joint-from-start-8e20c8nq/`.
It contains `smoke.py`, `results.json`, per-dataset generated configs, stdout,
stderr, reports and checkpoints. It is local ephemeral evidence, not a
distributed real-data result.

## Use and limits

```bash
# ACDC native, followed by a separate M&Ms native Candidate C run:
bash scripts/run_acdc_mnms_candidate_c.sh

# Explicit staged baseline:
CURRICULUM=staged bash scripts/run_acdc_mnms_candidate_c.sh
```

The runner retains the user's existing batch-8 override and GPU visibility
defaults. These were not validated on a GPU in this change. Running training
processes retain their loaded curriculum. The new profile requires a separate
run; strict resume of a staged checkpoint under a different schedule/config is
not supported, and no lineage guard was relaxed.

ACDC and M&Ms configuration/data identities remain separate. External ACDC→M&Ms
evaluation code was unchanged and not rerun. No external M&Ms data was used to
tune the schedule, learning rates, solver or threshold. Full-suite execution,
real-data validation, long training, GPU memory/performance measurements, and
scientific effectiveness experiments were intentionally skipped.

For a Candidate C comparison, use this same joint-from-start curriculum for
the current-window baseline and Candidate C. Comparing a staged baseline with
joint-from-start Candidate C alone would confound mechanism and curriculum.
