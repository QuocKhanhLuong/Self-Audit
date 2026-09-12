# M&Ms follow-up integration review and research decision

Date: 2026-09-12. Reviewed/fetched base HEAD:
`bb639ec118af72dac0d5e24339a5e325e2d6d163`.
Scope: M&Ms native launcher/data preflight, frozen ACDC→M&Ms compatibility,
and a structured research opinion after the joint-from-epoch-1 change.

## Review boundary

One Astra coordinator; two Claude Opus workers through Orca run
`run_7c5431424f17`. Fresh launch receipts recorded `opus/high`; subsequent
tasks reused those same terminals. No second Astra or non-Orca worker was used.
The coordinator inspected actual source and test diffs and independently ran
the final gates below.

| Workstream | Owned files/responsibility | Review |
|---|---|---|
| Native M&Ms | Native launcher; `mnms.py`; native branch of training `_utils.py`; README; native safety tests/report | Initial blanket PASS rejected: existing config overwrite and silently omitted pairs matter. Revisions cover exclusive config creation, exact pairing and isolated tests. Final result recorded below. |
| Frozen external M&Ms | New nested-checkpoint lineage tests; external review report | Production trace accepted. Initial claims that configurable mapping/axis themselves prove leakage rejected; report corrected. Regression tests add real nested schema and alias integrity coverage. |
| Astra synthesis/integration | This report and research opinion; final source review, gates and publication | No production model/loss/solver implementation by the coordinator. |

No new literature search was performed. Existing untracked research and
unrelated artifacts were preserved.

## Concrete changes

1. The native runner generates both configs inside a fresh `mktemp` directory
   on every invocation. Exclusive file creation refuses overwrites; generated
   YAMLs are read-only and retained. Interleaving a second invocation during the
   first ACDC leg cannot change the first run's pending M&Ms batch configuration,
   including when timestamps match.
2. Native supervised directory-split preflight requests strict pairing.
   Missing partners, duplicate normalized filenames, ambiguous mask matches
   and shared mask assignments are rejected instead of silently altering the
   labelled cohort. External/default discovery remains tolerant for intentionally
   unlabelled volumes; declared input encodings remain configurable.
3. README distinguishes generic staged defaults, the native Candidate C runner's
   joint-from-start default, baseline `window_mode=current` YAML profiles,
   profile-specific best-selection boundaries, and the actual source run's
   checkpoint path for external evaluation.
4. New tests cover interleaved launcher execution with a fake trainer and real
   config generation, strict native preflight, nested unified checkpoint schema,
   inherited/conflicting Candidate C settings, refusal of native M&Ms models as
   external ACDC evidence, and committed/stale best aliases.
5. [The research opinion](joint_from_start_research_opinion.md) updates the
   earlier formulation for the new curriculum, separating trainability,
   accepted-history exposure, ordinary-generator learning, restitution,
   transfer, final acceptance, and mechanism necessity.

Changed production/documentation paths: `README.md`,
`scripts/run_acdc_mnms_candidate_c.sh`, `src/self_audit/data/mnms.py`,
`src/self_audit/training/_utils.py`.
Added tests: `tests/test_mnms_native_safety.py`,
`tests/test_external_mnms_joint_lineage.py`.
Added reports: this file, the research opinion,
[mnms_native_followup_review.md](mnms_native_followup_review.md), and
[mnms_external_followup_review.md](mnms_external_followup_review.md).

## Unchanged scientific/runtime contracts

The native runner still defaults to joint training and threshold-gated feedback
from human epoch 1 for both independent datasets. Generic entrypoints and the
old full profiles still default to the staged baseline. No running process
reloads its already-loaded curriculum after a pull.

No change to A0, pretrained ConvNeXt, Dynamic Window, Candidate C C1/C2/C3,
Auditor detachment, runtime GT firewall, reject→HALT, replay invalidation,
optimizer/loss, calibration logic, or checkpoint commit/resume policy.
Candidate C remains the selected mechanism; no Candidate B was introduced.
Batch-8/CUDA visibility defaults were retained without a new GPU capacity claim.

Pair validation reads annotations as part of supervised dataset validation,
never as runtime proposal input. External evaluation computes offline metrics
from masks; the runtime Candidate C solver still receives predicted evidence
and differentiates only intervention coordinates through frozen replay.

The checked-in M&Ms profiles and protocol retain BG/RV/MYO/LV,
raw mapping `{0:0,1:3,2:2,3:1}`, depth axis 2, and patient split checks.
The native runs use separate processes, data/config/checkpoint identities and
no cross-dataset resume. External evaluation accepts a frozen ACDC checkpoint
and rejects native M&Ms provenance as independent ACDC→M&Ms evidence.

External `tau=0.0` is a prespecified protocol value, not an automatic import of
the source run's calibrated threshold. The evaluator provides no target-data
training, checkpoint selection or tuning routine; this does not make arbitrary
manual rerunning/tuning on external results scientifically valid.

## Independently executed final gates

**Final decision: PASS for both workstreams; GREEN for publication of this
bounded software change.** All workers had settled before these integration
gates. Earlier worker test counts were not used as substitutes.

Environment: macOS CPU, PyTorch 2.14.0, no CUDA or timm.
Compilation of both changed Python modules and both new test files passed.
Critical imports, strict loading of the two joint and two staged profiles, and
`bash -n scripts/run_acdc_mnms_candidate_c.sh` passed.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_mnms_native_safety.py \
  tests/test_external_mnms_joint_lineage.py \
  tests/test_mnms_wave2.py \
  tests/test_candidate_c_mnms.py \
  tests/test_joint_from_start_configs.py \
  tests/test_external_mnms_evaluator.py \
  tests/test_mnms_external_flow.py
```

**98 passed in 11.28 seconds.** This includes native split/mapping checks,
isolated shell interleaving with real config generation, and external
checkpoint/mode/solver/alias checks. `git diff --check` passed.

The coordinator additionally invoked the real external evaluator against the
previous CLI-produced tiny joint ACDC `best.pt` and a new one-case,
two-slice synthetic M&Ms testing cohort. It reported `training_dataset=acdc`,
`window_mode=candidate_c`, and protocol `tau_accept=0.0`. The bound checkpoint
SHA256 was
`9dfe196854338c9a0b78c81e6f73728eded3008a684bda4f782e8ca0d6f092c6`,
exactly the `best_reference.sha256` in the source run's `last.pt`.

Root-produced temporary evidence:
`/var/folders/zk/7qfmzgnd1c9fqnzv4lfsm8t00000gn/T/self-audit-mnms-followup-qzwsbwbg/`.
It contains the synthetic cohort, `external.json` and `summary.json`.
The durable nested-config/alias regression is in the new external test file;
these temporary artifacts may be reaped.

These gates establish software compatibility and refusal behavior, not a
feasible restitution on a trained real model. No new native optimizer smoke was
needed for this data-preflight/launcher patch: native tiny training was verified
in the previous [joint integration review](joint_from_start_review.md).
It is not represented as a new run here. RTX 4070 and target PyTorch 2.4.1
gates remain unexecuted; no full suite or scientific training was run.

## Research verdict

**Candidate C remains research-worthy, not an established paper-level method.**
The earlier work produced deductions, counterexamples and a falsification plan,
not a trained-model effectiveness result. Software PASS does not raise novelty
confidence (current Dynamic Window 2/10; Candidate C 4/10, subjective assessments).

The new curriculum removes the scheduled freeze. It does not guarantee useful
feedback, accepted history, ordinary annotation conditioning or nonzero
refinement gradients. In a valid three-turn C path, the normal order is
ordinary→frozen restitution→ordinary; early rejection can eliminate the only
later ordinary evidence exposure. Count
`ordinary_annotation_real_evidence_attempts` separately from all evidence uses.

The main method hypothesis is whether historical support intervention yields
useful correction under noisy evidence beyond protected rollback and current-state
intervention using that same evidence. The next controlled study should first
establish opportunity/evidence quality, then compare actions on identical frozen
histories, then measure the full policy on all patients with patient-level
uncertainty. Report counterfactual output, transferred candidate and retained
state separately. Do not attribute a curriculum gain to geometry.

## Remaining limits

Real ACDC/M&Ms cohort completeness, stored label/axis declarations, timm ConvNeXt,
CUDA/AMP replay behavior, RTX 4070 memory, epoch timing, correction utility and
generalization were not verified here. Existing generic dataset APIs still allow
tolerant discovery; the stricter rule belongs specifically to supervised native
directory-split startup. NIfTI fallback remains its existing separate path.

No long training, complete test suite, new calibration, external hyperparameter
selection, or scientific efficacy study was run. The user can run a controlled
experiment after these software gates; the reports do not establish superiority,
clinical safety or novelty.
