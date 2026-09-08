# Parallel execution update — user authorized, 2026-09-08

User: “cho nhiều worker nhiều gate đi, lâu quá”. This supersedes sequential dispatch,
not correctness gates. Claude Code via Orca remains implementer; Astra reviews independently.
No Proposal 2/3, training, real bank, architectural/objective changes or worker commits/push.
Remote CI remains waived as blocker; local integrated tests and compile remain required.

## Ownership and dispatch

| Worker | Exclusive ownership | Gate |
|---|---|---|
| W3.1 existing Claude | provenance.py, training/_utils.py checkpoint helpers, cache_validation_transitions.py, train_self_audit.py, test_checkpoint_binding.py | exact best/state binding, producer metadata, subset/recipe corrections |
| W3.2 Claude | evaluation/calibration_lineage.py if needed, threshold.py calibration serialization/consumer only, calibrate_threshold.py, audit_checkpoint.py, new test_calibration_lineage.py | strict matched/mismatched lineage, legacy fail-closed, CLI override, disjoint authorized cohort |
| W4 Claude | new evaluation/transition_bank.py, new export_transition_bank.py, new test_transition_bank.py | actual proposal/evaluator separation, schema, state IDs, HALT, GT-firewall and positive control |

Workers are not alone. Do not revert others or edit another ownership set. Shared helper
changes go by explicit request to owner. Do not edit evaluation/__init__.py in parallel;
use direct imports. W3.2 requests unified-runner integration from W3.1, never edits it itself.

Runtime-confirmed input accepted:

- W3.1: task `task_15d1ae019ddc`, dispatch `ctx_8d1708d5bcfb` (existing).
- W3.2: task `task_68ca5bc5de88`, dispatch `ctx_b0cee3308413`, terminal `term_af28d979-45d8-459a-8cba-372a69348ad8`.
- W4: task `task_cd3854a284b2`, dispatch `ctx_8bb7caf52c29`, terminal `term_0fb9c9dd-63d5-400c-bf6c-8faf36359ed4`.

Orca review gates created (all initially pending): producer `gate_1d403a643fc8`,
calibration `gate_f241942d8e36`, bank `gate_c1cae5f884a6`, integrated release
`gate_9cd548f6d459`. Gate tasks are separate from implementation tasks, so pending review
does not prevent independent worker progress.

Further parallelization: independent read-only Claude QA task `task_eae7eff0f81e`,
dispatch `ctx_33e22c2eaa7e`, terminal `term_d4ba427f-81e0-482b-a8af-722aa3b68ca2`
is input accepted. It owns only `parallel_invariant_qa.md`, not production or tests.
Scope: baseline/stop-gradient/HALT/GT-decision invariants and integration-route evidence.
Three implementers plus one independent QA worker now run concurrently.
Producer follow-on task is `task_65e1fdd77649` / `ctx_1729fcee7642` on the same terminal;
the initial producer dispatch is settled, not another concurrent editor.

User subsequently reported AGY quota reset and requested more agents if useful. Added AGY
task `task_d2e068c8d1f7`: exclusive NEW `tests/test_proposal1_cli_integration.py` plus
`agy_cli_integration_worker.md`; actual tiny cache → calibration → audit CLI regressions,
no production edits or overlap with bank tests. Initial AGY startup dispatch
`ctx_97b5ccd0a807` returned agent_prompt_stalled during CLI startup. Exact-terminal retry
`ctx_084f9722dc62` confirmed input_accepted on `term_66f7758d-b4f2-491f-b8bc-1301a16748eb`.
This is operational recovery, not test evidence. Five workers total; no duplicate AGY editor.

Producer revision receipt `msg_22b2c762fdf0`: metric/cohort correction and runner integration
delivered; worker reports 43 focused PASS and 337 full PASS/1 legacy schema assertion FAIL.
Root assigned that calibration-specific assertion migration to W3.2 (not general W1 edits).
Producer now reused for source-content identity follow-on `task_0ffee71e23df` /
`ctx_013be01cb875`; producing commit remains distinct from current functional source identity.
W4 can implement against current provenance interfaces; interface/integration PASS waits for
W3.1/W3.2 review. No dummy compatibility shim or mocks-only claim of integration.

Each worker reads docs.md, relevant current code, original wave task and this update, then
implements its bounded task and focused regressions. Original task statements requiring
prior sequential dispatch are overridden only for independent work, not final approval.

## Review gates

### Latest independent checks, 12:12 UTC

### Provider recovery, 12:15 UTC

Claude transcripts for all three remaining implementation dispatches show session quota
exhaustion. Exact workers were stopped; runtime confirmed terminal closure but returned
`dispatch_inactive` during stop bookkeeping. Release returned retained/identity_unproven;
no broad process cleanup was performed. Their working files are preserved.
Three AGY replacements hit first-start `agent_prompt_stalled`, then exact-terminal retries
all confirmed input_accepted:

- Bank task `task_4017b559c3ee`: dispatch `ctx_0ab9fbe2228d`.
- Source scope task `task_0133cd36806b`: dispatch `ctx_e7957f772da8`.
- Calibration closure task `task_5fd1fa4ebf62`: dispatch `ctx_f9d2b9acf721`.

Independent focused combined run: **177 passed, 3 failed in 59.82s**. Failures:
source-signature error-message expectation, legacy CLI cache missing lineage, legacy
precedence fixture missing expected runtime lineage. All AGY public CLI integration tests
passed in that run. AGY CLI receipt `msg_1fc2fc21d37b` accepted; terminal released.
New calibration review request also cross-checks duplicated artifact header identity with
the verified lineage; a correct nested lineage cannot excuse a contradictory declared SHA.
Gate A bounded PASS resolved as `gate_98f92c91fdfa`; integrated release still pending.

Root reran producer/runner tests: **43 passed in 10.61s**. Root reran baseline invariant
suite (entropy, hardening, regressions, masks, core, audit, volume): **67 passed,
1 pre-existing warning in 2.10s**. Root reran bank tests: **54 passed in 9.36s**.
These are focused software checks, not final integrated or trained-model evidence.

Read-only QA correction received as `msg_fca602166a9d`; its terminal was released.
Bank handoff `msg_c4f9e835ec68` received and terminal released. Final code inspection
identified additional semantic fields not cross-validated, so bank gate remains REVISE:
task `task_4017b559c3ee`, dispatch `ctx_5cc88aebb7d7`, new Claude terminal
`term_14a42e9b-5e3c-48b9-a3dd-98838438b4e4` input accepted. Owns only bank module,
bank tests and new `wave4_bank_final_review.md`; no concurrent bank editor.

Producer source identity review also remains REVISE: global directory-name exclusion
`data` skips active `src/self_audit/data`, and swallowed directory read errors could
certify a partial signature. Corrections/tests assigned to the existing producer owner.
Calibration migration briefly broke importers with stale measurement-hash symbols;
AGY and producer reported this independently. No compatibility bypass approved.

1. Producer gate: 18 then 27 focused tests passed on interim drafts, but subset regression
   and metadata corrections still under review. These counts are not final approval.
2. Calibration gate: runtime-derived expected identity, complete schema, every relevant
   mismatch hard-fails; allowed independent cohort positive test; source/producer distinct.
3. Bank gate: real tiny generation → evaluator → strict JSON; GT firewall powered control;
   no rows after rejected sample HALT; undefined Q retained.
4. Integration gate: combined actual diff, full tests/, source compile, git diff --check;
   architecture/stop-gradient/reject-HALT/objective invariants; no overlapping unresolved edits.

PASS/REVISE/BLOCK is recorded separately for each gate. Workers report exact commands/output
and unresolved dependencies; no whole-phase PASS while any required gate is unresolved.

## Tool constraints

## Final closure progress, 12:32 UTC

Producer receipt `msg_327f261fb321` accepted and released. Root **55 PASS in 11.42s**;
producer review gate `gate_1d403a643fc8` PASS. Scope v2 contains 57 source/config/wrapper
files including all five data modules. Baseline gate already PASS.

Root combined calibration/metric/legacy/CLI run **180 PASS in 59.90s** after migration.
Reprobe rejects original five header mismatches, but SHA=null and contract_version=1.5
still verified: calibration remains REVISE. Receipt `msg_3f768ff9a5e7` accepted operationally,
same terminal reused for exact closure task `task_66376ca66853`; first start stalled,
exact-terminal retry requested. Worker earlier 220 focused PASS is not final root gate.

Bank receipt `msg_c8f2b9fb423d` handled; same terminal reused for nonnegative/exact-class
statistics closure `task_e81001310856`, dispatch `ctx_7d31d78ea17e` after exact-terminal
retry. Root reran **71 PASS in 12.63s** on that correction. Awaiting final receipt.
Worker report statements about future clinical certification are not a promise or an
approved research claim: only synthetic software readiness is being evaluated here.

All final gates remain local under the user's remote-CI waiver. No commit/push or full
training; source/config/split identities of historical artifacts have not been rewritten.

Shell prefix `rtk`. Use apply_patch for edits. Coordinator verified executable at
`/Users/alvinluong/.codex/tmp/arg0/codex-arg0bSOkbZ/apply_patch`; workers may use this exact
path without scanning runtime directories. If unavailable, return an authored patch for
mechanical coordinator application. Do not substitute Python writes.
