# W&B v3 Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional single-run W&B tracking to the Self-Audit v3 orchestrator and document the server command without changing the no-GT pipeline contract.

**Architecture:** A small optional tracker module owns one W&B run in the parent orchestrator. The parent logs stage lifecycle events and consumes the existing teacher, evaluation, and student JSON artifacts after each child process completes. W&B failures degrade to local-only logs, while v3 artifacts remain authoritative.

**Tech Stack:** Python 3.10+, argparse, standard-library JSON/path handling, optional `wandb==0.30.0`, pytest, existing v3 subprocess runner.

**Spec:** `docs/superpowers/specs/2026-10-07-wandb-v3-design.md`

## Global Constraints

- W&B is opt-in; the existing v3 command remains behaviorally unchanged when `--wandb` is absent.
- Authentication comes only from `WANDB_API_KEY`; no secret is added to source, config, logs, or tests.
- Exactly one parent W&B run represents teacher → evaluation → optional student.
- W&B telemetry must never change the no-GT data, freeze, checkpoint, or evaluation contracts.
- W&B SDK absence, initialization failure, log failure, and finish failure degrade to local execution.
- Model checkpoints and pseudo-label arrays are not uploaded automatically.
- Push the final commit directly to `main`, preserving unrelated `README_AGENT_CMR_DATASETS.md`.

## Review Focus

- W&B is unavailable or the server has no outbound network: pipeline still produces normal local artifacts; test in Task 2.
- Teacher/student JSON contains nested, non-finite, or non-scalar values: telemetry is sanitized without aborting; test in Task 1.
- Metrics from multiple stages reuse local step numbers: one parent run remains monotonic through namespaced logs; test in Task 1.
- Default invocation has no W&B side effects: parser and runner default test in Task 3.
- API keys must not appear in command construction or payloads: source/test inspection in Task 4.

---

### Task 1: Add the W&B tracker contract and failing tests

**Files:**
- Create: `tests/test_pseudolabel_wandb.py`
- Create: `src/self_audit_pseudolabel/wandb_v3.py` (after the red test)

**Interfaces:**
- Produces `WandbV3Tracker(enabled, mode, project, entity, run_name, run_dir, config)` with `log`, `log_history`, `log_json`, `set_summary`, `finish`, and `warnings` behavior.

- [ ] **Step 1: Write failing tests**

  Test disabled mode makes no SDK calls; enabled mode initializes one run with project/entity/name/mode; history is logged under a stage prefix with finite scalar sanitation; initialization/logging errors become warnings and do not raise.

- [ ] **Step 2: Run the focused test and verify it fails**

  Run: `python -m pytest -q tests/test_pseudolabel_wandb.py`

  Expected: collection/import failure because the tracker module does not exist yet.

- [ ] **Step 3: Implement the minimal tracker**

  Add an optional-import adapter that never imports W&B when disabled, calls `wandb.init` once when enabled, flattens nested JSON keys with `/`, converts non-finite numbers to `None`, logs without explicit regressing steps, and records warning strings instead of raising SDK exceptions.

- [ ] **Step 4: Run the focused test and verify it passes**

  Run: `python -m pytest -q tests/test_pseudolabel_wandb.py`

- [ ] **Step 5: Commit the task**

  Commit message: `feat: add optional v3 wandb tracker`

### Task 2: Add parser flags and parent-run orchestration

**Files:**
- Modify: `scripts/run_full_pipeline_v3.py`
- Modify: `tests/test_pseudolabel_wandb.py`

**Interfaces:**
- `run_full_pipeline_v3.py` accepts `--wandb`, `--no-wandb`, `--wandb-mode`, `--wandb-project`, `--wandb-entity`, and `--wandb-run-name`.
- The parent creates one tracker per pipeline, logs `pipeline/*`, `teacher/*`, `evaluation/*`, and `student/*`, and finishes it in a `finally` path.

- [ ] **Step 1: Extend failing tests**

  Add parser tests for disabled-by-default behavior and explicit online settings; add orchestration-level tests using a fake tracker or mocked subprocess boundary to verify teacher/evaluation/student JSON is sent to the correct stage prefixes.

- [ ] **Step 2: Run tests and verify the new assertions fail**

  Run: `python -m pytest -q tests/test_pseudolabel_wandb.py`

  Expected: parser/import or missing-integration failures.

- [ ] **Step 3: Implement minimal integration**

  Extract a parser builder if needed, construct the tracker after creating the unique run directory, log stage start/finish and environment metadata, call `log_history` on `teacher/train_metrics.json` and student result history, call `log_json` on evaluation and final summary, write tracker warnings into `pipeline.log`, and finalize in `finally` without changing child commands.

- [ ] **Step 4: Run focused tests and CLI smoke checks**

  Run: `python -m pytest -q tests/test_pseudolabel_wandb.py`

  Run: `python scripts/run_full_pipeline_v3.py --help`

- [ ] **Step 5: Commit the task**

  Commit message: `feat: wire wandb into v3 pipeline`

### Task 3: Document server setup and command usage

**Files:**
- Modify: `README.md`
- Modify: `docs/pseudolabel_v3_review_runbook.md`

- [ ] **Step 1: Add documentation**

  Document `WANDB_API_KEY`, online/offline modes, the new v3 flags, the full teacher/evaluation/student command, and the fact that offline runs can be synced later. Explicitly state that the API key must not be placed in Git or the command line.

- [ ] **Step 2: Verify docs and parser examples**

  Run: `python scripts/run_full_pipeline_v3.py --help` and inspect the documented command for flag parity.

- [ ] **Step 3: Commit the task**

  Commit message: `docs: document v3 wandb tracking`

### Task 4: Full verification and push to main

**Files:**
- Verify: all modified files above

- [ ] **Step 1: Run focused and existing regression tests**

  Run: `python -m pytest -q tests/test_pseudolabel_wandb.py tests/test_pseudolabel_full_pipeline.py tests/test_pseudolabel_system_v3.py tests/test_pseudolabel_api_compat.py`

- [ ] **Step 2: Run syntax/import verification**

  Run: `python -m py_compile scripts/run_full_pipeline_v3.py src/self_audit_pseudolabel/wandb_v3.py`

- [ ] **Step 3: Inspect diff and secret safety**

  Run: `git diff --check`, inspect `git diff --stat`, and verify no token-like value or `WANDB_API_KEY` value is committed.

- [ ] **Step 4: Commit the complete change on main**

  Stage only the spec, plan, tracker, runner, tests, and docs; leave `README_AGENT_CMR_DATASETS.md` untracked.

- [ ] **Step 5: Push `main`**

  Run: `git push origin main` and report the resulting commit SHA. If the sandbox blocks `.git` or network access, request the required Git permission and retry; do not rewrite history or force-push.
