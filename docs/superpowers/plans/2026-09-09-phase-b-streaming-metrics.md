# Phase-B Streaming Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove validation-time GPU OOM by reducing local audit maps into streaming CPU summaries while preserving every Phase-B metric and checkpoint-selection value.

**Architecture:** A small `TransitionMetricAccumulator` owns a 3x3 local-label confusion matrix and CPU vectors of transition-level deltas. `_auditor_batch` produces summaries at the batch boundary; `validate_auditor_epoch` merges provenance accumulators and derives the existing namespaces without retaining dense tensors or a duplicated combined bucket.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-phase-b-oom-mnms-external-evaluation-design.md`

## Global Constraints

- Phase-B training loss composition and provenance names are unchanged.
- `primary_metric` remains on-policy AUROC, then on-policy accuracy, then `nan`.
- `transition_audit_metrics` keeps its neutral-margin and empty-class semantics.
- No dataset, checkpoint, or configuration file is added to Git by this workstream.
- No GPU memory workaround based only on `empty_cache()` or allocator environment variables is accepted.

---

### Task 1: Add failing parity tests for a streaming accumulator

**Files:**
- Create: `tests/test_transition_accumulator.py`
- Read-only reference: `src/self_audit/evaluation/metrics.py`

**Interfaces:**
- Consumes: synthetic local logits/targets and transition delta vectors.
- Produces: tests that define the required constructor, `update`, `merge`, and `finalize` behavior for Task 2.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import torch

from src.self_audit.evaluation.metrics import transition_audit_metrics
from src.self_audit.evaluation.transition_accumulator import TransitionMetricAccumulator


def _summary(logits, target, delta_q, delta_dice):
    prediction = logits.argmax(dim=1).reshape(-1)
    truth = target.reshape(-1)
    encoded = truth * 3 + prediction
    confusion = torch.bincount(encoded, minlength=9).reshape(3, 3)
    return confusion, delta_q, delta_dice


def test_streaming_finalize_matches_legacy_metrics():
    torch.manual_seed(7)
    logits = torch.randn(5, 3, 6, 6)
    target = torch.randint(0, 3, (5, 6, 6))
    delta_q = torch.tensor([0.2, -0.4, 0.0, 0.3, -0.1])
    delta_dice = torch.tensor([0.4, -0.3, 0.0, 0.2, -0.2])
    confusion, q, d = _summary(logits, target, delta_q, delta_dice)
    accumulator = TransitionMetricAccumulator()
    accumulator.update(local_confusion=confusion, delta_pred=q, delta_target=d)
    actual = accumulator.finalize(neutral_margin=0.005)
    expected = transition_audit_metrics(
        logits, target, delta_q, delta_dice, neutral_margin=0.005
    )
    for key in ("improve_regress_accuracy", "auroc", "auprc", "correlation_delta_q_delta_dice"):
        assert actual[key] == pytest.approx(float(expected[key]), nan_ok=True)
    assert actual["local_fix_f1"] == pytest.approx(float(expected["local_fix_f1"]))
    assert actual["local_regress_f1"] == pytest.approx(float(expected["local_regress_f1"]))


def test_merge_is_equal_to_one_update():
    left = TransitionMetricAccumulator()
    right = TransitionMetricAccumulator()
    all_data = (
        torch.tensor([[4, 1, 0], [2, 5, 1], [0, 2, 6]]),
        np.array([0.1, -0.2], dtype=np.float32),
        np.array([0.2, -0.3], dtype=np.float32),
    )
    left.update(local_confusion=all_data[0], delta_pred=all_data[1][:1], delta_target=all_data[2][:1])
    right.update(local_confusion=torch.zeros(3, 3), delta_pred=all_data[1][1:], delta_target=all_data[2][1:])
    left.merge(right)
    one = TransitionMetricAccumulator()
    one.update(local_confusion=all_data[0], delta_pred=all_data[1], delta_target=all_data[2])
    assert left.finalize(neutral_margin=0.005) == one.finalize(neutral_margin=0.005)


def test_accumulator_rejects_dense_local_arrays():
    accumulator = TransitionMetricAccumulator()
    with pytest.raises(ValueError, match="3x3"):
        accumulator.update(
            local_confusion=torch.zeros(2, 3, 6, 6),
            delta_pred=torch.tensor([0.1]),
            delta_target=torch.tensor([0.2]),
        )
```

Add `import pytest` at the top of the test file. The first run must fail because the module does not exist.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest -q tests/test_transition_accumulator.py`

Expected: FAIL with `ModuleNotFoundError` for `transition_accumulator`.

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/test_transition_accumulator.py
git commit -m "test: define streaming Phase-B metric parity"
```

### Task 2: Implement `TransitionMetricAccumulator`

**Files:**
- Create: `src/self_audit/evaluation/transition_accumulator.py`
- Modify: `src/self_audit/evaluation/__init__.py` only if the project exports public evaluation helpers there.
- Test: `tests/test_transition_accumulator.py`

**Interfaces:**
- Consumes: a CPU or GPU `3x3` integer confusion matrix and one-dimensional delta arrays.
- Produces: `TransitionMetricAccumulator.update`, `.merge`, and `.finalize` with the metric keys currently returned by `transition_audit_metrics`.

- [ ] **Step 1: Implement validation and update**

Use this concrete shape policy:

```python
def update(self, *, local_confusion, delta_pred, delta_target):
    confusion = torch.as_tensor(local_confusion)
    if tuple(confusion.shape) != (3, 3):
        raise ValueError(f"local_confusion must be 3x3, got {tuple(confusion.shape)}")
    q = np.asarray(delta_pred.detach().cpu() if torch.is_tensor(delta_pred) else delta_pred, dtype=np.float32).reshape(-1)
    d = np.asarray(delta_target.detach().cpu() if torch.is_tensor(delta_target) else delta_target, dtype=np.float32).reshape(-1)
    if q.shape != d.shape:
        raise ValueError(f"delta_pred and delta_target must match, got {q.size} vs {d.size}")
    if not np.isfinite(q).all() or not np.isfinite(d).all():
        raise ValueError("transition deltas must be finite")
    self.local_confusion += confusion.detach().cpu().to(torch.int64).numpy()
    self.delta_pred_parts.append(q.copy())
    self.delta_target_parts.append(d.copy())
    self.transition_count += int(q.size)
```

The constructor initializes `local_confusion=np.zeros((3,3), dtype=np.int64)`, empty lists, and `transition_count=0`.

- [ ] **Step 2: Implement merge and finalize**

`merge` adds confusion matrices, extends delta lists, and adds counts. `finalize` concatenates only one-dimensional CPU vectors. Reuse `classify_delta`, `binary_auroc`, `binary_auprc`, and `_safe_correlation` from `metrics.py`; compute local F1 from confusion rows/columns so no dense local map is reconstructed. Return `nan` for an empty accumulator and include `transition_count`, `neutral_count`, `beneficial_count`, `harmful_count`, and `ranking_count` exactly as the existing function does.

- [ ] **Step 3: Run tests and verify parity**

Run: `python -m pytest -q tests/test_transition_accumulator.py`

Expected: PASS.

- [ ] **Step 4: Commit the accumulator**

```bash
git add src/self_audit/evaluation/transition_accumulator.py src/self_audit/evaluation/__init__.py tests/test_transition_accumulator.py
git commit -m "feat: add streaming transition metric accumulator"
```

### Task 3: Integrate summaries into auditor validation

**Files:**
- Modify: `src/self_audit/training/train_auditor.py:230-293,527-603`
- Test: `tests/test_transition_accumulator.py`
- Test: `tests/test_self_audit_regressions.py`

**Interfaces:**
- Consumes: `TransitionMetricAccumulator` from Task 2.
- Produces: `_auditor_batch(..., collect=True)` returns CPU summary records; `validate_auditor_epoch` returns the existing namespace and legacy keys.

- [ ] **Step 1: Add a structural regression test**

Exercise `_auditor_batch` with a tiny mock model and `collect=True`. Assert that each returned summary contains `local_confusion.shape == (3,3)`, one-dimensional CPU delta arrays, and no tensor with four or more dimensions. Assert that `collect=False` keeps the training path free of summaries.

- [ ] **Step 2: Replace dense collection at the batch boundary**

Inside `_auditor_batch`, initialize `batch_summaries` before the transition loop and replace the four dense values in `row` with:

```python
batch_summaries: dict[str, list[dict[str, torch.Tensor]]] = {
    name: [] for name in PROVENANCE_KINDS
}

if collect:
    prediction = audit_output.local_logits.detach().argmax(dim=1)
    encoded = targets.local.detach().reshape(-1) * 3 + prediction.reshape(-1)
    confusion = torch.bincount(encoded, minlength=9).reshape(3, 3)
    summary = {
        "local_confusion": confusion,
        "delta_pred": audit_output.delta_q.detach().reshape(-1).float().cpu(),
        "delta_target": targets.delta_dice.detach().reshape(-1).float().cpu(),
    }
    batch_summaries[kind].append(summary)
```

Keep `group_counts` and local label counts unchanged. Do not create a `combined` dense bucket.

- [ ] **Step 3: Replace end-of-batch `torch.cat`**

Return `transition_summaries` keyed by `on_policy` and `synthetic` when `collect=True`; return an empty summary mapping when `collect=False`. In validation, instantiate one accumulator per provenance and call `.update` for every summary immediately. Construct combined by cloning/merging the two accumulators after the loader ends. Remove the old `transition_data` field because the only reader is this validation function; update that reader to use `details.get("transition_summaries", {})`.

- [ ] **Step 4: Run focused regression tests**

Run: `python -m pytest -q tests/test_transition_accumulator.py tests/test_self_audit_regressions.py`

Expected: PASS with no changed metric keys.

- [ ] **Step 5: Commit the integration**

```bash
git add src/self_audit/training/train_auditor.py tests/test_transition_accumulator.py tests/test_self_audit_regressions.py
git commit -m "fix: stream Phase-B validation metrics off GPU"
```

### Task 4: Full verification and memory evidence

**Files:**
- Modify: no production files unless a test exposes an integration mismatch.
- Test: `tests/test_transition_accumulator.py`

**Interfaces:**
- Consumes: completed Workstream-A implementation.
- Produces: evidence that full validation is independent of validation-loader length in GPU memory.

- [ ] **Step 1: Run all metric and training tests**

Run: `python -m pytest -q tests/test_transition_accumulator.py tests/test_self_audit_regressions.py tests/test_remediation_metrics.py tests/test_metric_contract_and_replay.py`

Expected: PASS.

- [ ] **Step 2: Run an optional CUDA smoke test**

On the target server, run Phase-B validation once with `--max_val_batches 2`, then once without the cap. Record `torch.cuda.max_memory_allocated()` after warm-up and at epoch end. The uncapped run must complete without OOM; the retained accumulator state must remain CPU-only and fixed-size for local labels.

- [ ] **Step 3: Run syntax and repository checks**

Run: `python -m compileall -q src scripts tests`

Expected: exit code 0.

- [ ] **Step 4: Commit verification notes if needed**

Do not commit raw logs or CUDA machine-specific output. The implementation commit is complete only after the focused and full tests pass.
