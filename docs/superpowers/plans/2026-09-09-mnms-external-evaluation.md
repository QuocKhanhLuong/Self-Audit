# M&Ms External Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing four-class M&Ms testing cohort executable as a frozen external evaluation after ACDC A→B→C training.

**Architecture:** Add real ACDC/M&Ms dataset dispatch and validation, keep the existing preprocessed `preprocessed_data/mnm` arrays as deployment input, and add an evaluation-only M&Ms entrypoint. Reuse shared volume inference and cohort code while keeping training, checkpoint selection, and threshold calibration strictly ACDC-only.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, YAML, pytest, existing Self-Audit evaluation modules.

**Spec:** `docs/superpowers/specs/2026-09-09-phase-b-oom-mnms-external-evaluation-design.md`

## Global Constraints

- Use `preprocessed_data/mnm`, never `preprocessed_data/mnm_binary`.
- External evaluation consumes only the `testing` split and does not train.
- Do not infer ED/ES from `<patient>_tXX`; report timepoints/all cohort only.
- Record the explicit M&Ms-to-ACDC mapping `{0: 0, 1: 3, 2: 2, 3: 1}`.
- Report `metric_space: volume_resized` and never claim native geometry from the resized arrays.
- ACDC and M&Ms remain outside Git; only code, configs, tests, and documentation are committed.

---

### Task 1: Add failing dataset-dispatch and external-validation tests

**Files:**
- Modify: `tests/test_self_audit_data.py`
- Create: `tests/test_mnms_external_flow.py`
- Read-only reference: `src/self_audit/data/mnms.py`, `src/self_audit/training/_utils.py`

**Interfaces:**
- Consumes: temporary `volumes/*.npy` and `masks/*.npy` fixtures.
- Produces: failing tests defining `build_patient_dataset` dispatch, M&Ms split validation, patient IDs, and four-class rejection behavior.

- [ ] **Step 1: Write dataset factory tests**

```python
import numpy as np
import pytest

from src.self_audit.training._utils import build_patient_dataset, validate_dataset_splits


def _write_mnms_split(root, split="testing"):
    volumes = root / split / "volumes"
    masks = root / split / "masks"
    volumes.mkdir(parents=True)
    masks.mkdir(parents=True)
    volume = np.zeros((224, 224, 4), dtype=np.float32)
    mask = np.zeros((224, 224, 4), dtype=np.uint8)
    mask[20:30, 20:30, 1] = 1
    np.save(volumes / "A0S9V9_t00.npy", volume)
    np.save(masks / "A0S9V9_t00.npy", mask)


def test_mnms_config_dispatches_to_mnms_dataset(tmp_path):
    _write_mnms_split(tmp_path)
    dataset = build_patient_dataset(
        {"dataset": "mnms", "data_root": str(tmp_path), "image_size": 256},
        split="testing",
        train=False,
    )
    assert dataset.__class__.__name__ == "MNMSDataset"
    assert dataset.records[0].case_id == "A0S9V9_t00"


def test_mnms_validation_reports_real_pairs(tmp_path):
    _write_mnms_split(tmp_path)
    result = validate_dataset_splits(
        {"dataset": "mnms", "data_root": str(tmp_path), "test_split": "testing"}
    )
    assert result["validated"] is True
    assert result["case_counts"]["testing"] == 1
    assert result["dataset"] == "mnms"


def test_mnms_binary_contract_is_rejected(tmp_path):
    _write_mnms_split(tmp_path)
    with pytest.raises(ValueError, match="four-class"):
        build_patient_dataset(
            {"dataset": "mnms", "data_root": str(tmp_path), "num_classes": 2},
            split="testing",
            train=False,
        )
```

Add tests for a missing pair and an unknown mask label; each must assert a specific `FileNotFoundError` or `ValueError` message.

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m pytest -q tests/test_mnms_external_flow.py tests/test_self_audit_data.py`

Expected: FAIL because the factory still always returns `ACDCDataset` and external validation returns `validated=False`.

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/test_mnms_external_flow.py tests/test_self_audit_data.py
git commit -m "test: define executable M&Ms external dataset flow"
```

### Task 2: Implement dataset factory dispatch and M&Ms validation

**Files:**
- Modify: `src/self_audit/training/_utils.py:280-410`
- Modify: `src/self_audit/data/mnms.py` only for missing preflight helpers or metadata-safe discovery.
- Test: `tests/test_mnms_external_flow.py`, `tests/test_self_audit_data.py`

**Interfaces:**
- Consumes: `config["dataset"]`, `config["data_root"]`, split aliases, `raw_to_acdc`, and optional `num_classes`.
- Produces: `build_patient_dataset(config, split, train)` returning `ACDCDataset` or `MNMSDataset`; `validate_dataset_splits` returning validated counts/signature for M&Ms.

- [ ] **Step 1: Add explicit dataset dispatch**

Implement this branch before the existing ACDC import:

```python
dataset_name = str(config.get("dataset", "acdc")).strip().lower()
num_classes = int(config.get("num_classes", config.get("model", {}).get("num_classes", 4)))
if dataset_name == "mnms":
    if num_classes != 4:
        raise ValueError("M&Ms external evaluation requires the four-class contract")
    from self_audit.data.mnms import MNMSDataset
    kwargs = {
        "data_root": config.get("data_root", "preprocessed_data/mnm"),
        "split": split,
        "image_size": validate_image_size(config.get("image_size")),
        "augment": False,
        "raw_to_acdc": config.get("raw_to_acdc"),
    }
    for key in ("depth_axis", "expected_slices", "max_cache"):
        if key in config:
            kwargs[key] = config[key]
    return MNMSDataset(**kwargs)
if dataset_name != "acdc":
    raise ValueError(f"Unsupported dataset {dataset_name!r}")
```

Keep the current ACDC branch unchanged after the dispatch.

- [ ] **Step 2: Implement real M&Ms split validation**

For `dataset_name == "mnms"`, call `discover_mnms_records(data_root, split=split_name)`, inspect every record with `load_array`, compare image/mask shapes, collect the union of raw labels, and compute case/patient counts plus `compute_split_signature`. Return `validated=True`, `dataset`, `case_counts`, `patient_counts`, `slice_counts`, `effective_identities`, `label_values`, `raw_to_acdc`, and `membership_signature`. Do not return the old external-dataset shortcut. When validating the protocol's `testing` split, pass `test_split="testing"` and validate that split; do not require nonexistent train/validation files just to run an external evaluation.

Validation must not mutate files and must fail for a missing root, an empty split, a pair mismatch, shape mismatch, or unknown label. Use the existing `MNMSClassMapping` to validate the mapping before constructing the return block.

- [ ] **Step 3: Run focused tests**

Run: `python -m pytest -q tests/test_mnms_external_flow.py tests/test_self_audit_data.py tests/test_self_audit_regressions.py`

Expected: PASS.

- [ ] **Step 4: Commit dispatch**

```bash
git add src/self_audit/training/_utils.py src/self_audit/data/mnms.py tests/test_mnms_external_flow.py tests/test_self_audit_data.py
git commit -m "feat: dispatch and validate four-class M&Ms data"
```

### Task 3: Lock the external protocol config

**Files:**
- Modify: `configs/self_audit_acdc_to_mnms.yaml`
- Test: `tests/test_self_audit_regressions.py`

**Interfaces:**
- Consumes: existing ACDC Phase A/B/C configs and the server path override.
- Produces: a protocol config with `source_training` and `external_test` blocks, four-class model identity, explicit mapping, and fixed tau.

- [ ] **Step 1: Add a config contract test**

```python
def test_acdc_to_mnms_protocol_declares_external_test_contract():
    config = load_config(ROOT / "configs" / "self_audit_acdc_to_mnms.yaml")
    external = config["external_test"]
    assert external["dataset"] == "mnms"
    assert external["split"] == "testing"
    assert external["raw_to_acdc"] == {0: 0, 1: 3, 2: 2, 3: 1}
    assert config["num_classes"] == 4
    assert config["audit"]["tau_accept"] == 0.0
```

- [ ] **Step 2: Rewrite the protocol config**

Use `source_training.dataset: acdc`, `source_training.phases: [annotation, auditor, joint]`, `external_test.dataset: mnms`, `external_test.split: testing`, `external_test.data_root: preprocessed_data/mnm`, and the explicit mapping. Keep `image_size: 256`, model architecture matching the Phase-C checkpoint, and `audit.neutral_margin: 0.005`, `audit.t_max: 3`, `audit.tau_accept: 0.0`.

- [ ] **Step 3: Run config tests and commit**

Run: `python -m pytest -q tests/test_self_audit_regressions.py::test_acdc_to_mnms_protocol_declares_external_test_contract`

Expected: PASS.

```bash
git add configs/self_audit_acdc_to_mnms.yaml tests/test_self_audit_regressions.py
git commit -m "config: lock ACDC-to-M&Ms external protocol"
```

### Task 4: Extract reusable cohort evaluation and add external evaluator tests

**Files:**
- Create: `src/self_audit/evaluation/cohort.py`
- Modify: `scripts/audit_checkpoint.py` to import the moved cohort helper and preserve its existing diagnostic behavior.
- Create: `scripts/evaluate_external_mnms.py`
- Create: `tests/test_external_mnms_evaluator.py`

**Interfaces:**
- Consumes: `evaluate_volume_cohort` behavior from `audit_checkpoint.py`, a frozen model, `MNMSDataset`, and a fixed tau.
- Produces: `evaluate_external_mnms.py --config --checkpoint --data-root --split --tau-accept --device --output` and a versioned independent-evaluation JSON report.

- [ ] **Step 1: Write evaluator safety tests**

Use a tiny temporary M&Ms fixture and a CPU model fixture. Import `torch` in the test module. Assert the evaluator:

```python
payload = run_external_evaluation(
    config=config,
    checkpoint=checkpoint,
    data_root=tmp_path,
    split="testing",
    tau_accept=0.0,
    device=torch.device("cpu"),
    output=tmp_path / "external_mnms.json",
)
assert payload["evidence_class"] == "independent_external_evaluation"
assert payload["dataset"] == "mnms"
assert payload["split"] == "testing"
assert payload["metric_space"] == "volume_resized"
assert payload["phase_partition"] == "unavailable_without_authoritative_metadata"
assert payload["tau_accept_source"] in {"cli:--tau_accept", "config.audit.tau_accept"}
```

Patch `torch.optim` and threshold-sweep functions to raise if called; the test must still pass. Add a test that omitting `--tau-accept` uses the config's fixed value and never reads M&Ms to calibrate it.

- [ ] **Step 2: Run evaluator tests to verify failure**

Run: `python -m pytest -q tests/test_external_mnms_evaluator.py`

Expected: FAIL because the shared cohort module and external entrypoint do not exist.

- [ ] **Step 3: Move shared cohort logic without semantic changes**

Move `evaluate_volume_cohort` and its private cohort aggregation helpers from `scripts/audit_checkpoint.py` to `src/self_audit/evaluation/cohort.py`. Preserve `COMPARISON_MODES`, `volume_resized`, empty-class policy, and all per-case/cohort metric fields. Update `audit_checkpoint.py` imports and run its existing tests before adding M&Ms-specific behavior.

- [ ] **Step 4: Implement the evaluation-only entrypoint**

Implement `run_external_evaluation(config, checkpoint, data_root, split, tau_accept, device, output)` and a CLI wrapper. It must validate the external config and dataset, bind the checkpoint, set `model.eval()`, select the fixed tau from CLI or config, call the shared cohort evaluator, and write a report containing checkpoint binding, cohort signature, counts, mapping, metric-space declaration, unavailable phase partition, comparison modes, and metrics. Do not instantiate an optimizer/scheduler or expose calibration/sweep flags.

Before dataset construction, normalize the protocol block into the flat dataset config expected by the factory: copy the root config, overlay `external_test`, set `dataset`, `data_root`, `test_split`/`split`, `num_classes`, and `raw_to_acdc`, then pass that normalized mapping to `validate_dataset_splits` and `build_patient_dataset`.

- [ ] **Step 5: Run focused evaluator tests and commit**

Run: `python -m pytest -q tests/test_external_mnms_evaluator.py tests/test_remediation_evaluation.py tests/test_geometry_safety.py`

Expected: PASS.

```bash
git add src/self_audit/evaluation/cohort.py scripts/audit_checkpoint.py scripts/evaluate_external_mnms.py tests/test_external_mnms_evaluator.py
git commit -m "feat: add frozen M&Ms external evaluator"
```

### Task 5: Server smoke test and documentation

**Files:**
- Modify: `README.md`
- Modify: `docs.md`
- Test: `tests/test_mnms_external_flow.py`

**Interfaces:**
- Consumes: completed dispatch, config, and evaluator.
- Produces: reproducible server commands and a documented ACDC-only → M&Ms-only boundary.

- [ ] **Step 1: Add a command-contract test**

Assert that the documented command includes `phase_c_best.pt`, `--split testing`, `--tau-accept`, and `preprocessed_data/mnm`, and does not invoke `train_self_audit.py` with an M&Ms config.

- [ ] **Step 2: Document the server flow**

Add the following sequence to `README.md` and `docs.md`:

```bash
python scripts/train_self_audit.py \
  --config_a configs/self_audit_annotation.yaml \
  --config_b configs/self_audit_auditor.yaml \
  --config_c configs/self_audit_joint.yaml \
  --output_dir weights/self_audit_full

python scripts/evaluate_external_mnms.py \
  --config configs/self_audit_acdc_to_mnms.yaml \
  --checkpoint weights/self_audit_full/phase_c_best.pt \
  --data-root preprocessed_data/mnm \
  --split testing \
  --tau-accept 0.0 \
  --device cuda \
  --output reports/external_mnms.json
```

Document that `data/ACDC`, `preprocessed_data/ACDC`, and `preprocessed_data/mnm` remain untracked data paths, and that `mnm_binary` is incompatible with the four-class checkpoint.

- [ ] **Step 3: Run full verification**

Run: `python -m pytest -q tests/test_mnms_external_flow.py tests/test_external_mnms_evaluator.py tests/test_self_audit_data.py tests/test_self_audit_regressions.py tests/test_remediation_evaluation.py tests/test_geometry_safety.py`

Expected: PASS.

Run on the server after setting `PYTHONPATH=src`:

```bash
python scripts/evaluate_external_mnms.py --config configs/self_audit_acdc_to_mnms.yaml --checkpoint weights/self_audit_full/phase_c_best.pt --data-root preprocessed_data/mnm --split testing --tau-accept 0.0 --device cuda --output reports/external_mnms.json
```

Expected: validation reports the discovered 272 testing pairs, inference completes, and the report marks phase partition unavailable rather than inventing ED/ES.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs.md tests/test_mnms_external_flow.py
git commit -m "docs: document ACDC-only training and M&Ms external test"
```
