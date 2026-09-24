from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "evaluate_cardiac_baseline_reference.py"


@pytest.fixture(scope="module")
def evaluator_module():
    spec = importlib.util.spec_from_file_location("cardiac_reference_evaluator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_is_fixed_and_rejects_changed_void_policy(tmp_path: Path, evaluator_module):
    source = ROOT / "configs" / "cardiac_reference_evaluator_v1.json"
    contract = evaluator_module.load_contract(source)
    assert contract["semantic_class_map"] == {"0": "BG", "1": "RV", "2": "MYO", "3": "LV", "4": "VOID"}

    changed = dict(contract)
    changed["void_policy"] = "VOID becomes background"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(evaluator_module.ReferenceEvaluationError, match="does not match"):
        evaluator_module.load_contract(path)


def test_inverse_preserves_void_and_gt_path_is_confined(tmp_path: Path, evaluator_module):
    semantic = np.asarray([[0, 1], [4, 3]], dtype=np.uint8)
    restored = evaluator_module.restore_semantic_slice(semantic, (2, 2))
    assert np.array_equal(restored, semantic)
    assert restored[1, 0] == evaluator_module.VOID

    record = {"source": {"locator": "training/patient004/patient004_frame01.nii"}}
    assert evaluator_module.gt_path_for_record(record, tmp_path) == tmp_path / "training/patient004/patient004_frame01_gt.nii"
    with pytest.raises(evaluator_module.ReferenceEvaluationError, match="safe relative"):
        evaluator_module.gt_path_for_record({"source": {"locator": "../outside.nii"}}, tmp_path)


def test_patient_aggregation_keeps_volume_equal_and_never_remaps_classes(evaluator_module):
    score = lambda value: {"dice": value, "iou": value / 2.0, "defined": True, "one_empty": False}
    per_volume = [
        {"patient_id": "p1", "scores": {"1": score(0.2), "2": score(0.4), "3": score(0.6)}},
        {"patient_id": "p1", "scores": {"1": score(0.4), "2": score(0.6), "3": score(0.8)}},
    ]
    patients = evaluator_module._patient_scores(per_volume)
    assert patients["p1"][1]["dice"] == pytest.approx(0.3)
    assert patients["p1"][2]["dice"] == pytest.approx(0.5)
    assert patients["p1"][3]["dice"] == pytest.approx(0.7)
    assert set(patients["p1"]) == {1, 2, 3}
