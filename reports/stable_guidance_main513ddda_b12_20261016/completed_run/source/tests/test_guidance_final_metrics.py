import importlib.util
from pathlib import Path

import numpy as np
import pytest

from self_audit_guidance.evaluate import aggregate_scores, score_volume

spec = importlib.util.spec_from_file_location(
    "guidance_final_metrics", Path(__file__).parents[1] / "scripts/publish_guidance_metrics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report(rows):
    return dict(frames=rows, summary=aggregate_scores(rows))


def test_precision_recall_iou_with_unknown_false_negative():
    reference = np.array([3, 3, 0, 0]).reshape(2, 2, 1)
    prediction = np.array([3, 255, 3, 0]).reshape(2, 2, 1)
    rows = [dict(patient_id="p1", metrics=score_volume(prediction, reference))]
    result = module.evaluation_scalars(report(rows))
    assert result["val/dice"] == .5
    assert result["val/iou"] == pytest.approx(1 / 3)
    assert result["val/precision"] == .5
    assert result["val/recall"] == .5
    assert result["val/coverage"] == .75


def test_patient_macro_not_frame_weighted_and_missing_prediction():
    reference = np.full((2, 2, 1), 3)
    good = score_volume(reference, reference)
    unknown = score_volume(np.full_like(reference, 255), reference)
    rows = [dict(patient_id="p1", metrics=good)] * 3 + [dict(patient_id="p2", metrics=unknown)]
    result = module.evaluation_scalars(report(rows))
    assert result["val/dice"] == .5
    assert result["val/iou"] == .5
    assert result["val/recall"] == .5
    assert result["val/LV/precision_defined_patients"] == 1
    assert result["val/LV/recall_defined_patients"] == 2


def test_all_unknown_is_not_perfect_and_auroc_is_not_fabricated():
    reference = np.full((2, 2, 1), 3)
    rows = [dict(patient_id="p1", metrics=score_volume(np.full_like(reference, 255), reference))]
    result = module.evaluation_scalars(report(rows))
    assert result["val/dice"] == 0
    assert result["val/iou"] == 0
    assert result["val/recall"] == 0
    assert result["val/precision"] is None
    assert not any("auroc" in key for key in result)


def test_inconsistent_evaluation_reduction_refused():
    reference = np.full((2, 2, 1), 3)
    r = report([dict(patient_id="p1", metrics=score_volume(reference, reference))])
    r["summary"]["macro_foreground_dice"] = .1
    with pytest.raises(ValueError, match="reduction"):
        module.evaluation_scalars(r)
