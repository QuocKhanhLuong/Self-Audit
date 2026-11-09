import json
import csv

import numpy as np
import pytest

from self_audit.metrics import ConfusionAccumulator, append_csv, append_jsonl, class_metrics, confusion_matrix, epoch_rows
from self_audit.candidate_a.engine import _validation_rows


def test_unknown_prediction_counts_as_false_negative_and_ignore_is_excluded():
    reference = np.array([[0, 1, 255, 3]], dtype=np.int64)
    prediction = np.array([[0, 255, 2, 3]], dtype=np.int64)
    matrix = confusion_matrix(reference, prediction)
    assert matrix.shape == (4, 5)
    assert matrix[1, 4] == 1
    assert matrix[3, 3] == 1
    assert int(matrix.sum()) == 3


def test_absent_class_metrics_are_explicit_and_never_inflate_dice(tmp_path):
    matrix = np.zeros((4, 5), dtype=np.int64)
    matrix[0, 0] = 3
    matrix[0, 2] = 2  # false positives for absent MYO
    row = class_metrics(matrix)[2]
    assert row["status"] == "false_positive_only"
    assert row["dice"] is None and row["f1"] is None and row["recall"] is None
    assert row["precision"] == 0.0
    assert row["support"] == 0
    with pytest.raises(ValueError):
        class_metrics(matrix.astype(np.float32) + 0.5)
    acc = ConfusionAccumulator.create()
    acc.update(np.zeros((1, 1), dtype=np.int64), np.zeros((1, 1), dtype=np.int64))
    rows = epoch_rows(acc.matrix, epoch=1, split="train_online", branch="student")
    append_jsonl(tmp_path / "metrics.jsonl", rows)
    append_csv(tmp_path / "metrics.csv", rows)
    assert json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])["epoch"] == 1
    assert (tmp_path / "metrics.csv").read_text().splitlines()[0].startswith("epoch,split,branch")


def test_validation_counts_and_metrics_remain_pooled_with_separate_patient_macro(tmp_path):
    # A one-voxel patient is perfect; a nine-voxel patient is entirely missed.
    first = confusion_matrix(np.ones(1, dtype=int), np.ones(1, dtype=int))
    second = confusion_matrix(np.ones(9, dtype=int), np.zeros(9, dtype=int))
    pooled = first + second
    rows = _validation_rows({"pooled_confusion": pooled.tolist(), "per_class": {
        "RV": {"dice": 0.5, "f1": 0.5, "recall": 0.5, "precision": 1.0},
    }}, epoch=1, branch="student")
    row = rows[0]
    assert row["dice"] == row["f1"] == pytest.approx(2 / 11)
    assert row["recall"] == pytest.approx(0.1)
    assert row["TP"] == 1 and row["FN"] == 9
    assert row["aggregation"] == "pooled_voxels"
    assert row["macro_patient_dice"] == 0.5
    append_csv(tmp_path / "metrics.csv", rows)
    with (tmp_path / "metrics.csv").open() as stream:
        saved = next(csv.DictReader(stream))
    assert float(saved["dice"]) == pytest.approx(2 / 11)
    assert float(saved["macro_patient_dice"]) == 0.5
