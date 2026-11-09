"""Shared segmentation metrics with explicit absent-class semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Iterable, Mapping

import numpy as np


CLASS_NAMES = ("BG", "RV", "MYO", "LV")
VALID_LABELS = (0, 1, 2, 3, 255)


def confusion_matrix(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return ``[class, predicted]`` counts, excluding ignore=255 reference."""

    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction shapes must match")
    if not np.isin(reference, VALID_LABELS).all() or not np.isin(prediction, VALID_LABELS).all():
        raise ValueError("segmentation arrays contain an unsupported class ID")
    valid = reference != 255
    pred = np.where(prediction[valid] == 255, 4, prediction[valid])
    return np.bincount(
        reference[valid].astype(np.int64) * 5 + pred.astype(np.int64),
        minlength=20,
    ).reshape(4, 5)


def class_metrics(matrix: np.ndarray, class_names: Iterable[str] = CLASS_NAMES) -> list[dict]:
    """Compute Dice/F1, precision and recall per class.

    Absent reference classes remain ``None`` for recall and Dice/F1; a false
    prediction for an absent class has precision 0. This prevents an empty
    class from silently inflating a macro score.
    """

    matrix = np.asarray(matrix)
    if (matrix.shape != (4, 5) or not np.issubdtype(matrix.dtype, np.integer)
            or np.any(matrix < 0)):
        raise ValueError("confusion matrix must be finite nonnegative [4,5]")
    names = tuple(class_names)
    if len(names) != 4:
        raise ValueError("exactly four class names are required")
    rows = []
    for index, name in enumerate(names):
        tp = int(matrix[index, index])
        reference_count = int(matrix[index].sum())
        prediction_count = int(matrix[:, index].sum())
        fp = prediction_count - tp
        fn = reference_count - tp
        precision = tp / prediction_count if prediction_count else None
        recall = tp / reference_count if reference_count else None
        # A class absent from the reference is not a positive Dice/F1 case,
        # even when the model emits false positives. Keep precision=0 for that
        # diagnostic and leave recall/Dice/F1 undefined.
        dice = (2 * tp / (reference_count + prediction_count)
                if reference_count and reference_count + prediction_count else None)
        rows.append({
            "class": str(name), "TP": tp, "FP": fp, "FN": fn,
            "reference_present": bool(reference_count),
            "reference_count": reference_count,
            "support": reference_count,
            "prediction_count": prediction_count,
            "status": "ok" if reference_count else ("false_positive_only" if prediction_count else "absent_reference"),
            "dice": dice,
            "f1": dice,
            "precision": precision,
            "recall": recall,
        })
    return rows


def epoch_rows(matrix: np.ndarray, *, epoch: int, split: str, branch: str,
               loss: float | None = None) -> list[dict]:
    """Flatten one epoch/branch table into JSONL/CSV-ready rows."""

    rows = []
    for row in class_metrics(matrix):
        item = {"epoch": int(epoch), "split": str(split), "branch": str(branch),
                "aggregation": "pooled_voxels", **row}
        if loss is not None:
            item["loss"] = float(loss)
        rows.append(item)
    return rows


def append_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")


def append_csv(path: str | Path, rows: Iterable[Mapping]) -> None:
    records = [dict(row) for row in rows]
    if not records:
        return
    fields = ["epoch", "split", "branch", "aggregation", "class", "TP", "FP", "FN", "reference_count",
              "support", "prediction_count", "reference_present", "status", "dice", "f1",
              "precision", "recall", "macro_patient_dice", "macro_patient_f1",
              "macro_patient_precision", "macro_patient_recall", "loss"]
    destination = Path(path)
    exists = destination.exists() and destination.stat().st_size > 0
    with destination.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(records)


@dataclass
class ConfusionAccumulator:
    """Small deterministic accumulator for epoch-level metric logging."""

    matrix: np.ndarray

    @classmethod
    def create(cls) -> "ConfusionAccumulator":
        return cls(np.zeros((4, 5), dtype=np.int64))

    def update(self, reference: np.ndarray, prediction: np.ndarray) -> None:
        self.matrix += confusion_matrix(reference, prediction)

    def result(self) -> list[dict]:
        return class_metrics(self.matrix)


__all__ = ["CLASS_NAMES", "VALID_LABELS", "confusion_matrix", "class_metrics",
           "ConfusionAccumulator", "epoch_rows", "append_jsonl", "append_csv"]

