from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from self_audit.candidate_pg.evaluation import (
    METRIC_COLUMNS,
    evaluate_records,
    pooled_grid_metrics,
    select_best_epoch,
    write_epoch_metrics,
)
from self_audit.candidate_pg.analysis import (
    _shuffle_codes,
    code_spatial_shuffle_diagnostic,
    compare_logged_export,
    paired_comparisons,
)
from scripts.pg_report import build_report


def _mask(offset: int = 0) -> np.ndarray:
    value = np.zeros((4, 4, 4), dtype=np.uint8)
    value[1 + offset:3 + offset, 1:3, 1:3] = 3
    return value


def test_native_patient_macro_and_surface_spacing() -> None:
    target = _mask()
    result = evaluate_records([
        {"patient": "p1", "frame": 0, "target": target, "prediction": target,
         "geometry": {"spacing_zyx": [2.0, 1.0, 1.0]}, "split": "VAL18"},
        {"patient": "p1", "frame": 1, "target": target, "prediction": target,
         "geometry": {"spacing_zyx": [2.0, 1.0, 1.0]}, "split": "VAL18"},
    ], split="VAL18", epoch=1)
    assert result["patient_count"] == 1
    assert result["fg_dice"] == 1.0
    assert result["per_class"]["LV"]["hd95_mm"] == 0.0


def test_qt_grid_uses_cell_spacing_and_masks_internal_padding() -> None:
    target = _mask()
    prediction = _mask(1)
    metrics = pooled_grid_metrics(target, prediction, [2.0, 1.0, 1.0], (2, 1, 1),
                                  valid_mask=np.ones_like(target, dtype=bool))
    assert metrics["per_class"]["LV"]["hd95_mm"] >= 2.0
    padded = np.zeros_like(target, dtype=bool)
    padded[1:, :, :] = True
    masked = pooled_grid_metrics(target, prediction, [2.0, 1.0, 1.0], (2, 1, 1),
                                 valid_mask=padded)
    assert masked["per_class"]["LV"]["dice"] is not None


def test_selection_is_val_student_only_and_epoch_schema(tmp_path) -> None:
    rows = [
        {"epoch": 1, "split": "VAL18", "path": "student_only", "fg_dice": 0.8},
        {"epoch": 2, "split": "VAL18", "path": "student_only", "fg_dice": 0.8},
        {"epoch": 1, "split": "TEST50", "path": "student_only", "fg_dice": 0.99},
    ]
    assert select_best_epoch(rows)["epoch"] == 1
    result = evaluate_records([{"patient": "p", "target": _mask(), "prediction": _mask(),
                               "geometry": {"spacing_zyx": [1, 1, 1]}, "split": "VAL18"}],
                              split="VAL18", epoch=1, path="student_only")
    paths = write_epoch_metrics(tmp_path, [result], best_epoch=1)
    with open(paths["csv"], newline="", encoding="utf-8") as stream:
        assert tuple(csv.reader(stream).__next__()) == METRIC_COLUMNS


def _patient(value: float) -> dict[str, object]:
    return {"fg_dice": value, "per_class": {
        name: {"dice": value, "hd95_mm": 1.0, "assd_mm": 0.5}
        for name in ("RV", "MYO", "LV")}}


def test_paired_comparisons_raise_on_mismatched_patients() -> None:
    arms = {arm: {"per_patient": {"p1": _patient(0.5)}} for arm in ("A1", "A2", "A3", "A4")}
    arms["A2"]["per_patient"]["p2"] = _patient(0.6)
    with pytest.raises(ValueError, match="patient ID sets differ"):
        paired_comparisons(arms, bootstrap_resamples=2)


def test_logged_export_missing_required_metric_fails() -> None:
    row = {"fg_dice": 0.5, "per_class": {
        name: {"dice": 0.5, "hd95_mm": 1.0, "assd_mm": 0.5}
        for name in ("RV", "MYO", "LV")}}
    del row["per_class"]["LV"]["assd_mm"]
    result = compare_logged_export(row, row)
    assert result["status"] == "FAIL"
    assert result["fields"]["lv_assd"]["status"] == "FAIL_MISSING_METRIC"


def test_qs_shuffle_only_valid_cells_and_uses_hsq_head() -> None:
    class Head:
        def __call__(self, values):
            return torch.as_tensor(values, dtype=torch.float32).unsqueeze(1)

    class Model:
        H_S_Q = Head()

    codes = np.arange(8, dtype=np.int64).reshape(1, 2, 2, 2)
    valid = np.array([[[[True, True], [True, True]], [[False, False], [False, False]]]])
    result = code_spatial_shuffle_diagnostic(Model(), codes, valid_mask=valid)
    assert result["status"] == "PASS"
    # Invalid/padded z=1 cells must remain bitwise unchanged.
    assert result["changed_positions"] <= 4
    with pytest.raises(ValueError, match="valid_mask"):
        code_spatial_shuffle_diagnostic(Model(), codes)


def test_qs_shuffle_preserves_whole_code_vectors_and_padding() -> None:
    first = np.arange(8).reshape(1, 1, 2, 2, 2)
    codes = np.concatenate((first, first + 100, first + 200), axis=1)
    valid = np.ones_like(codes, dtype=bool)
    valid[..., 1, :, :] = False
    shuffled = _shuffle_codes(codes, np.random.default_rng(42), valid)
    assert np.any(shuffled != codes)
    assert np.array_equal(shuffled[~valid], codes[~valid])
    assert np.array_equal(shuffled[:, 1] - shuffled[:, 0], np.full((1, 2, 2, 2), 100))
    assert np.array_equal(shuffled[:, 2] - shuffled[:, 0], np.full((1, 2, 2, 2), 200))
    assert sorted(shuffled[:, 0].ravel()) == sorted(codes[:, 0].ravel())


def test_report_consumes_canonical_logger_outputs_end_to_end(tmp_path) -> None:
    run_dir = tmp_path / "run"
    result = evaluate_records([{"patient": "p", "target": _mask(), "prediction": _mask(),
                               "geometry": {"spacing_zyx": [1, 1, 1]}, "split": "VAL18"}],
                              split="VAL18", epoch=1, path="student_only")
    write_epoch_metrics(run_dir, [result], best_epoch=1)
    report = build_report(metrics_dir=run_dir, out_dir=tmp_path / "report")
    assert report["rows"] == 1
    assert report["patient_rows"] == 1
    assert report["selection"]["status"] == "SELECTED"
    absent = build_report(metrics_dir=tmp_path / "absent", out_dir=tmp_path / "absent-report")
    assert absent["evidence_status"] == "NOT_RUN"
    assert absent["selection"]["status"].startswith("NOT_RUN")
