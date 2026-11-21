"""SSL4MIS-compatible evaluator against hand-built volumes with known answers (CPU only)."""
from __future__ import annotations

import numpy as np
import pytest

from self_audit.candidate_pg.evaluation_ssl4mis_compat import HD95_UNIT, class_metrics, evaluate_volumes


def _vol(shape=(6, 16, 16)):
    return np.zeros(shape, dtype=np.int64)


def test_single_voxel_offset_dice_zero_hd95_equals_distance_in_voxels_and_mm() -> None:
    a, b = _vol().astype(bool), _vol().astype(bool)
    a[2, 5, 5], b[2, 5, 8] = True, True  # 3 voxels apart along W
    dice, hd95, asd, empty = class_metrics(a, b)
    assert (dice, hd95, asd, empty) == (0.0, 3.0, 3.0, False)
    a2, b2 = _vol().astype(bool), _vol().astype(bool)
    a2[1, 5, 5], b2[2, 5, 5] = True, True  # 1 slice apart
    assert class_metrics(a2, b2)[1] == 1.0  # voxel protocol ignores the 10 mm slice spacing
    assert class_metrics(a2, b2, voxelspacing=(10.0, 1.0, 1.0))[1] == 10.0  # mm variant


def test_identical_masks_and_overlap_dice() -> None:
    a = _vol().astype(bool)
    a[1:4, 4:8, 4:8] = True
    assert class_metrics(a, a)[:3] == (1.0, 0.0, 0.0)
    b = _vol().astype(bool)
    b[1:4, 4:8, 6:10] = True  # half of each 3x4x4 block overlaps: inter 24, sizes 48+48
    assert class_metrics(a, b)[0] == pytest.approx(2 * 24 / 96)


def test_empty_masks_follow_the_documented_zero_policy() -> None:
    a = _vol().astype(bool)
    a[1:3, 4:6, 4:6] = True
    assert class_metrics(a, _vol().astype(bool)) == (0.0, 0.0, 0.0, True)
    assert class_metrics(_vol().astype(bool), _vol().astype(bool)) == (0.0, 0.0, 0.0, True)


def test_volume_average_is_over_volumes_then_classes_and_counts_empties() -> None:
    gt = _vol()
    gt[1:4, 4:8, 4:8] = 1
    gt[1:4, 9:12, 9:12] = 2
    gt[1:4, 1:3, 1:3] = 3
    pred = gt.copy()
    pred[gt == 2] = 0  # MYO missed entirely -> empty prediction for class 2
    shifted = gt.copy()
    shifted[gt == 3] = 0
    shifted[1:4, 1:3, 3:5] = 3  # LV shifted by 2 voxels: dice 0, hd95 2
    out = evaluate_volumes([{"patient": "a", "frame": 0, "prediction": pred, "target": gt},
                            {"patient": "b", "frame": 1, "prediction": shifted, "target": gt}])
    assert out["volumes"] == 2 and out["hd95_unit"] == HD95_UNIT
    assert out["per_class"]["RV"]["dice"] == pytest.approx(1.0)
    assert out["per_class"]["MYO"]["dice"] == pytest.approx(0.5)  # volume a: missed (0), volume b: perfect (1)
    assert out["per_class"]["LV"]["hd95"] == pytest.approx(0.5 * 0.0 + 0.5 * 2.0)
    assert out["empty_cases"] == 1
    assert out["mean"]["dice"] == pytest.approx(np.mean([out["per_class"][c]["dice"] for c in ("RV", "MYO", "LV")]))


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="native grid"):
        evaluate_volumes([{"prediction": _vol((2, 4, 4)), "target": _vol((3, 4, 4))}])


def test_jaccard_in_volume_metrics_and_empty_case():
    from self_audit.candidate_pg.evaluation_ssl4mis_compat import jaccard

    gt = np.zeros((2, 6, 6), dtype=np.int64)
    gt[:, 1:4, 1:4] = 1
    pred = gt.copy()
    pred[:, 1:4, 3] = 0  # drop one column: 6 of 18 voxels
    out = evaluate_volumes([{"patient": "a", "frame": 0, "prediction": pred, "target": gt}])
    expected = 12 / 18
    assert out["per_class"]["RV"]["jaccard"] == pytest.approx(expected)
    assert out["per_class"]["RV"]["dice"] == pytest.approx(2 * 12 / (12 + 18))
    assert out["mean"]["jaccard"] == pytest.approx(expected / 3)  # MYO and LV are empty in both: 0
    assert jaccard(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 0.0


def test_epoch_full_metrics_table_replaces_replayed_epochs(tmp_path):
    from self_audit.candidate_pg.engine import _append_epoch_full

    unit = {"mean": {"dice": .5, "jaccard": .4, "hd95": 3., "asd": 1.}, "per_class": {c: {"dice": .5, "jaccard": .4, "hd95": 3., "asd": 1.} for c in ("RV", "MYO", "LV")}}
    entry = {"native_mm": unit, "ssl4mis_voxels": unit}
    for epoch in (1, 2, 2, 3):  # epoch 2 replayed after a resume
        _append_epoch_full(tmp_path, {"epoch": epoch, "update": epoch * 80, "val": entry, "test": entry})
    import json
    rows = [json.loads(l) for l in (tmp_path / "epoch_full_metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2, 3]
    csv_lines = (tmp_path / "epoch_full_metrics.csv").read_text().splitlines()
    assert csv_lines[0] == "epoch,update,split,units,class,dice,jaccard,hd95,asd" and len(csv_lines) == 1 + 3 * 2 * 2 * 4
