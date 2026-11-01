"""Constructed numerical checks; these establish no patient efficacy."""

import json
from copy import deepcopy

import numpy as np
import pytest

from self_audit.context_metrics import (
    ContextMetricsAccumulator,
    aggregate_records,
    analyze_frame,
    frame_csv_rows,
    frame_metric_csv_rows,
    patient_class_common_failure_csv_rows,
    summary_csv_rows,
    validate_frame_record,
)


def predictions(baseline, final=None, half=None):
    final = baseline if final is None else final
    half = final if half is None else half
    return {0: baseline, 0.5: half, 1: final}


def volume(values):
    return np.asarray(values, dtype=np.uint8).reshape(1, 1, -1)


def class_transition(record, name="RV", lam="1", stratum="all"):
    return record["strata"][stratum]["comparisons"][lam]["foreground"][name]


def test_four_cells_exclude_already_correct_student_and_separate_regression_opportunity():
    gt = volume([1, 1, 1, 1, 1, 1])
    t0, s0 = volume([0, 0, 0, 0, 0, 0]), volume([0, 0, 0, 0, 1, 1])
    t1, s1 = volume([1, 0, 1, 0, 1, 1]), volume([0, 1, 1, 0, 1, 0])
    record = analyze_frame("p", "ED", gt, predictions(t0, t1, t0), predictions(s0, s1, s0))
    result = class_transition(record)
    counts, rates = result["counts"], result["rates"]
    assert counts["gt_count"] == 6
    assert counts["common_failure"] == 4
    for cell in ("teacher_only_fix", "student_only_fix", "both_fix", "neither_fix"):
        assert counts[cell] == 1
        assert rates[f"{cell}_per_common_failure"] == 0.25
        assert rates[f"{cell}_per_gt"] == pytest.approx(1 / 6)
    # One primary teacher-only fix plus one newly wrong student voxel.
    assert counts["supplemental_teacher_rescue_student_wrong"] == 2
    assert counts["supplemental_common_failure_teacher_only"] == 1
    assert counts["supplemental_student_regression"] == 1
    assert rates["supplemental_teacher_rescue_student_wrong_per_teacher_baseline_wrong"] == pytest.approx(2 / 6)
    assert counts["teacher_rescue"] == 4
    assert counts["student_harm"] == 1
    assert class_transition(record, lam="0.5")["counts"]["neither_fix"] == 4


def test_true_class_harm_false_positives_and_regression_confusion_include_background():
    gt = volume([1, 1, 0, 2, 3, 255])
    before = volume([1, 0, 0, 2, 0, 255])
    after = volume([2, 1, 1, 0, 3, 255])
    record = analyze_frame("p", 0, gt, predictions(before, after), predictions(before, after))
    counts = class_transition(record)["counts"]
    assert counts["teacher_rescue"] == 1
    assert counts["teacher_harm"] == 1
    assert counts["teacher_fp_introduced"] == 1
    assert counts["teacher_baseline_fp"] == 0
    target_rv = record["strata"]["all"]["predictions"]["teacher"]["1"]["classes"]["RV"]
    assert (target_rv["tp"], target_rv["fp"], target_rv["fn"]) == (1, 1, 1)
    all_pixels = record["strata"]["all"]["comparisons"]["1"]["all_pixels"]["teacher"]
    assert all_pixels["counts"]["rescued"] == 2
    assert all_pixels["counts"]["harmed"] == 3
    assert all_pixels["counts"]["net_correct_change"] == -1
    assert all_pixels["regression_confusion"][1][2] == 1
    assert all_pixels["regression_confusion"][0][1] == 1
    assert all_pixels["regression_confusion"][2][0] == 1
    assert sum(map(sum, all_pixels["prediction_transition_confusion"])) == 5


def test_absent_class_dice_is_undefined_despite_false_positives():
    gt, before, after = volume([0, 0, 0]), volume([0, 0, 1]), volume([1, 0, 0])
    record = analyze_frame("p", 0, gt, predictions(before, after), predictions(before))
    rv = record["strata"]["all"]["predictions"]["teacher"]["1"]["classes"]["RV"]
    assert rv["reference_present"] is False
    assert rv["dice"] is None
    assert rv["fp"] == 1
    assert rv["absent_fp_rate"] == pytest.approx(1 / 3)
    transition = class_transition(record)
    assert transition["counts"]["teacher_fp_introduced"] == 1
    assert transition["counts"]["teacher_fp_removed"] == 1
    assert transition["rates"]["teacher_only_fix_per_gt"] is None
    assert transition["rates"]["teacher_only_fix_per_common_failure"] is None
    summary = aggregate_records([record], leave_one_out=False)
    macro = summary["patient_macro"]["all"]["predictions"]["teacher"]["1"]
    assert macro["classes"]["RV"]["dice"] is None
    assert macro["classes"]["RV"]["absent_fp_rate"] == pytest.approx(1 / 3)
    assert macro["defined_classes_foreground"] == 0
    assert macro["class_first_foreground_macro"] is None


@pytest.mark.parametrize("slices,edge_count,interior_count", [(1, 1, 0), (2, 2, 0), (3, 2, 1), (5, 2, 3)])
def test_axial_edge_union_and_interior_are_disjoint(slices, edge_count, interior_count):
    gt = np.ones((slices, 2, 3), dtype=np.uint8)
    before = np.zeros_like(gt)
    record = analyze_frame("p", 0, gt, predictions(before, gt), predictions(before))
    assert record["strata"]["edge"]["valid_voxels"] == edge_count * 6
    assert record["strata"]["interior"]["valid_voxels"] == interior_count * 6
    assert class_transition(record, stratum="edge")["counts"]["teacher_only_fix"] == edge_count * 6
    assert class_transition(record, stratum="interior")["counts"]["teacher_only_fix"] == interior_count * 6
    assert set(record["strata"]["edge"]["slice_indices"]).isdisjoint(record["strata"]["interior"]["slice_indices"])
    if not interior_count:
        assert class_transition(record, stratum="interior")["rates"]["teacher_only_fix_per_gt"] is None


def test_ignore_and_valid_mask_apply_to_every_denominator_without_changing_inputs():
    gt = volume([1, 1, 255, 1])
    before, after = volume([0, 255, 255, 0]), volume([1, 255, 255, 0])
    valid = np.asarray([True, False, True, True]).reshape(gt.shape)
    original_gt, original_before, original_valid = gt.copy(), before.copy(), valid.copy()
    record = analyze_frame("p", 0, gt, predictions(before, after), predictions(before), valid)
    transition = class_transition(record)
    assert transition["counts"]["valid_voxels"] == 2
    assert transition["counts"]["gt_count"] == 2
    assert transition["counts"]["common_failure"] == 2
    assert transition["rates"]["teacher_only_fix_per_common_failure"] == 0.5
    assert record["strata"]["all"]["predictions"]["teacher"]["1"]["classes"]["RV"]["dice"] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(gt, original_gt)
    np.testing.assert_array_equal(before, original_before)
    np.testing.assert_array_equal(valid, original_valid)


def test_entirely_ignored_frame_has_no_fabricated_dice_or_absent_fp_rate():
    gt = np.full((1, 2, 3), 255, dtype=np.uint8)
    record = analyze_frame("p", 0, gt, predictions(gt), predictions(gt))
    assert record["valid_voxels"] == 0
    summary = aggregate_records([record])
    assert summary["patient_macro"]["all"]["predictions"]["teacher"]["0"]["class_first_macro4"] is None
    assert summary["patients"]["p"]["strata"]["all"]["predictions"]["teacher"]["0"]["classes"]["RV"]["absent_fp_rate"] is None
    assert summary["leave_one_patient_out"]["p"]["patient_count"] == 0
    json.dumps(summary, allow_nan=False)


def test_dice_pools_slices_then_averages_frames_then_patients():
    # First frame: 1/1 correct on one slice and 0/9 on the other. Full-frame
    # Dice is 2/11, rather than the per-slice mean 1/2.
    gt = np.full((2, 1, 9), 255, dtype=np.uint8)
    gt[0, 0, 0] = 1
    gt[1] = 1
    pred = np.zeros_like(gt)
    pred[0, 0, 0] = 1
    a0 = analyze_frame("a", 0, gt, predictions(pred), predictions(pred))
    a1 = analyze_frame("a", 1, volume([1]), predictions(volume([1])), predictions(volume([1])))
    b0 = analyze_frame("b", 0, volume([1]), predictions(volume([0])), predictions(volume([0])))
    assert a0["strata"]["all"]["predictions"]["teacher"]["0"]["classes"]["RV"]["dice"] == pytest.approx(2 / 11)
    summary = aggregate_records([a0, a1, b0])
    patient_a = summary["patients"]["a"]["strata"]["all"]["predictions"]["teacher"]["0"]["classes"]["RV"]
    assert patient_a["dice"] == pytest.approx(13 / 22)
    macro = summary["patient_macro"]["all"]["predictions"]["teacher"]["0"]
    assert macro["classes"]["RV"]["dice"] == pytest.approx(13 / 44)
    assert macro["class_first_foreground_macro"] == pytest.approx(13 / 44)
    assert summary["pooled"]["all"]["predictions"]["teacher"]["0"]["classes"]["RV"]["pooled_voxel_dice"] == pytest.approx(2 / 7)
    assert summary["leave_one_patient_out"]["a"]["frame_count"] == 1
    assert summary["leave_one_patient_out"]["a"]["patient_macro"]["all"]["predictions"]["teacher"]["0"]["classes"]["RV"]["dice"] == 0
    assert summary["leave_one_patient_out"]["b"]["patient_macro"]["all"]["predictions"]["teacher"]["0"]["classes"]["RV"]["dice"] == pytest.approx(13 / 22)


def test_class_first_macro_differs_from_patient_first_when_classes_are_absent():
    a = analyze_frame("a", "ED", volume([1, 2]), predictions(volume([1, 0])), predictions(volume([1, 0])))
    b = analyze_frame("b", "ED", volume([1]), predictions(volume([1])), predictions(volume([1])))
    summary = aggregate_records([a, b], leave_one_out=False)
    metrics = summary["patient_macro"]["all"]["predictions"]["teacher"]["0"]
    assert metrics["class_first_foreground_macro"] == 0.5
    assert metrics["patient_first_foreground_macro"] == 0.75
    assert metrics["defined_classes_foreground"] == 2


def test_transition_patient_macro_and_pooled_rates_have_distinct_weighting():
    gt_small, gt_large = volume([1]), volume([1] * 9)
    wrong_small, wrong_large = np.zeros_like(gt_small), np.zeros_like(gt_large)
    small = analyze_frame("small", 0, gt_small, predictions(wrong_small, gt_small), predictions(wrong_small))
    large = analyze_frame("large", 0, gt_large, predictions(wrong_large), predictions(wrong_large))
    summary = aggregate_records([small, large], leave_one_out=False)
    macro = summary["patient_macro"]["all"]["comparisons"]["1"]["foreground"]["RV"]["rates"]
    pooled = summary["pooled"]["all"]["comparisons"]["1"]["foreground"]["RV"]["rates"]
    assert macro["teacher_only_fix_per_common_failure"] == {"mean": 0.5, "defined_patients": 2}
    assert pooled["teacher_only_fix_per_common_failure"] == 0.1


def test_patient_class_common_failure_table_keeps_edge_interior_rescue_harm_and_fp():
    gt = np.tile(np.asarray([[[1, 0]]], dtype=np.uint8), (3, 1, 1))
    baseline = np.zeros_like(gt)
    teacher_target = gt.copy()
    teacher_target[:, 0, 1] = 1  # One new FP per slice, reported separately from rescue.
    student_target = baseline.copy()
    record = analyze_frame(
        "patient-a", "ED", gt,
        predictions(baseline, teacher_target, baseline),
        predictions(baseline, student_target, baseline),
    )
    summary = aggregate_records([record], leave_one_out=False)
    rows = patient_class_common_failure_csv_rows(summary)
    edge = next(row for row in rows if row["stratum"] == "edge" and row["lambda"] == 1.0 and row["class_name"] == "RV")
    interior = next(row for row in rows if row["stratum"] == "interior" and row["lambda"] == 1.0 and row["class_name"] == "RV")
    assert edge["patient_id"] == "patient-a"
    assert (edge["valid_voxels"], edge["gt_count"], edge["common_failure"]) == (4, 2, 2)
    assert (edge["teacher_only_fix"], edge["student_only_fix"], edge["both_fix"], edge["neither_fix"]) == (2, 0, 0, 0)
    assert edge["teacher_rescue"] == 2 and edge["teacher_harm"] == 0
    assert edge["teacher_fp_introduced"] == 2 and edge["teacher_fp_removed"] == 0
    assert edge["teacher_only_fix_per_common_failure"] == 1.0
    assert (interior["valid_voxels"], interior["gt_count"], interior["common_failure"]) == (2, 1, 1)
    assert interior["teacher_only_fix"] == 1 and interior["teacher_fp_introduced"] == 1


def test_class_absent_frames_are_omitted_from_patient_dice_but_retained_for_fp():
    present = analyze_frame("p", 0, volume([1]), predictions(volume([1])), predictions(volume([1])))
    absent = analyze_frame("p", 1, volume([0]), predictions(volume([1])), predictions(volume([1])))
    summary = aggregate_records([present, absent], leave_one_out=False)
    metrics = summary["patients"]["p"]["strata"]["all"]["predictions"]["teacher"]["0"]["classes"]["RV"]
    assert metrics["dice"] == 1.0
    assert metrics["pooled_voxel_dice"] == pytest.approx(2 / 3)
    assert metrics["defined_frames"] == 1
    assert metrics["absent_frames"] == 1
    assert metrics["absent_frame_fp"] == 1
    assert metrics["absent_fp_rate"] == 1.0


def test_identity_is_patient_plus_frame_and_accumulator_isolated_from_returned_records():
    acc = ContextMetricsAccumulator()
    first = acc.update_frame("a", "ED", volume([1]), predictions(volume([1])), predictions(volume([1])))
    first["strata"]["all"]["valid_voxels"] = 999
    copied = acc.records
    copied.clear()
    acc.update_frame("b", "ED", volume([1]), predictions(volume([0])), predictions(volume([0])))
    with pytest.raises(ValueError, match="Duplicate"):
        acc.update_frame("a", "ED", volume([1]), predictions(volume([1])), predictions(volume([1])))
    summary = acc.finalize()
    assert summary["patient_count"] == 2
    assert summary["patients"]["a"]["strata"]["all"]["valid_voxels"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_records([acc.records[0], acc.records[0]])


def test_integrity_checks_reject_tampered_partition_and_confusion():
    gt = np.ones((3, 1, 1), dtype=np.uint8)
    record = analyze_frame("p", 0, gt, predictions(np.zeros_like(gt), gt), predictions(np.zeros_like(gt)))
    bad = deepcopy(record)
    class_transition(bad)["counts"]["both_fix"] += 1
    with pytest.raises(ValueError, match="partition"):
        aggregate_records([bad])
    bad = deepcopy(record)
    class_transition(bad)["counts"]["supplemental_student_regression"] += 1
    with pytest.raises(ValueError, match="Supplemental"):
        validate_frame_record(bad)
    bad = deepcopy(record)
    bad["strata"]["all"]["predictions"]["teacher"]["1"]["confusion"][0][0] += 1
    with pytest.raises(ValueError, match="confusion"):
        validate_frame_record(bad)
    bad = deepcopy(record)
    bad["strata"]["all"]["predictions"]["teacher"]["1"]["classes"]["RV"]["dice"] = 0.123
    with pytest.raises(ValueError, match="Class statistics"):
        validate_frame_record(bad)
    bad = deepcopy(record)
    bad["strata"]["edge"]["slice_indices"] = [0, 1]
    with pytest.raises(ValueError, match="axial endpoint"):
        validate_frame_record(bad)


def test_input_contract_rejects_missing_lambdas_invalid_predictions_and_invalid_masks():
    gt, good = volume([1, 1]), volume([0, 0])
    with pytest.raises(ValueError, match="all frozen predictions"):
        analyze_frame("p", 0, gt, {0: good, 1: good}, predictions(good))
    with pytest.raises(ValueError, match="uniquely"):
        analyze_frame("p", 0, gt, {0: good, "0": good, 0.5: good, 1: good}, predictions(good))
    with pytest.raises(ValueError, match="uniquely"):
        analyze_frame("p", 0, gt, {0: good, 0.50000000001: good, 1: good}, predictions(good))
    with pytest.raises(ValueError, match="invalid label"):
        analyze_frame("p", 0, gt, predictions(volume([0, 255])), predictions(good))
    with pytest.raises(ValueError, match="integer"):
        analyze_frame("p", 0, gt, predictions(good.astype(np.float32)), predictions(good))
    with pytest.raises(ValueError, match="valid_mask must be boolean"):
        analyze_frame("p", 0, gt, predictions(good), predictions(good), np.ones_like(gt))
    with pytest.raises(ValueError, match="labels other"):
        analyze_frame("p", 0, volume([1, 4]), predictions(good), predictions(good))


def test_json_roundtrip_determinism_and_csv_exports_are_scalar_and_complete():
    a = analyze_frame("a", 0, volume([0, 1, 2, 3]), predictions(volume([0, 0, 2, 3])), predictions(volume([0, 1, 0, 3])))
    b = analyze_frame("b", 0, volume([0, 1, 2, 3]), predictions(volume([0, 1, 2, 3])), predictions(volume([0, 1, 2, 3])))
    restored = json.loads(json.dumps([a, b], allow_nan=False))
    summary = aggregate_records(restored)
    assert summary == aggregate_records([b, a])
    json.dumps(summary, allow_nan=False)
    transitions = frame_csv_rows(restored)
    assert len(transitions) == 2 * 3 * 2 * 3
    assert "supplemental_student_regression" in transitions[0]
    assert "teacher_baseline_fp" in transitions[0]
    metric_rows = frame_metric_csv_rows(restored)
    assert len(metric_rows) == 2 * 3 * 2 * 3 * 4
    assert {row["class_name"] for row in metric_rows} == {"BG", "RV", "MYO", "LV"}
    summary_rows = summary_csv_rows(summary)
    assert {row["scope"] for row in summary_rows} == {"patient", "patient_macro", "pooled", "leave_one_patient_out"}
    assert any("regression_confusion" in row["metric_path"] for row in summary_rows)
    for row in transitions + metric_rows + summary_rows:
        assert all(value is None or isinstance(value, (str, int, float, bool)) for value in row.values())


def test_empty_cohort_has_defined_empty_outputs_and_no_nan():
    summary = aggregate_records([])
    assert summary["patient_count"] == summary["frame_count"] == 0
    assert summary["patients"] == summary["leave_one_patient_out"] == {}
    assert summary["patient_macro"]["all"]["predictions"]["student"]["1"]["class_first_macro4"] is None
    assert frame_csv_rows([]) == frame_metric_csv_rows([]) == []
    json.dumps(summary, allow_nan=False)
