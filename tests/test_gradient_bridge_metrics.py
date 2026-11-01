"""CPU fixtures for frozen gradient-bridge scoring and paired gates."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from self_audit.gradient_bridge_metrics import (
    ARMS,
    GATE_SCHEMA,
    GradientBridgeFrameAccumulator,
    MetricsIntegrityError,
    NonFiniteMetricsError,
    aggregate_patient_rows,
    evaluate_paired_gate,
    paired_patient_bootstrap,
    score_frame_logits,
    summarize_patient_rows,
)
from self_audit.losses import supervised_loss


CLASS_NAMES = ("BG", "RV", "MYO", "LV")


def test_invalid_frame_dice_rejected_before_patient_aggregation():
    truth = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
    logits = _logits_for_prediction(truth)
    frame = score_frame_logits("fixture-patient", "0", truth, {arm: logits for arm in ARMS})
    frame["arms"]["D2"]["class_dice"]["RV"] = 1.2
    frame["arms"]["D2"]["class_metrics"]["RV"]["dice"] = 1.2
    with pytest.raises(MetricsIntegrityError, match=r"\[0,1\]"):
        aggregate_patient_rows([frame])


def _gate_config(*, minimum=0.1, n_resamples=300, seed=17):
    return {
        "schema": GATE_SCHEMA,
        "minimum_fg_absolute_delta": minimum,
        "bootstrap": {"n_resamples": n_resamples, "seed": seed},
        "confidence_rule": {
            "method": "paired_percentile",
            "level": 0.95,
            "minimum_defined_fraction": 1.0,
            "acceptance": "lower_bound_strictly_above_minimum",
        },
        "loo_rule": {"method": "all_deltas_at_least_minimum", "require_all_defined": True},
    }


def _row(patient_id, d0, d1=None, d2=None, *, support=None, fallback=False):
    d1 = d0 if d1 is None else d1
    d2 = d0 if d2 is None else d2
    support = {name: True for name in CLASS_NAMES} if support is None else support
    arm_scores = {"D0": d0, "D1": d1, "D2": d2, "D3": d0, "D4": d1, "D5": d2}
    arms = {}
    for arm in ARMS:
        arms[arm] = {
            "class_dice": {
                class_name: (arm_scores[arm] if support[class_name] else None)
                for class_name in CLASS_NAMES
            },
            "fallback_any": bool(fallback and arm == "D2"),
        }
    return {"patient_id": patient_id, "arms": arms}


def _logits_for_prediction(prediction, *, amplitude=8.0):
    logits = np.zeros((4, *prediction.shape), dtype=np.float32)
    for class_id in range(4):
        logits[class_id] = np.where(prediction == class_id, amplitude, 0.0)
    return logits


def _manual_frame(patient_id, frame_id, dice_by_class, *, valid_voxels):
    class_metrics = {}
    for class_name, dice in dice_by_class.items():
        present = dice is not None
        tp = 1 if present and dice > 0.0 else 0
        reference_count = 1 if present else 0
        prediction_count = tp
        class_metrics[class_name] = {
            "tp": tp,
            "fp": 0,
            "fn": reference_count - tp,
            "reference_count": reference_count,
            "prediction_count": prediction_count,
            "reference_present": present,
            "dice": None,
        }
    # Unmatched predictions are assigned to BG so this compact fixture still
    # has a valid four-class count partition.
    remaining_predictions = valid_voxels - sum(
        counts["prediction_count"] for counts in class_metrics.values()
    )
    class_metrics["BG"]["fp"] += remaining_predictions
    class_metrics["BG"]["prediction_count"] += remaining_predictions
    for class_name, counts in class_metrics.items():
        if counts["reference_present"]:
            counts["dice"] = 2 * counts["tp"] / (counts["reference_count"] + counts["prediction_count"])
    dice_values = {name: counts["dice"] for name, counts in class_metrics.items()}
    return {
        "schema": "self_audit.gradient_bridge_frame.v1",
        "patient_id": patient_id,
        "frame_id": str(frame_id),
        "valid_voxels": valid_voxels,
        "arms": {
            arm: {
                "class_dice": dict(dice_values),
                "class_metrics": {name: dict(value) for name, value in class_metrics.items()},
                "native_supervised": {"ce": 0.0, "dice_loss": 0.0, "total": 0.0},
            }
            for arm in ARMS
        },
    }


def test_score_frozen_logits_reports_known_counts_ignore_and_native_losses():
    gt = np.asarray([[[0, 1, 1, 2, 3, 255]]], dtype=np.uint8)
    prediction = np.asarray([[[0, 0, 1, 2, 0, 3]]], dtype=np.uint8)
    logits = _logits_for_prediction(prediction)
    record = score_frame_logits(
        "p1",
        "ED",
        gt,
        {arm: logits.copy() for arm in ARMS},
    )

    assert record["valid_voxels"] == 5
    assert record["ignored_or_invalid_voxels"] == 1
    d0 = record["arms"]["D0"]
    assert d0["class_metrics"]["RV"]["tp"] == 1
    assert d0["class_metrics"]["RV"]["fp"] == 0
    assert d0["class_metrics"]["RV"]["fn"] == 1
    assert d0["class_dice"]["RV"] == pytest.approx(2 / 3)
    assert d0["rv_recall"] == pytest.approx(0.5)
    assert d0["rv_precision"] == pytest.approx(1.0)
    assert d0["class_dice"]["LV"] == 0.0
    native = d0["native_supervised"]
    assert native["support_voxels"] == 5
    assert native["present_foreground_classes"] == ["RV", "MYO", "LV"]
    assert native["ce"] > 0
    assert native["dice_loss"] > 0
    assert native["total"] == pytest.approx(native["ce"] + native["dice_loss"])


def test_streamed_chunks_match_canonical_supervised_loss_and_full_frame_wrapper():
    rng = np.random.default_rng(20261101)
    gt = np.asarray(
        [
            [[0, 1], [255, 1]],
            [[0, 2], [2, 0]],
            [[0, 0], [1, 255]],
        ],
        dtype=np.uint8,
    )
    valid = np.ones(gt.shape, dtype=bool)
    valid[1, 1, 1] = False
    logits = {
        arm: rng.normal(size=(4, *gt.shape)).astype(np.float32)
        for arm in ARMS
    }
    fallbacks = {arm: arm == "D2" for arm in ARMS}

    accumulator = GradientBridgeFrameAccumulator(
        "p-stream",
        "ED",
        gt.shape,
        fallback_by_arm=fallbacks,
    )
    accumulator.update_chunk(0, gt[:1], {arm: values[:, :1] for arm, values in logits.items()}, valid[:1])
    accumulator.update_chunk(1, gt[1:], {arm: values[:, 1:] for arm, values in logits.items()}, valid[1:])
    streamed = accumulator.finalize()
    convenience = score_frame_logits(
        "p-stream", "ED", gt, logits, valid, fallback_by_arm=fallbacks
    )

    support = valid & (gt != 255)
    assert streamed["valid_voxels"] == int(support.sum())
    assert streamed["ignored_or_invalid_voxels"] == gt.size - int(support.sum())
    for arm in ARMS:
        streamed_arm = streamed["arms"][arm]
        convenience_arm = convenience["arms"][arm]
        assert streamed_arm["class_metrics"] == convenience_arm["class_metrics"]
        assert streamed_arm["native_supervised"]["present_foreground_classes"] == ["RV", "MYO"]
        assert streamed_arm["native_supervised"]["absent_foreground_classes"] == ["LV"]
        assert streamed_arm["native_supervised"]["support_voxels"] == int(support.sum())

        labels_np = np.where(valid, gt, 255).astype(np.int64)
        logits_zchw = torch.from_numpy(np.ascontiguousarray(logits[arm].transpose(1, 0, 2, 3)))
        labels_tensor = torch.from_numpy(labels_np.astype(np.int64))
        canonical_total, canonical_components = supervised_loss(
            logits_zchw, labels_tensor, return_components=True
        )
        assert streamed_arm["native_supervised"]["ce"] == pytest.approx(canonical_components["ce"], abs=1e-6)
        assert streamed_arm["native_supervised"]["dice_loss"] == pytest.approx(canonical_components["dice"], abs=1e-6)
        assert streamed_arm["native_supervised"]["total"] == pytest.approx(float(canonical_total), abs=1e-6)
        assert streamed_arm["fallback"] is fallbacks[arm]


def test_stream_accumulator_rejects_gaps_and_incomplete_coverage():
    gt = np.zeros((3, 1, 2), dtype=np.uint8)
    logits = {arm: np.zeros((4, 1, 1, 2), dtype=np.float32) for arm in ARMS}
    accumulator = GradientBridgeFrameAccumulator("p", 0, gt.shape)
    with pytest.raises(MetricsIntegrityError, match="expected z_start=0"):
        accumulator.update_chunk(1, gt[:1], logits)
    one_slice = {arm: values.copy() for arm, values in logits.items()}
    accumulator.update_chunk(0, gt[:1], one_slice)
    with pytest.raises(MetricsIntegrityError, match="incomplete frame coverage"):
        accumulator.finalize()


def test_absent_positive_class_has_null_dice_and_ignore_only_support_has_undefined_loss():
    gt = np.full((1, 1, 2), 255, dtype=np.uint8)
    logits = np.zeros((4, *gt.shape), dtype=np.float32)
    result = score_frame_logits("p1", 0, gt, {arm: logits.copy() for arm in ARMS})
    for arm in ARMS:
        assert result["arms"][arm]["class_dice"]["RV"] is None
        assert result["arms"][arm]["class_metrics"]["RV"]["reference_count"] == 0
        assert result["arms"][arm]["class_metrics"]["RV"]["fp"] == 0
        assert result["arms"][arm]["native_supervised"]["ce"] is None
        assert result["arms"][arm]["native_supervised"]["undefined_reason"] == "empty_valid_support"


def test_frame_to_patient_to_class_reduction_is_class_first_with_unequal_absent_support():
    frames = [
        _manual_frame("a", 0, {"BG": 1.0, "RV": 1.0, "MYO": 1.0, "LV": 1.0}, valid_voxels=4),
        _manual_frame("b", 0, {"BG": 2 / 3, "RV": 0.0, "MYO": None, "LV": None}, valid_voxels=2),
    ]
    patient_rows = aggregate_patient_rows(frames)
    summary = summarize_patient_rows(patient_rows)
    assert summary["D0"]["class_dice"]["RV"] == pytest.approx(0.5)
    assert summary["D0"]["class_dice"]["MYO"] == pytest.approx(1.0)
    assert summary["D0"]["class_dice"]["LV"] == pytest.approx(1.0)
    assert summary["D0"]["foreground_dice"] == pytest.approx(5 / 6)
    assert patient_rows[1]["arms"]["D0"]["class_dice"]["MYO"] is None


def test_parity_with_historical_evaluator_and_patient_class_counts_absent_fp():
    from self_audit.compact_inference import compute_historical_metrics

    fixtures = [
        (
            "pA",
            "0",
            np.asarray([[[0, 1, 1, 2]]], dtype=np.uint8),
            np.asarray([[[0, 1, 0, 2]]], dtype=np.uint8),
        ),
        (
            "pA",
            "1",
            np.asarray([[[0, 0, 3, 255]]], dtype=np.uint8),
            np.asarray([[[1, 0, 0, 1]]], dtype=np.uint8),
        ),
        (
            "pB",
            "0",
            np.asarray([[[0, 1]]], dtype=np.uint8),
            np.asarray([[[0, 1]]], dtype=np.uint8),
        ),
    ]
    scored_frames = [
        score_frame_logits(
            patient_id,
            frame_id,
            gt,
            {arm: _logits_for_prediction(prediction) for arm in ARMS},
        )
        for patient_id, frame_id, gt, prediction in fixtures
    ]
    patient_rows = aggregate_patient_rows(scored_frames)
    patient_by_id = {row["patient_id"]: row for row in patient_rows}
    our_summary = summarize_patient_rows(patient_rows)

    historical_frame_rows = []
    for record in scored_frames:
        d0 = record["arms"]["D0"]
        historical_frame_rows.append({
            "patient": record["patient_id"],
            "valid_voxels": record["valid_voxels"],
            "raw": [
                {
                    "dice": d0["class_dice"][name],
                    "reference_present": d0["class_metrics"][name]["reference_present"],
                    "prediction_count": d0["class_metrics"][name]["prediction_count"],
                }
                for name in CLASS_NAMES
            ],
        })
    historical = compute_historical_metrics(historical_frame_rows, ["pA", "pB"])

    for class_name in CLASS_NAMES:
        assert our_summary["D0"]["class_dice"][class_name] == pytest.approx(
            historical["macro_patient"][class_name]["dice"]
        )
        for patient_id in ("pA", "pB"):
            actual = patient_by_id[patient_id]["arms"]["D0"]["class_dice"][class_name]
            expected = historical["patients"][patient_id][class_name]
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected)
    assert our_summary["D0"]["macro4_dice"] == pytest.approx(historical["class_first_macro4"])
    assert our_summary["D0"]["foreground_dice"] == pytest.approx(historical["class_first_foreground_macro"])
    assert our_summary["D0"]["foreground_dice"] != pytest.approx(
        historical["patient_first_foreground_macro"]
    )

    patient_rv = patient_by_id["pA"]["arms"]["D0"]["class_counts"]["RV"]
    assert patient_rv == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "reference_count": 2,
        "prediction_count": 2,
    }
    patient_rv_absent_fp = patient_by_id["pA"]["arms"]["D0"]["absent_class_fp"]["RV"]
    assert patient_rv_absent_fp == {
        "absent_frames": 1,
        "absent_valid_voxels": 3,
        "false_positive_voxels": 1,
        "rate": pytest.approx(1 / 3),
    }
    for class_name in CLASS_NAMES:
        total_absent_valid = sum(
            row["arms"]["D0"]["absent_class_fp"][class_name]["absent_valid_voxels"]
            for row in patient_rows
        )
        total_absent_fp = sum(
            row["arms"]["D0"]["absent_class_fp"][class_name]["false_positive_voxels"]
            for row in patient_rows
        )
        assert total_absent_valid == historical["absent_class_tracking"][class_name]["absent_valid_voxels"]
        assert total_absent_fp == historical["absent_class_tracking"][class_name]["absent_false_positive_voxels"]


def test_bootstrap_reuses_same_patient_resamples_is_deterministic_and_retains_fallback():
    rows = [
        _row("p1", 0.5, 0.5, 0.7, fallback=True),
        _row("p2", 0.7, 0.6, 0.8),
        _row("p3", 0.4, 0.5, 0.6),
        _row("p4", 0.8, 0.7, 0.9),
    ]
    first = paired_patient_bootstrap(rows, n_resamples=80, seed=123, confidence_level=0.95)
    second = paired_patient_bootstrap(rows, n_resamples=80, seed=123, confidence_level=0.95)
    assert first == second
    assert len(first["shared_resample_matrix"]) == 80
    assert all(len(sample) == 4 for sample in first["shared_resample_matrix"])
    assert first["contrasts"]["D2-D0"]["metrics"]["foreground_dice"]["bootstrap_deltas"]
    assert first["contrasts"]["D2-D1"]["metrics"]["foreground_dice"]["bootstrap_deltas"]
    assert first["contrasts"]["D2-D0"]["leave_one_patient_out"]["foreground_dice"]["coverage"] == {
        "defined": 4,
        "undefined": 0,
        "total": 4,
    }
    result = evaluate_paired_gate(rows, _gate_config(minimum=0.0, n_resamples=80, seed=123))
    assert result["cohort"]["patient_count"] == 4
    assert result["cohort"]["fallback_patients_retained"] == 1
    assert "p1" in result["cohort"]["patient_ids"]


def test_gate_go_requires_both_d2_contrasts_to_clear_locked_ci_and_loo():
    rows = [_row(f"p{i}", 0.5, 0.5, 0.8) for i in range(6)]
    result = evaluate_paired_gate(rows, _gate_config(minimum=0.1, n_resamples=120, seed=3))
    assert result["status"] == "GO"
    assert set(result["contrasts"]) == {"D2-D0", "D2-D1"}
    for contrast in result["contrasts"].values():
        assert contrast["status"] == "GO"
        assert contrast["point_delta"] == pytest.approx(0.3)
        assert contrast["confidence_interval"]["lower"] > 0.1
        assert contrast["loo_rule_passed"] is True
    # D3-D5 have no role in the gate criterion.
    rows = [_row(f"q{i}", 0.5, 0.5, 0.8) for i in range(6)]
    for row in rows:
        row["arms"]["D4"]["class_dice"] = {name: 0.0 for name in CLASS_NAMES}
        row["arms"]["D5"]["class_dice"] = {name: 0.0 for name in CLASS_NAMES}
    assert evaluate_paired_gate(rows, _gate_config(minimum=0.1, n_resamples=40, seed=3))["status"] == "GO"


def test_gate_no_go_when_ci_upper_or_locked_loo_rule_rules_out_minimum():
    rows = [_row(f"p{i}", 0.7, 0.7, 0.4) for i in range(5)]
    result = evaluate_paired_gate(rows, _gate_config(minimum=0.1, n_resamples=100, seed=4))
    assert result["status"] == "NO_GO"
    assert result["contrasts"]["D2-D0"]["confidence_interval"]["upper"] <= 0.1
    assert result["contrasts"]["D2-D1"]["confidence_interval"]["upper"] <= 0.1


def test_gate_inconclusive_when_paired_interval_overlaps_minimum_without_loo_failure():
    # Patient-level deltas are 0.09, 0.12, 0.12: every LOO mean is >= .1,
    # while bootstrap draws can omit the higher observations and cross .1.
    rows = [
        _row("p1", 0.5, 0.5, 0.59),
        _row("p2", 0.5, 0.5, 0.62),
        _row("p3", 0.5, 0.5, 0.62),
    ]
    result = evaluate_paired_gate(rows, _gate_config(minimum=0.1, n_resamples=500, seed=5))
    contrast = result["contrasts"]["D2-D0"]
    assert contrast["status"] == "INCONCLUSIVE"
    assert contrast["point_delta"] > 0.1
    assert contrast["confidence_interval"]["lower"] <= 0.1 < contrast["confidence_interval"]["upper"]
    assert contrast["loo_rule_passed"] is True
    assert result["status"] == "INCONCLUSIVE"


def test_gate_marks_undefined_bootstrap_and_loo_coverage_inconclusive():
    rows = [
        _row("p1", 0.5, 0.5, 0.8, support={"BG": True, "RV": True, "MYO": False, "LV": False}),
        _row("p2", 0.5, 0.5, 0.8, support={"BG": True, "RV": False, "MYO": False, "LV": False}),
    ]
    result = evaluate_paired_gate(rows, _gate_config(minimum=0.1, n_resamples=80, seed=1))
    assert result["status"] == "INCONCLUSIVE"
    for contrast in result["contrasts"].values():
        loo = contrast["leave_one_patient_out"]
        assert loo["coverage"]["undefined"] == 1
        coverage = contrast["bootstrap_coverage"]
        assert coverage["fraction"] < 1.0
        assert coverage["undefined"] > 0


def test_gate_integrity_and_nonfinite_blockers_are_reported_separately():
    rows = [_row("p1", 0.5, 0.5, 0.8), _row("p2", 0.5, 0.5, 0.8)]
    malformed = [rows[0], rows[0]]
    blocked = evaluate_paired_gate(malformed, _gate_config())
    assert blocked["status"] == "BLOCKED"
    assert blocked["blocking"]["integrity"]
    assert blocked["blocking"]["nonfinite"] == []

    nonfinite = [_row("p1", 0.5, 0.5, 0.8), _row("p2", 0.5, 0.5, 0.8)]
    nonfinite[0]["arms"]["D2"]["class_dice"]["RV"] = float("nan")
    blocked = evaluate_paired_gate(nonfinite, _gate_config())
    assert blocked["status"] == "BLOCKED"
    assert blocked["blocking"]["integrity"] == []
    assert blocked["blocking"]["nonfinite"]

    explicit = evaluate_paired_gate(
        rows,
        _gate_config(),
        integrity_errors=["frozen prediction receipt mismatch"],
        nonfinite_errors=["D2 logits contained NaN"],
    )
    assert explicit["blocking"]["integrity"] == ["frozen prediction receipt mismatch"]
    assert explicit["blocking"]["nonfinite"] == ["D2 logits contained NaN"]


def test_nonfinite_frozen_logits_fail_closed_with_distinct_error_type():
    gt = np.zeros((1, 1, 1), dtype=np.uint8)
    logits = np.zeros((4, *gt.shape), dtype=np.float32)
    arms = {arm: logits.copy() for arm in ARMS}
    arms["D2"][0, 0, 0, 0] = np.nan
    with pytest.raises(NonFiniteMetricsError, match="D2 frozen chunk logits"):
        score_frame_logits("p1", 0, gt, arms)


def test_shared_reference_support_mismatch_across_arms_is_rejected():
    row = _row("p1", 0.5, 0.5, 0.8)
    row["arms"]["D2"]["class_dice"]["RV"] = None
    row["arms"]["D2"]["class_dice"]["MYO"] = None
    row["arms"]["D2"]["class_dice"]["LV"] = None
    result = evaluate_paired_gate([row], _gate_config())
    assert result["status"] == "BLOCKED"
    assert "support" in result["blocking"]["integrity"][0]
