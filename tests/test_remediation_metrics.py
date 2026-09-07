"""Regression tests for the metrics-and-evaluation remediation pass (AGY-8, part 1).

This file covers the *pure numeric* half of the remediation contract: the one
canonical neutral margin, the empty-class policy, the headroom estimator, the
attribution identity, and batch-partition invariance of the reducers.  The
model-level and artifact-level half lives in ``test_remediation_evaluation.py``.

Every assertion here pins a specific number or a specific structural property.
Nothing in this file asserts merely that a call does not raise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from self_audit.audit import semantics
from self_audit.audit.semantics import (
    BENEFICIAL,
    DEFAULT_NEUTRAL_MARGIN,
    HARMFUL,
    NEUTRAL,
    classify_delta,
    macro_mean,
    resolve_neutral_margin,
)
from self_audit.audit.targets import multiclass_dice
from self_audit.evaluation import metrics as metrics_module
from self_audit.evaluation.audit_decomposition import (
    AuditModeSamples,
    StageTransition,
    audit_mode_metrics,
    decompose_self_audit_output,
    stage_transition_metrics,
)
from self_audit.evaluation.metrics import (
    acceptance_metrics,
    per_class_dice,
    slice_proxy_dice,
    transition_audit_metrics,
)
from self_audit.evaluation.threshold import evaluate_threshold


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _one_hot_logits(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Hard one-hot logits so Dice is exactly reproducible from the labels."""

    return (
        torch.nn.functional.one_hot(labels.long(), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
        * 8.0
    )


def _three_class_target() -> torch.Tensor:
    """A [1,3,2] target holding every foreground class exactly once per row.

    No foreground class is empty, so every Dice below is policy-independent and
    a perfect prediction scores exactly 1.0 while an all-background prediction
    scores exactly 0.0.
    """

    return torch.tensor([[[1, 1], [2, 2], [3, 3]]], dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. neutral deltas are neither harmful nor beneficial
# ---------------------------------------------------------------------------


def test_neutral_deltas_are_neither_harmful_nor_beneficial() -> None:
    result = acceptance_metrics(
        accepted=[True, True, True],
        actual_delta_dice=[0.0, +0.001, -0.001],
        neutral_margin=0.005,
    )

    assert result["beneficial_count"] == 0
    assert result["harmful_count"] == 0
    assert result["neutral_count"] == 3
    assert result["harmful_acceptance_rate"] == 0.0
    # The whole accepted population was no-op traffic, and that is visible.
    assert result["neutral_acceptance_rate"] == 1.0
    assert result["accepted_count"] == 3
    assert result["rejected_count"] == 0
    assert result["harmful_accepted_count"] == 0


# ---------------------------------------------------------------------------
# 2. ONE epsilon, single-sourced, boundary closed on the neutral side
# ---------------------------------------------------------------------------


def test_canonical_margin_is_single_sourced_and_boundary_is_closed_on_neutral() -> None:
    assert DEFAULT_NEUTRAL_MARGIN == 0.005
    assert resolve_neutral_margin(None) == 0.005
    # Same object, not a re-declared literal: the metric layer imports the
    # canonical value rather than owning a second copy of it.
    assert metrics_module.DEFAULT_NEUTRAL_MARGIN is semantics.DEFAULT_NEUTRAL_MARGIN

    eps = DEFAULT_NEUTRAL_MARGIN
    # abs(delta) == eps is NEUTRAL (closed on the neutral side).
    assert classify_delta([eps, -eps]).tolist() == [NEUTRAL, NEUTRAL]
    # One ULP outside the margin already flips the class.
    just_above = float(np.nextafter(eps, 1.0))
    assert classify_delta([just_above, -just_above]).tolist() == [BENEFICIAL, HARMFUL]
    assert classify_delta([0.0, 0.006, -0.006]).tolist() == [NEUTRAL, BENEFICIAL, HARMFUL]


def test_every_module_classifies_the_same_delta_vector_identically() -> None:
    """metrics / threshold / audit_decomposition must agree, boundary included."""

    delta = [0.005, -0.005, 0.006, -0.006]  # neutral, neutral, beneficial, harmful

    accepted = acceptance_metrics(
        accepted=[True, True, True, True], actual_delta_dice=delta, neutral_margin=None
    )
    assert (
        accepted["neutral_count"],
        accepted["beneficial_count"],
        accepted["harmful_count"],
    ) == (2, 1, 1)

    transition = transition_audit_metrics(
        predicted_local=torch.zeros(4, 2, 2, dtype=torch.long),
        target_local=torch.zeros(4, 2, 2, dtype=torch.long),
        predicted_delta_q=[1.0, 1.0, 1.0, 1.0],
        actual_delta_dice=delta,
    )
    assert (
        transition["neutral_count"],
        transition["beneficial_count"],
        transition["harmful_count"],
    ) == (2, 1, 1)
    assert transition["ranking_count"] == 2  # the two neutrals are dropped

    # threshold.py: tau below every delta_q, so all four are accepted.
    thresholded = evaluate_threshold(
        tau_accept=0.0,
        initial_dice=[0.0, 0.0, 0.0, 0.0],
        delta_q=[[1.0], [1.0], [1.0], [1.0]],
        actual_delta_dice=[[d] for d in delta],
    )
    assert thresholded["accepted_total"] == 4
    assert thresholded["harmful_total"] == 1
    assert thresholded["neutral_accepted_total"] == 2
    assert thresholded["accepted_non_neutral_total"] == 2

    # audit_decomposition.py
    stage = StageTransition(
        delta=torch.tensor(delta, dtype=torch.float64),
        accepted=torch.ones(4, dtype=torch.bool),
        previous_dice=torch.zeros(4, dtype=torch.float64),
        candidate_dice=torch.tensor(delta, dtype=torch.float64),
        attempt_count=4,
        row_count=4,
    )
    decomposed = stage_transition_metrics([stage])
    assert decomposed["stage_0/neutral_candidate_rate"] == pytest.approx(0.5)
    assert decomposed["stage_0/beneficial_candidate_rate"] == pytest.approx(0.25)
    assert decomposed["stage_0/harmful_candidate_rate"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 3. headroom capture is a ratio of sums, not a mean of ratios
# ---------------------------------------------------------------------------


def _mode_samples(initial, always, self_dice, oracle) -> AuditModeSamples:
    as_tensor = lambda values: torch.tensor(values, dtype=torch.float64)  # noqa: E731
    return AuditModeSamples(
        initial_dice=as_tensor(initial),
        always_dice=as_tensor(always),
        self_dice=as_tensor(self_dice),
        oracle_dice=as_tensor(oracle),
        stages=[],
    )


def test_headroom_capture_is_a_ratio_of_sums_not_a_mean_of_ratios() -> None:
    # oracle_headroom = [2e-8, 0.30]; self_gain = [0.0125, 0.20].
    # Mean-of-ratios with a 1e-8 guard would return ~3.1e5; the correct
    # ratio-of-sums over MATERIAL headroom is 0.20 / 0.30.
    samples = _mode_samples(
        initial=[0.0, 0.0],
        always=[0.0, 0.0],
        self_dice=[0.0125, 0.20],
        oracle=[2e-8, 0.30],
    )
    result = audit_mode_metrics(samples, neutral_margin=0.005)

    assert result["modes/headroom_capture_ratio"] == pytest.approx(0.20 / 0.30, abs=1e-12)
    assert result["modes/headroom_capture_ratio"] < 1.0  # never the ~3e5 blow-up
    assert result["modes/headroom_eligible_count"] == 1.0
    assert result["modes/headroom_available_rate"] == pytest.approx(0.5)
    assert result["modes/oracle_headroom_sum"] == pytest.approx(0.30)
    assert result["modes/self_gain_on_headroom_sum"] == pytest.approx(0.20)


def test_cohort_without_material_headroom_reports_nan_never_zero() -> None:
    samples = _mode_samples(
        initial=[0.40, 0.40],
        always=[0.40, 0.40],
        self_dice=[0.40, 0.40],
        oracle=[0.4 + 2e-8, 0.40 + 0.001],  # both inside the neutral margin
    )
    result = audit_mode_metrics(samples, neutral_margin=0.005)

    ratio = result["modes/headroom_capture_ratio"]
    assert math.isnan(ratio), f"expected nan for a zero-headroom cohort, got {ratio!r}"
    assert ratio != 0.0 or math.isnan(ratio)
    assert result["modes/headroom_eligible_count"] == 0.0
    assert result["modes/headroom_available_rate"] == 0.0


# ---------------------------------------------------------------------------
# 4. attribution identity, with the signs verified
# ---------------------------------------------------------------------------


def _single_row_stage_output(previous_labels, candidate_labels, accepted: bool) -> dict:
    return {
        "transition_previous": [_one_hot_logits(previous_labels)],
        "transition_candidates": [_one_hot_logits(candidate_labels)],
        "transition_active_masks": [torch.tensor([True])],
        "transition_state_masks": [torch.tensor([bool(accepted)])],
    }


def test_attribution_identity_holds_and_rejecting_a_harmful_candidate_is_positive() -> None:
    target = _three_class_target()
    perfect = target.clone()
    background = torch.zeros_like(target)

    # Harmful candidate (1.0 -> 0.0) that the gate rejected: gate value POSITIVE.
    harmful = decompose_self_audit_output(
        _single_row_stage_output(perfect, background, accepted=False), target
    )
    assert harmful["stage_0/candidate_gain"] == pytest.approx(-1.0)
    assert harmful["stage_0/realized_gain"] == pytest.approx(0.0)
    assert harmful["stage_0/audit_gate_value"] == pytest.approx(+1.0)
    assert harmful["stage_0/audit_gate_value"] > 0.0
    assert harmful["stage_0/attribution_residual"] < 1e-6

    # Beneficial candidate (0.0 -> 1.0) that the gate rejected: gate value NEGATIVE.
    beneficial = decompose_self_audit_output(
        _single_row_stage_output(background, perfect, accepted=False), target
    )
    assert beneficial["stage_0/candidate_gain"] == pytest.approx(+1.0)
    assert beneficial["stage_0/realized_gain"] == pytest.approx(0.0)
    assert beneficial["stage_0/audit_gate_value"] == pytest.approx(-1.0)
    assert beneficial["stage_0/audit_gate_value"] < 0.0
    assert beneficial["stage_0/attribution_residual"] < 1e-6

    # Accepting a beneficial candidate realizes it and leaves no gate value.
    accepted = decompose_self_audit_output(
        _single_row_stage_output(background, perfect, accepted=True), target
    )
    assert accepted["stage_0/realized_gain"] == pytest.approx(+1.0)
    assert accepted["stage_0/audit_gate_value"] == pytest.approx(0.0)
    assert accepted["stage_0/attribution_residual"] < 1e-6


def test_attribution_identity_holds_on_a_mixed_multi_row_stage() -> None:
    rng = np.random.default_rng(20260906)
    delta = torch.tensor(rng.uniform(-0.4, 0.4, size=64), dtype=torch.float64)
    accepted = torch.tensor(rng.integers(0, 2, size=64).astype(bool))
    stage = StageTransition(
        delta=delta,
        accepted=accepted,
        previous_dice=torch.zeros(64, dtype=torch.float64),
        candidate_dice=delta.clone(),
        attempt_count=64,
        row_count=64,
    )
    result = stage_transition_metrics([stage])
    identity = (
        result["stage_0/candidate_gain"]
        + result["stage_0/audit_gate_value"]
        - result["stage_0/realized_gain"]
    )
    assert abs(identity) < 1e-6
    assert result["stage_0/attribution_residual"] < 1e-6
    assert result["audit/attribution_residual"] < 1e-6


# ---------------------------------------------------------------------------
# 5. reducers are invariant to batch partitioning (pure form)
# ---------------------------------------------------------------------------


def _stage_from(delta: torch.Tensor, accepted: torch.Tensor) -> StageTransition:
    return StageTransition(
        delta=delta,
        accepted=accepted,
        previous_dice=torch.zeros_like(delta),
        candidate_dice=delta.clone(),
        attempt_count=int(delta.numel()),
        row_count=int(delta.numel()),
    )


def test_mode_and_stage_reducers_are_invariant_to_batch_partitioning() -> None:
    rng = np.random.default_rng(7)
    count = 6
    initial = torch.tensor(rng.uniform(0.2, 0.6, count), dtype=torch.float64)
    always = initial + torch.tensor(rng.uniform(-0.1, 0.2, count), dtype=torch.float64)
    self_dice = initial + torch.tensor(rng.uniform(-0.05, 0.25, count), dtype=torch.float64)
    oracle = initial + torch.tensor(rng.uniform(0.0, 0.4, count), dtype=torch.float64)
    delta = torch.tensor(rng.uniform(-0.3, 0.3, count), dtype=torch.float64)
    accepted = torch.tensor(rng.integers(0, 2, count).astype(bool))

    def build(lo: int, hi: int) -> AuditModeSamples:
        return AuditModeSamples(
            initial_dice=initial[lo:hi].clone(),
            always_dice=always[lo:hi].clone(),
            self_dice=self_dice[lo:hi].clone(),
            oracle_dice=oracle[lo:hi].clone(),
            stages=[_stage_from(delta[lo:hi].clone(), accepted[lo:hi].clone())],
        )

    whole = build(0, 6)
    # Split 1 / 2 / 3, merged in loader order.
    split = build(0, 1).merged_with(build(1, 3)).merged_with(build(3, 6))

    whole_metrics = {**audit_mode_metrics(whole), **stage_transition_metrics(whole.stages)}
    split_metrics = {**audit_mode_metrics(split), **stage_transition_metrics(split.stages)}

    assert set(whole_metrics) == set(split_metrics)
    for key, value in whole_metrics.items():
        other = split_metrics[key]
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(other), key
        else:
            assert value == other, f"{key}: {value!r} != {other!r}"
    # Non-degenerate: the aggregate actually carries all six rows.
    assert whole_metrics["modes/sample_count"] == 6.0
    assert whole_metrics["stage_0/attempt_count"] == 6.0


def test_acceptance_metrics_counts_are_additive_across_any_partition() -> None:
    rng = np.random.default_rng(11)
    delta = rng.uniform(-0.05, 0.05, 30)
    accepted = rng.integers(0, 2, 30).astype(bool)

    whole = acceptance_metrics(accepted, delta)
    parts = [acceptance_metrics(accepted[a:b], delta[a:b]) for a, b in ((0, 5), (5, 15), (15, 30))]

    for key in (
        "harmful_accepted_count",
        "beneficial_rejected_count",
        "neutral_accepted_count",
        "accepted_count",
        "rejected_count",
        "neutral_count",
        "beneficial_count",
        "harmful_count",
    ):
        assert whole[key] == sum(part[key] for part in parts), key
    # And the headline rate is recoverable from those counts alone.
    assert whole["harmful_acceptance_rate"] == pytest.approx(
        sum(p["harmful_accepted_count"] for p in parts)
        / max(sum(p["accepted_count"] for p in parts), 1)
    )


# ---------------------------------------------------------------------------
# 6. one proxy Dice for Phase A and Phase C; batch-size invariant
# ---------------------------------------------------------------------------


def _proxy_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    """Two 4x4 samples whose per-sample macro differs from the pooled macro."""

    target = torch.zeros(2, 4, 4, dtype=torch.long)
    target[0] = 1
    target[1] = 1
    prediction = torch.zeros(2, 4, 4, dtype=torch.long)
    prediction[0] = 1  # sample 0 perfect
    # sample 1 predicts pure background -> class 1 scores exactly 0.0
    return prediction, target


def test_slice_proxy_dice_is_batch_partition_invariant() -> None:
    prediction, target = _proxy_fixture()

    whole = slice_proxy_dice(prediction, target, num_classes=4)
    assert whole.shape == (2,)
    assert whole[0] == pytest.approx(1.0)
    assert whole[1] == pytest.approx(0.0)
    assert macro_mean(whole) == pytest.approx(0.5)

    per_sample = np.concatenate(
        [slice_proxy_dice(prediction[i : i + 1], target[i : i + 1], num_classes=4) for i in range(2)]
    )
    assert np.array_equal(whole, per_sample)

    # The pooled confusion-count form is a DIFFERENT number and is the one the
    # Phase-A headline must stop using (AGY-7 owns that wiring).
    pooled = macro_mean(list(per_class_dice(prediction, target, num_classes=4).values()))
    assert pooled == pytest.approx(2.0 / 3.0)
    assert abs(pooled - float(macro_mean(whole))) > 0.1


def test_phase_a_and_phase_c_proxy_dice_agree_on_identical_inputs() -> None:
    prediction, target = _proxy_fixture()

    # Phase C today scores per sample with ``multiclass_dice`` (legacy_one for a
    # class empty on both sides).  The shared proxy must reproduce it exactly
    # under the matching policy, so the two headline numbers are comparable.
    phase_c = multiclass_dice(_one_hot_logits(prediction), target).numpy().astype(np.float64)
    shared = slice_proxy_dice(prediction, target, num_classes=4, empty_policy="legacy_one")

    assert shared.shape == phase_c.shape == (2,)
    assert float(np.max(np.abs(shared - phase_c))) < 1e-6
    assert shared[0] == pytest.approx(1.0)
    assert shared[1] == pytest.approx(2.0 / 3.0)


def test_phase_a_validation_routes_through_the_shared_proxy() -> None:
    """AGY-7 owns this wiring; assert the property once it lands."""

    from self_audit.training import train_annotation

    if not hasattr(train_annotation, "slice_proxy_dice"):
        pytest.skip(
            "AGY-7 has not landed: training/train_annotation.py still aggregates with "
            "per_class_dice over a whole [B,H,W] block, which is batch-size dependent. "
            "The property it must satisfy is pinned by "
            "test_slice_proxy_dice_is_batch_partition_invariant above."
        )
    prediction, target = _proxy_fixture()
    routed = train_annotation.slice_proxy_dice(prediction, target, num_classes=4)
    assert np.array_equal(routed, slice_proxy_dice(prediction, target, num_classes=4))


# ---------------------------------------------------------------------------
# 7. empty-class policy: exclude both-empty, never exclude a one-sided miss
# ---------------------------------------------------------------------------


def test_both_empty_class_is_excluded_from_the_headline_macro() -> None:
    # Class 1 present in both; classes 2 and 3 empty in both.
    target = torch.ones(1, 4, 4, dtype=torch.long)
    prediction = torch.ones(1, 4, 4, dtype=torch.long)

    excluded = per_class_dice(prediction, target, num_classes=4)
    assert excluded[1] == pytest.approx(1.0)
    assert math.isnan(excluded[2])
    assert math.isnan(excluded[3])
    assert macro_mean(list(excluded.values())) == pytest.approx(1.0)

    legacy = per_class_dice(prediction, target, num_classes=4, empty_policy="legacy_one")
    assert legacy[2] == 1.0
    assert legacy[3] == 1.0
    assert macro_mean(list(legacy.values())) == pytest.approx(1.0)

    # The two policies genuinely differ once a real class is imperfect.
    half = torch.zeros(1, 4, 4, dtype=torch.long)
    half[0, :2] = 1
    assert macro_mean(
        list(per_class_dice(half, target, num_classes=4).values())
    ) == pytest.approx(2.0 / 3.0)
    assert macro_mean(
        list(per_class_dice(half, target, num_classes=4, empty_policy="legacy_one").values())
    ) == pytest.approx((2.0 / 3.0 + 1.0 + 1.0) / 3.0)


def test_class_present_in_gt_but_predicted_empty_scores_zero_under_both_policies() -> None:
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :, :2] = 1  # class 1 present in GT
    target[0, :, 2:] = 2  # class 2 present in GT
    prediction = torch.zeros(1, 4, 4, dtype=torch.long)
    prediction[0] = 2  # class 1 never predicted; class 2 over-predicted

    for policy in (None, "exclude", "legacy_one", "zero"):
        scores = per_class_dice(prediction, target, num_classes=4, empty_policy=policy)
        assert scores[1] == 0.0, f"policy={policy!r} must not hide a total miss"
        assert scores[2] == pytest.approx(2.0 * 8 / (16 + 8))
        # class 3 is empty on both sides and follows the policy
        if policy in (None, "exclude"):
            assert math.isnan(scores[3])
        elif policy == "legacy_one":
            assert scores[3] == 1.0
        else:
            assert scores[3] == 0.0
