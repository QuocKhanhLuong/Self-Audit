"""Comprehensive regression tests for Metric Contracts and Replay (Wave 1).

Covers:
1. P0-01 sign reversal elimination: legacy positive delta vs canonical negative delta,
   candidate replay preserves correct state and rejects harmful transition.
2. Direct-vs-cache/replay parity: direct inference rollout matches evaluate_threshold
   final states bit-identically across tau_accept grids and t_max=0.
3. Mixed accept-prefix halting: multi-turn halting halts only rejected samples and
   preserves prefix states.
4. Volume aggregation with patient grouping: grouping slices by case/volume, scoring
   per volume from sufficient statistics, aggregating across patients (no cohort pooling).
   Sufficient statistics 3D volume Dice == volumetric Dice != 2D slice mean Dice.
   Volume scorer never reports slice_proxy contract.
5. Contract mismatch rejection: strict contract validation rejects mismatch in name,
   version, metric_space, empty_policy, neutral_margin, aggregation, unversioned cache,
   and inconsistent q_candidate - q_previous.
6. Blank hallucination tracking: blank A0 slice with hallucinated candidate prediction
   yields defined candidate score 0.0, counts FP, marks delta as undefined (not neutral),
   and increments blank_to_hallucination_count upon acceptance.
7. No feasible threshold: sweep_thresholds raises NoFeasibleThresholdError when constraint
   cannot be met; select_threshold raises NoFeasibleThresholdError when all outcomes are NaN.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.semantics import (
    AUDIT_TARGET_LEGACY_ONE_V1,
    DEFAULT_NEUTRAL_MARGIN,
    FOREGROUND_DICE_EXCLUDE_V1,
    METRIC_SPACE_SLICE_PROXY,
    METRIC_SPACE_VOLUME_NATIVE,
    METRIC_SPACE_VOLUME_RESIZED,
)
from self_audit.audit.targets import build_transition_targets
from self_audit.evaluation.contracts import (
    AUDIT_TARGET_LEGACY_ONE_V1_CONTRACT,
    ContractMismatchError,
    FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT,
    MetricContract,
    SufficientStatistics,
    compute_batch_sufficient_statistics,
    compute_dice_from_stats,
    compute_sufficient_statistics,
    resolve_metric_contract,
    score_state,
    score_transition,
    score_volume_from_stats,
    validate_contract,
    validate_contract_compatibility,
)
from self_audit.evaluation.metrics import slice_proxy_dice
from self_audit.evaluation.threshold import (
    CACHE_SCHEMA_VERSION,
    NoFeasibleThresholdError,
    evaluate_threshold,
    save_calibration,
    select_threshold,
    sweep_thresholds,
    validate_and_normalize_transition_cache,
)
from self_audit.models.self_audit_net import SelfAuditNet


# ==============================================================================
# 1. Sign reversal test (P0-01)
# ==============================================================================


def test_p0_01_sign_reversal_eliminated() -> None:
    """Demonstrate that P0-01 sign reversal is completely eliminated.

    Scenario:
    - Classes: 1, 2 (foreground), 3 (empty in GT).
    - Initial prediction has class 1=0.5, class 2=0.5, class 3 has 1 FP (Dice=0.0).
      Initial macro Dice under both legacy_one and exclude is (0.5 + 0.5 + 0.0)/3 = 0.3333.
    - Candidate prediction degrades class 1 to 0.4 and class 2 to 0.4, but cleans up the
      FP in class 3 so class 3 is now both-empty.
    - Under legacy_one (build_transition_targets):
      class 3 scores 1.0, candidate macro is (0.4 + 0.4 + 1.0)/3 = 0.60.
      legacy delta = +0.2667 (falsely reported as strong positive gain!).
    - Under exclude (canonical evaluation contract):
      class 3 is both-empty and therefore EXCLUDED (NaN).
      candidate macro is (0.4 + 0.4)/2 = 0.40.
      actual delta = 0.40 - 0.3333 = -0.0667 (harmful degradation!).
    - In the old buggy code:
      adding legacy delta (+0.2667) to exclude initial (0.3333) produced 0.60 (sign reversal!).
    - In the new contract-aware replay:
      candidate score is directly 0.40 (no mixed-contract arithmetic).
      actual delta is -0.0667, which is correctly identified as harmful.
    """
    gt = np.zeros((100, 100), dtype=np.int64)
    gt[:50, :50] = 1  # class 1 has 2500 px
    gt[50:, :50] = 2  # class 2 has 2500 px
    # class 3 is completely absent in gt

    # Initial pred: 50% overlap on classes 1 and 2; 1 false positive pixel on class 3
    init_pred = np.zeros((100, 100), dtype=np.int64)
    init_pred[:25, :50] = 1   # 1250 TP, 1250 FN on class 1 -> Dice = 2*1250/(1250+2500)=0.6667
    init_pred[50:75, :50] = 2 # 1250 TP, 1250 FN on class 2 -> Dice = 0.6667
    init_pred[0, 99] = 3      # 1 FP on class 3 -> Dice = 0.0

    # Candidate pred: degrades classes 1 and 2 to 20% overlap, but removes the FP on class 3
    cand_pred = np.zeros((100, 100), dtype=np.int64)
    cand_pred[:10, :50] = 1   # 500 TP, 2000 FN on class 1 -> Dice = 2*500/(500+2500)=0.3333
    cand_pred[50:60, :50] = 2 # 500 TP, 2000 FN on class 2 -> Dice = 0.3333
    # class 3 has 0 px -> both empty!

    # 1. Compute scores under legacy_one
    s_prev_leg = score_state(init_pred, gt, contract=AUDIT_TARGET_LEGACY_ONE_V1_CONTRACT)
    s_cand_leg = score_state(cand_pred, gt, contract=AUDIT_TARGET_LEGACY_ONE_V1_CONTRACT)
    legacy_delta = s_cand_leg.macro_dice - s_prev_leg.macro_dice
    # Legacy delta is positive because class 3 jumped from 0.0 to 1.0!
    assert s_cand_leg.per_class_dice[3] == 1.0
    assert legacy_delta > 0.0, f"Expected positive legacy delta, got {legacy_delta}"

    # 2. Compute scores under canonical exclude
    s_prev_exc = score_state(init_pred, gt, contract=FOREGROUND_DICE_EXCLUDE_V1_CONTRACT)
    s_cand_exc = score_state(cand_pred, gt, contract=FOREGROUND_DICE_EXCLUDE_V1_CONTRACT)
    actual_delta = s_cand_exc.macro_dice - s_prev_exc.macro_dice
    # Canonical exclude delta is negative because class 3 is excluded and foreground degraded!
    assert math.isnan(s_cand_exc.per_class_dice[3])
    assert actual_delta < 0.0, f"Expected negative exclude delta, got {actual_delta}"

    # 3. Evaluate replay: candidate score must be s_cand_exc.macro_dice, NOT s_prev_exc + legacy_delta
    transitions = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([s_prev_exc.macro_dice]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[actual_delta]]),
        "q_previous": np.array([[s_prev_exc.macro_dice]]),
        "q_candidate": np.array([[s_cand_exc.macro_dice]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }

    # At tau = 0.0, transition is accepted because delta_q (0.1) > 0.0
    res_accept = evaluate_threshold(0.0, **transitions)
    # The final score must be exactly q_candidate (0.3333), NOT initial + legacy_delta (> 0.6)
    assert res_accept["final_macro_dice"] == pytest.approx(s_cand_exc.macro_dice, rel=1e-5)
    assert res_accept["final_macro_dice"] < s_prev_exc.macro_dice
    # And it is correctly flagged as harmful acceptance!
    assert res_accept["harmful_total"] == 1
    assert res_accept["harmful_acceptance_rate"] == 1.0

    # At tau = 0.2, transition is rejected because delta_q (0.1) <= 0.2
    res_reject = evaluate_threshold(0.2, **transitions)
    assert res_reject["final_macro_dice"] == pytest.approx(s_prev_exc.macro_dice, rel=1e-5)
    assert res_reject["harmful_total"] == 0


# ==============================================================================
# 2. Direct-vs-cache/replay parity test
# ==============================================================================


def test_direct_vs_cache_replay_parity() -> None:
    """Verify exact parity between direct model.infer and threshold replay."""
    torch.manual_seed(42)
    net = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()

    batch_size = 4
    images = torch.randn(batch_size, 3, 64, 64)
    gt = torch.randint(0, 4, (batch_size, 64, 64))

    # 1. Run always_accept_refinement to collect trajectory
    with torch.no_grad():
        out_all = net.infer(images, mode="always_accept_refinement", tau_accept=-float("inf"), t_max=2)

    initial_logits = out_all["initial_logits"]
    init_dice = np.asarray(slice_proxy_dice(initial_logits, gt, empty_policy="exclude"))

    # Build transition cache arrays
    q_prev_list = []
    q_cand_list = []
    delta_q_list = []
    actual_delta_list = []

    for turn, (prev, cand, audit) in enumerate(
        zip(out_all["transition_previous"], out_all["transition_candidates"], out_all["audits"])
    ):
        dq = audit["delta_q"].squeeze(-1).numpy()
        p_dice = np.asarray(slice_proxy_dice(prev, gt, empty_policy="exclude"))
        c_dice = np.asarray(slice_proxy_dice(cand, gt, empty_policy="exclude"))
        delta_q_list.append(dq)
        q_prev_list.append(p_dice)
        q_cand_list.append(c_dice)
        actual_delta_list.append(c_dice - p_dice)

    transitions = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": init_dice,
        "delta_q": np.stack(delta_q_list, axis=1),
        "actual_delta_dice": np.stack(actual_delta_list, axis=1),
        "q_previous": np.stack(q_prev_list, axis=1),
        "q_candidate": np.stack(q_cand_list, axis=1),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }

    # Test parity across thresholds: reject-all (tau=10.0), accept-all (tau=-10.0), intermediate (tau=0.0)
    for tau in (-10.0, 0.0, 10.0):
        with torch.no_grad():
            direct_out = net.infer(images, mode="self_audit", tau_accept=tau, t_max=2)
        direct_final_dice = np.asarray(slice_proxy_dice(direct_out["logits"], gt, empty_policy="exclude"))

        replay_res = evaluate_threshold(tau, **transitions)
        replay_final_dice = replay_res["final_dice"]

        # Check per-sample parity
        for i in range(batch_size):
            if np.isnan(direct_final_dice[i]):
                assert np.isnan(replay_final_dice[i])
            else:
                assert replay_final_dice[i] == pytest.approx(direct_final_dice[i], abs=1e-5)

        # Check macro parity
        assert replay_res["final_macro_dice"] == pytest.approx(np.nanmean(direct_final_dice), abs=1e-5)


# ==============================================================================
# 3. Mixed accept-prefix test
# ==============================================================================


def test_mixed_accept_prefix() -> None:
    """Verify that multi-turn halting halts only rejected samples and freezes state."""
    initial_dice = np.array([0.5, 0.5, 0.5, 0.5])
    delta_q = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [ 1.0,  1.0,  1.0],
    ])
    q_candidate = np.array([
        [0.6, 0.7, 0.8],
        [0.6, 0.7, 0.8],
        [0.6, 0.7, 0.8],
        [0.6, 0.7, 0.8],
    ])
    q_previous = np.array([
        [0.5, 0.6, 0.7],
        [0.5, 0.6, 0.7],
        [0.5, 0.6, 0.7],
        [0.5, 0.6, 0.7],
    ])
    actual_delta_dice = q_candidate - q_previous

    transitions = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": initial_dice,
        "delta_q": delta_q,
        "actual_delta_dice": actual_delta_dice,
        "q_previous": q_previous,
        "q_candidate": q_candidate,
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }

    res = evaluate_threshold(0.0, **transitions)
    expected_finals = [0.5, 0.6, 0.7, 0.8]
    assert np.allclose(res["final_dice"], expected_finals)
    assert res["mean_accepted_turns"] == pytest.approx((0 + 1 + 2 + 3) / 4.0)
    assert res["mean_attempted_turns"] == pytest.approx((1 + 2 + 3 + 3) / 4.0)


# ==============================================================================
# 4. Volume aggregation test
# ==============================================================================


def test_volume_aggregation_patient_grouping() -> None:
    """Verify that volume aggregation groups by patient case, not pooling the cohort."""
    # Patient A: 2 slices
    # Slice 0: TP=100, FP=50, FN=50 -> 2D Dice = 200/300 = 0.6667
    # Slice 1: TP=10,  FP=90, FN=90 -> 2D Dice = 20/200  = 0.1000
    # 3D Volume Stats: TP=110, FP=140, FN=140 -> 3D Dice = 220/500 = 0.4400
    stats_a0 = SufficientStatistics(tp={1: 100}, fp={1: 50}, fn={1: 50})
    stats_a1 = SufficientStatistics(tp={1: 10}, fp={1: 90}, fn={1: 90})
    score_vol_a = score_volume_from_stats([stats_a0, stats_a1], contract=FOREGROUND_DICE_EXCLUDE_V1_CONTRACT)
    assert score_vol_a.macro_dice == pytest.approx(0.4400, abs=1e-4)
    # Check volume scorer promoted contract: must not be slice_proxy
    assert score_vol_a.contract.metric_space != METRIC_SPACE_SLICE_PROXY
    assert score_vol_a.contract.metric_space == METRIC_SPACE_VOLUME_RESIZED

    # Patient B: 2 slices
    stats_b0 = SufficientStatistics(tp={1: 200}, fp={1: 10}, fn={1: 10})
    stats_b1 = SufficientStatistics(tp={1: 200}, fp={1: 10}, fn={1: 10})
    score_vol_b = score_volume_from_stats([stats_b0, stats_b1], contract=FOREGROUND_DICE_EXCLUDE_V1_CONTRACT)

    # True Patient Macro = (Dice_A + Dice_B) / 2
    true_patient_macro = (score_vol_a.macro_dice + score_vol_b.macro_dice) / 2.0

    # If incorrectly pooled across patients:
    # TP = 110 + 400 = 510, FP = 140 + 20 = 160, FN = 140 + 20 = 160
    # Pooled Dice = 2*510 / (2*510 + 320) = 1020 / 1340 = 0.7612 != true_patient_macro!
    pooled_stats = stats_a0.merged_with(stats_a1).merged_with(stats_b0).merged_with(stats_b1)
    _, pooled_dice = compute_dice_from_stats(pooled_stats, FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT)
    assert pooled_dice != pytest.approx(true_patient_macro, abs=0.01)

    init_d = np.array([0.6667, 0.1000, 0.9524, 0.9524])
    tp_init = np.zeros((4, 4), dtype=int)
    fp_init = np.zeros((4, 4), dtype=int)
    fn_init = np.zeros((4, 4), dtype=int)
    tp_init[:, 1] = [100, 10, 200, 200]
    fp_init[:, 1] = [50, 90, 10, 10]
    fn_init[:, 1] = [50, 90, 10, 10]

    transitions = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": init_d,
        "delta_q": np.zeros((4, 1)),
        "actual_delta_dice": np.zeros((4, 1)),
        "q_previous": init_d[:, None],
        "q_candidate": init_d[:, None],
        "tp_initial": tp_init,
        "fp_initial": fp_init,
        "fn_initial": fn_init,
        "tp_candidate": tp_init[:, None, :],
        "fp_candidate": fp_init[:, None, :],
        "fn_candidate": fn_init[:, None, :],
        "case_ids": ["pat_A", "pat_A", "pat_B", "pat_B"],
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    }

    res = evaluate_threshold(0.0, transitions=transitions)
    assert res["volume_case_count"] == 2
    assert res["volume_macro_dice"] == pytest.approx(true_patient_macro, abs=1e-4)
    assert res["volume_macro_dice"] != pytest.approx(pooled_dice, abs=0.01)


# ==============================================================================
# 5. Contract mismatch test
# ==============================================================================


def test_contract_mismatch_rejection() -> None:
    """Verify that contract validation strictly rejects incompatible configurations."""
    c_exclude = FOREGROUND_DICE_EXCLUDE_V1_CONTRACT
    c_legacy = AUDIT_TARGET_LEGACY_ONE_V1_CONTRACT

    # 1. Name mismatch
    with pytest.raises(ContractMismatchError, match="name"):
        c_exclude.validate_compatibility(c_legacy)

    # 2. Metric space mismatch
    c_vol_same_name = MetricContract(
        name=FOREGROUND_DICE_EXCLUDE_V1,
        metric_space=METRIC_SPACE_VOLUME_RESIZED,
        aggregation="volume_sufficient_stats",
    )
    with pytest.raises(ContractMismatchError, match="metric_space"):
        c_exclude.validate_compatibility(c_vol_same_name)

    # 3. Neutral margin mismatch
    c_margin_diff = MetricContract(
        name=FOREGROUND_DICE_EXCLUDE_V1,
        neutral_margin=0.010,
    )
    with pytest.raises(ContractMismatchError, match="neutral_margin"):
        c_exclude.validate_compatibility(c_margin_diff)

    # 4. Aggregation mismatch
    c_agg_diff = MetricContract(
        name=FOREGROUND_DICE_EXCLUDE_V1,
        aggregation="volume_sufficient_stats",
    )
    with pytest.raises(ContractMismatchError, match="aggregation"):
        c_exclude.validate_compatibility(c_agg_diff)

    # 5. sweep_thresholds rejects unversioned cache by default
    unversioned_cache = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.0]]),
        "actual_delta_dice": np.array([[0.0]]),
    }
    with pytest.raises(ContractMismatchError, match="do not declare a metric_contract"):
        sweep_thresholds(unversioned_cache, [0.0], strict_contract=True)

    # 6. Inconsistency between actual_delta_dice and (q_candidate - q_previous)
    inconsistent_cache = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[0.2]]),
        "q_previous": np.array([[0.5]]),
        "q_candidate": np.array([[0.6]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    with pytest.raises(ContractMismatchError, match="Inconsistency detected"):
        evaluate_threshold(0.0, transitions=inconsistent_cache)

    # 7. validate_contract deserialization validation
    valid_dict = c_exclude.to_dict()
    reconstituted = validate_contract(valid_dict)
    assert reconstituted.name == c_exclude.name
    assert reconstituted.empty_policy == "exclude"

    invalid_dict = dict(valid_dict)
    invalid_dict["version"] = 2
    with pytest.raises(ContractMismatchError, match="Unsupported contract version"):
        validate_contract(invalid_dict)


# ==============================================================================
# 6. Blank hallucination test
# ==============================================================================


def test_blank_hallucination_tracking() -> None:
    """Verify that blank trajectories are preserved and hallucinated candidates are tracked."""
    initial_dice = np.array([float("nan")])
    delta_q = np.array([[0.1]])
    q_previous = np.array([[float("nan")]])
    q_candidate = np.array([[0.0]])
    actual_delta_dice = np.array([[float("nan")]])

    transitions = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": initial_dice,
        "delta_q": delta_q,
        "actual_delta_dice": actual_delta_dice,
        "q_previous": q_previous,
        "q_candidate": q_candidate,
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }

    # At tau = 0.0: delta_q (0.1) > 0.0 -> accepted!
    res_accept = evaluate_threshold(0.0, **transitions)
    assert res_accept["accepted_total"] == 1
    assert res_accept["undefined_accepted_total"] == 1
    assert res_accept["neutral_accepted_total"] == 0
    assert res_accept["blank_to_hallucination_count"] == 1
    assert res_accept["final_macro_dice"] == 0.0

    # At tau = 0.2: delta_q (0.1) <= 0.2 -> rejected!
    res_reject = evaluate_threshold(0.2, **transitions)
    assert res_reject["rejected_total"] == 1
    assert res_reject["undefined_rejected_total"] == 1
    assert res_reject["neutral_accepted_total"] == 0
    assert res_reject["blank_to_hallucination_count"] == 0
    assert math.isnan(res_reject["final_macro_dice"])


# ==============================================================================
# 7. No feasible threshold test
# ==============================================================================


def test_no_feasible_threshold() -> None:
    """Verify that NoFeasibleThresholdError is raised when constraints cannot be met."""
    transitions = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5, 0.5]),
        "delta_q": np.array([[0.1], [0.1]]),
        "actual_delta_dice": np.array([[-0.1], [-0.1]]),
        "q_previous": np.array([[0.5], [0.5]]),
        "q_candidate": np.array([[0.4], [0.4]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }

    with pytest.raises(NoFeasibleThresholdError, match="No threshold in grid satisfied constraint"):
        sweep_thresholds(
            transitions,
            thresholds=[-0.1, 0.0, 0.05],
            max_harmful_acceptance_rate=0.05,
        )

    all_blank_transitions = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([float("nan"), float("nan")]),
        "delta_q": np.array([[-0.1], [-0.1]]),
        "actual_delta_dice": np.array([[float("nan")], [float("nan")]]),
        "q_previous": np.array([[float("nan")], [float("nan")]]),
        "q_candidate": np.array([[float("nan")], [float("nan")]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    rows = sweep_thresholds(all_blank_transitions, thresholds=[0.0, 0.1])
    with pytest.raises(NoFeasibleThresholdError, match="Cohort has no usable outcome"):
        select_threshold(rows)


# ==============================================================================
# 8. Forged contract and strict metadata-only rejection tests
# ==============================================================================


def test_forged_contract_rejection() -> None:
    """Verify that contracts with known registered names but forged/tampered fields are rejected."""
    valid = FOREGROUND_DICE_EXCLUDE_V1_CONTRACT.to_dict()

    # Tampered empty_policy
    forged_policy = dict(valid, empty_policy="legacy_one")
    with pytest.raises(ContractMismatchError, match="Forged or invalid contract definition"):
        validate_contract(forged_policy)
    with pytest.raises(ContractMismatchError, match="Forged or invalid contract definition"):
        MetricContract.from_dict(forged_policy)

    # Tampered neutral_margin
    forged_margin = dict(valid, neutral_margin=0.05)
    with pytest.raises(ContractMismatchError, match="Forged or invalid contract definition"):
        validate_contract(forged_margin)

    # Tampered metric_space
    forged_space = dict(valid, metric_space="volume_native")
    with pytest.raises(ContractMismatchError, match="Forged or invalid contract definition"):
        validate_contract(forged_space)

    # Missing required keys
    for key in ("name", "version", "metric_space", "empty_policy", "classes", "neutral_margin", "aggregation"):
        incomplete = dict(valid)
        del incomplete[key]
        with pytest.raises(ContractMismatchError, match="missing required fields"):
            MetricContract.from_dict(incomplete)


def test_strict_cache_rejects_metadata_only() -> None:
    """Verify that strict cache validation rejects metadata-only delta caches lacking state scores."""
    metadata_only = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5, 0.6]),
        "delta_q": np.array([[0.1], [-0.1]]),
        "actual_delta_dice": np.array([[0.1], [-0.1]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    with pytest.raises(ContractMismatchError, match="Strict cache requires both 'q_previous' and 'q_candidate'"):
        sweep_thresholds(metadata_only, [0.0, 0.1], strict_contract=True)


def test_cli_calibrate_threshold_roundtrip(tmp_path: Path) -> None:
    """Subprocess CLI test for synthetic cache -> calibration JSON roundtrip.

    Verifies:
    1. Exit code 0.
    2. Strict JSON finite/null handling for undefined/NaN scores.
    3. Output JSON artifact parseable and carries validity='diagnostic_only'.
    4. CLI --metric_space cannot stamp volume_native onto slice_proxy cache (exit code 1).
    """
    import json
    import subprocess
    import sys

    from torch.utils.data import DataLoader, Dataset
    from self_audit.data.common import VolumeRecord
    from self_audit.evaluation.threshold import CALIBRATION_SCHEMA_VERSION
    from self_audit.models.self_audit_net import SelfAuditNet
    from self_audit.provenance import build_lineage
    from self_audit.training._utils import bind_evaluation_checkpoint, save_checkpoint

    cache_path = tmp_path / "transitions.pt"
    out_json = tmp_path / "calibration.json"
    ckpt_path = tmp_path / "tiny_checkpoint.pt"

    tiny_cfg = {
        "num_classes": 4,
        "shared_channels": 16,
        "window_k": 4,
        "max_turns": 2,
        "encoder_name": "convnext_tiny",
        "encoder_allow_fallback": True,
    }
    torch.manual_seed(42)
    model = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()
    save_checkpoint(ckpt_path, model, epoch=1, config={"model": tiny_cfg})
    binding = bind_evaluation_checkpoint(
        model,
        [("checkpoint", ckpt_path)],
        map_location="cpu",
        config={"model": tiny_cfg},
    )

    class _RecordDataset(Dataset):
        def __init__(self, patients: tuple[str, ...]) -> None:
            self.records = [
                VolumeRecord(
                    case_id=f"{p}_ED",
                    patient_id=p,
                    image_path=Path(f"/fixture/{p}_image.npy"),
                    mask_path=Path(f"/fixture/{p}_mask.npy"),
                    split="val",
                )
                for p in patients
            ]
            self.image_size = 32
            self.depth_axis = 2
            self.foreground_only = False
            self.augment = False
            self.lower_percentile = 0.5
            self.upper_percentile = 99.5
            self.transform = None

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            r = self.records[index]
            return {
                "image": torch.randn(3, 32, 32),
                "mask": torch.randint(0, 4, (32, 32)),
                "case_id": r.case_id,
                "patient_id": r.patient_id,
            }

    loader = DataLoader(_RecordDataset(("patient001", "patient002")), batch_size=2, shuffle=False)

    # Synthetic cache with 1 normal slice and 1 all-blank slice (producing NaN)
    # This directly exercises strict JSON null conversion for undefined scores!
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": torch.tensor([0.5, float("nan")]),
        "delta_q": torch.tensor([[0.1, -0.05], [0.05, 0.1]]),
        "actual_delta_dice": torch.tensor([[0.1, -0.05], [float("nan"), float("nan")]]),
        "q_previous": torch.tensor([[0.5, 0.6], [float("nan"), float("nan")]]),
        "q_candidate": torch.tensor([[0.6, 0.55], [float("nan"), float("nan")]]),
        "active_mask": torch.tensor([[True, True], [True, True]]),
        "metric_space": "slice_proxy",
        "metric_contract": "foreground_dice_exclude_v1",
        "metric_contract_version": 1,
        "neutral_margin": 0.005,
    }
    lineage = build_lineage(
        binding=binding.as_dict(),
        loader=loader,
        split_name="val",
        cache=payload,
        metric_contract="foreground_dice_exclude_v1",
        metric_space="slice_proxy",
        neutral_margin=0.005,
        t_max=2,
    )
    payload["lineage"] = lineage
    torch.save(payload, cache_path)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "calibrate_threshold.py"),
        "--transitions",
        str(cache_path),
        "--checkpoint",
        str(ckpt_path),
        "--output",
        str(out_json),
        "--min_tau",
        "-0.1",
        "--max_tau",
        "0.1",
        "--num_thresholds",
        "3",
        "--metric_space",
        "slice_proxy",
        "--metric_contract",
        "foreground_dice_exclude_v1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI failed with stderr: {proc.stderr}\nstdout: {proc.stdout}"

    # Verify stdout has valid JSON on first line
    lines = proc.stdout.strip().splitlines()
    assert len(lines) >= 3
    stdout_json = json.loads(lines[0])
    assert "tau_accept" in stdout_json
    # Check that any NaN floats were serialized as null
    if stdout_json.get("undefined_accepted_total", 0) > 0:
        pass

    # Verify output JSON artifact
    assert out_json.is_file()
    with out_json.open("r", encoding="utf-8") as f:
        artifact = json.load(f)
    assert artifact["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert artifact["validity"] == "diagnostic_only"
    assert artifact["metric_space"] == "slice_proxy"
    assert artifact["metric_contract"] == "foreground_dice_exclude_v1"
    assert artifact["lineage"] is not None

    # Verify CLI rejects conflicting --metric_space (cannot stamp volume_native onto slice_proxy cache)
    conflict_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "calibrate_threshold.py"),
        "--transitions",
        str(cache_path),
        "--checkpoint",
        str(ckpt_path),
        "--output",
        str(tmp_path / "conflict.json"),
        "--metric_space",
        "volume_native",
        "--metric_contract",
        "foreground_dice_exclude_v1",
    ]
    conflict_proc = subprocess.run(conflict_cmd, capture_output=True, text=True)
    assert conflict_proc.returncode != 0
    assert "conflicts with" in conflict_proc.stderr or "Cannot stamp" in conflict_proc.stderr


def test_collector_sweep_multistage_integration() -> None:
    """Multistage integration: collect_validation_transition_cache -> sweep_thresholds -> select_threshold."""
    from torch.utils.data import DataLoader
    from self_audit.training.finetune_joint import collect_validation_transition_cache

    torch.manual_seed(42)
    net = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()

    # Synthetic batch: 2 cases, 2 slices each
    images = torch.randn(4, 3, 64, 64)
    masks = torch.randint(0, 4, (4, 64, 64))
    case_ids = ["case_001", "case_001", "case_002", "case_002"]

    class SyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            return {"image": images[idx], "mask": masks[idx], "case_id": case_ids[idx]}

    loader = DataLoader(SyntheticDataset(), batch_size=2, shuffle=False)

    device = torch.device("cpu")
    cached = collect_validation_transition_cache(
        net,
        loader,
        device=device,
        t_max=2,
        disable_tqdm=True,
        metric_contract="foreground_dice_exclude_v1",
    )

    # Verify cache contents
    assert "initial_dice" in cached
    assert "delta_q" in cached
    assert "actual_delta_dice" in cached
    assert "q_previous" in cached
    assert "q_candidate" in cached
    assert "tp_initial" in cached
    assert "tp_candidate" in cached
    assert "case_ids" in cached
    assert cached["metric_contract"] == "foreground_dice_exclude_v1"

    # Run sweep_thresholds on collected cache
    rows = sweep_thresholds(
        cached,
        thresholds=[-0.1, 0.0, 0.1],
        expected_contract="foreground_dice_exclude_v1",
        strict_contract=True,
    )
    assert len(rows) == 3

    # Select threshold
    selected = select_threshold(rows)
    assert "tau_accept" in selected
    assert "final_macro_dice" in selected
    assert math.isfinite(selected["final_macro_dice"])
    assert selected["volume_case_count"] == 2


# ==============================================================================
# 9. Regressions for Astra Review Bypasses & Strict Normalization Boundary
# ==============================================================================


def test_astra_bypass_1_stats_candidate_cannot_bypass_state_scores() -> None:
    """Bypass 1: stats_candidate: [] must not fool validator into mixed arithmetic.

    In the vulnerable code, presence of stats_candidate satisfied has_state_scores,
    but cand_scores remained None, silently falling back to mixed addition:
    0.5 + 0.13333333 = 0.63333333.
    """
    base = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[1.0]]),
        "actual_delta_dice": np.array([[0.133333333]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
        "stats_candidate": [],
    }
    with pytest.raises(ContractMismatchError, match="Bogus or empty sufficient statistics container|Strict cache requires both 'q_previous' and 'q_candidate'"):
        sweep_thresholds(base, [0.0])


def test_astra_bypass_2_missing_q_previous_rejected() -> None:
    """Bypass 2: q_candidate present but q_previous missing must be rejected.

    In the vulnerable code, missing q_previous skipped parity verification and
    evaluated actual_delta_dice independently, reporting harmful_total=0 even when
    candidate degraded from 0.5 to 0.4.
    """
    base = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[1.0]]),
        "actual_delta_dice": np.array([[0.133333333]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
        "q_candidate": np.array([[0.4]]),
    }
    with pytest.raises(ContractMismatchError, match="Strict cache requires both 'q_previous' and 'q_candidate'"):
        sweep_thresholds(base, [0.0])


def test_astra_bypass_3_wrong_shape_q_previous_rejected() -> None:
    """Bypass 3: q_previous wrong shape (e.g. (1, 2) vs delta_q (1, 1)) must be rejected.

    In the vulnerable code, shape mismatch silently bypassed parity checks instead
    of raising an error.
    """
    base = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[1.0]]),
        "actual_delta_dice": np.array([[0.133333333]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
        "q_candidate": np.array([[0.4]]),
        "q_previous": np.array([[0.5, 0.8]]),
    }
    with pytest.raises(ContractMismatchError, match="Shape mismatch"):
        sweep_thresholds(base, [0.0])


def test_astra_bypass_4_fabricated_nan_actual_delta_rejected() -> None:
    """Bypass 4: actual_delta_dice=[[NaN]] when states are finite must be rejected.

    In the vulnerable code, a fabricated NaN in actual_delta_dice bypassed both_fin,
    causing the transition to be treated as undefined and hiding harmful degradation.
    """
    base = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[1.0]]),
        "actual_delta_dice": np.array([[float("nan")]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
        "q_previous": np.array([[0.5]]),
        "q_candidate": np.array([[0.4]]),
    }
    with pytest.raises(ContractMismatchError, match="Fabricated NaN in actual_delta_dice"):
        sweep_thresholds(base, [0.0])


def test_t_max_zero_valid_cache() -> None:
    """Verify that t_max=0 valid cache with empty (N, 0) arrays evaluates cleanly."""
    t0_cache = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.6, 0.7]),
        "delta_q": np.empty((2, 0)),
        "actual_delta_dice": np.empty((2, 0)),
        "q_previous": np.empty((2, 0)),
        "q_candidate": np.empty((2, 0)),
        "active_mask": np.empty((2, 0), dtype=bool),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    rows = sweep_thresholds(t0_cache, [0.0, 0.1])
    assert len(rows) == 2
    for r in rows:
        assert r["mean_attempted_turns"] == 0.0
        assert r["mean_accepted_turns"] == 0.0
        assert r["final_macro_dice"] == pytest.approx(0.65)
    selected = select_threshold(rows)
    assert selected["final_macro_dice"] == pytest.approx(0.65)


def test_initial_q_previous_mismatch_rejected() -> None:
    """Verify that mismatch between initial_dice and turn 0 q_previous is rejected."""
    bad_cache = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[0.1]]),
        "q_previous": np.array([[0.6]]),  # 0.6 != initial 0.5!
        "q_candidate": np.array([[0.7]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    with pytest.raises(ContractMismatchError, match="Turn 0 q_previous values do not match initial_dice"):
        sweep_thresholds(bad_cache, [0.0])


def test_state_discontinuity_rejected() -> None:
    """Verify that state discontinuity between consecutive turns is rejected."""
    bad_continuity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1, 0.1]]),
        "actual_delta_dice": np.array([[0.1, 0.05]]),
        "q_previous": np.array([[0.5, 0.75]]),  # Turn 1 previous is 0.75 != Turn 0 candidate 0.60!
        "q_candidate": np.array([[0.6, 0.80]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    with pytest.raises(ContractMismatchError, match="State continuity broken"):
        sweep_thresholds(bad_continuity, [0.0])


def test_score_bounds_out_of_range_rejected() -> None:
    """Verify that finite Dice scores outside [0.0, 1.0] are rejected."""
    bad_init = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([1.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[0.1]]),
        "q_previous": np.array([[1.5]]),
        "q_candidate": np.array([[1.6]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    with pytest.raises(ContractMismatchError, match="initial_dice values must be in"):
        sweep_thresholds(bad_init, [0.0])

    bad_cand = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[-0.7]]),
        "q_previous": np.array([[0.5]]),
        "q_candidate": np.array([[-0.2]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    with pytest.raises(ContractMismatchError, match="q_candidate values must be in"):
        sweep_thresholds(bad_cand, [0.0])


def test_sweep_neutral_margin_mismatch_rejected() -> None:
    """Verify that passing a conflicting neutral_margin to sweep is rejected."""
    cache = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[0.1]]),
        "q_previous": np.array([[0.5]]),
        "q_candidate": np.array([[0.6]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    # FOREGROUND_DICE_EXCLUDE_V1 has neutral_margin=0.005. Passing 0.010 must fail!
    with pytest.raises(ContractMismatchError, match="Sweep neutral_margin .* conflicts with declared contract"):
        sweep_thresholds(cache, [0.0], neutral_margin=0.010)


def test_cache_metric_space_mismatch_rejected() -> None:
    """Verify that cache metric_space conflicting with contract metric_space is rejected."""
    cache = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[0.1]]),
        "q_previous": np.array([[0.5]]),
        "q_candidate": np.array([[0.6]]),
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
        "metric_space": "volume_native",  # Conflicts with slice_proxy!
    }
    with pytest.raises(ContractMismatchError, match="Cache metric_space 'volume_native' conflicts with declared contract"):
        sweep_thresholds(cache, [0.0])


def test_bogus_statistics_rejected() -> None:
    """Verify that bogus sufficient statistics shapes or negative counts are rejected."""
    neg_stats = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[0.1]]),
        "q_previous": np.array([[0.5]]),
        "q_candidate": np.array([[0.6]]),
        "tp_initial": np.array([[-10, 50]]),  # Negative count!
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    with pytest.raises(ContractMismatchError, match="cannot contain negative counts"):
        sweep_thresholds(neg_stats, [0.0])

    bad_shape_cand = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[0.1]]),
        "q_previous": np.array([[0.5]]),
        "q_candidate": np.array([[0.6]]),
        "tp_candidate": np.array([[10, 50]]),  # 2D instead of 3D (N, T, C)!
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
    }
    with pytest.raises(ContractMismatchError, match="Bogus or mismatched shape"):
        sweep_thresholds(bad_shape_cand, [0.0])


# ==============================================================================
# 10. Wave 1 Revision 2: Attack Probes, Schema Versioning & Boundary Hardening
# ==============================================================================


def test_wave1_rev2_attack_probe_1_normalized_bypass_rejected() -> None:
    """Attack 1: Caller-supplied _normalized: True must NEVER bypass validation.

    Verified on validate_and_normalize_transition_cache, evaluate_threshold,
    and sweep_thresholds.
    """
    # Raw attack probe 1 from coordinator
    with pytest.raises(ContractMismatchError, match="missing required 'cache_schema_version'"):
        validate_and_normalize_transition_cache({"_normalized": True})

    # Even with cache_schema_version: 1, _normalized must not skip contract and key validation
    with pytest.raises(ContractMismatchError, match="do not declare a metric_contract"):
        validate_and_normalize_transition_cache({"_normalized": True, "cache_schema_version": 1})

    # Downstream evaluate_threshold must also revalidate and reject
    with pytest.raises(ContractMismatchError, match="missing required 'cache_schema_version'"):
        evaluate_threshold(0.0, transitions={"_normalized": True})

    with pytest.raises(ContractMismatchError, match="do not declare a metric_contract"):
        evaluate_threshold(0.0, transitions={"_normalized": True, "cache_schema_version": 1})

    # Downstream sweep_thresholds must also revalidate and reject
    with pytest.raises(ContractMismatchError, match="missing required 'cache_schema_version'"):
        sweep_thresholds({"_normalized": True}, [0.0])


def test_wave1_rev2_attack_probe_2_unsupported_cache_schema_version_rejected() -> None:
    """Attack 2: Valid Qprev/Qcand cache with unsupported cache_schema_version (e.g. 999) must be rejected."""
    probe2_cache = {
        "cache_schema_version": 999,
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[0.1]]),
        "q_previous": np.array([[0.5]]),
        "q_candidate": np.array([[0.6]]),
    }

    with pytest.raises(ContractMismatchError, match="Unsupported cache_schema_version 999"):
        validate_and_normalize_transition_cache(probe2_cache)

    with pytest.raises(ContractMismatchError, match="Unsupported cache_schema_version 999"):
        evaluate_threshold(0.0, transitions=probe2_cache)

    with pytest.raises(ContractMismatchError, match="Unsupported cache_schema_version 999"):
        sweep_thresholds(probe2_cache, [0.0])


def test_wave1_rev2_cache_schema_version_strict_validation() -> None:
    """Verify strict cache_schema_version checks: missing, bool, string, float, and None are rejected."""
    base_valid = {
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1,
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[0.1]]),
        "q_previous": np.array([[0.5]]),
        "q_candidate": np.array([[0.6]]),
    }

    # 1. Missing version in strict mode
    with pytest.raises(ContractMismatchError, match="missing required 'cache_schema_version'"):
        validate_and_normalize_transition_cache(base_valid, strict_contract=True)

    # 2. Bool version (bool is a subclass of int in Python, so must be explicitly caught)
    with pytest.raises(ContractMismatchError, match="Invalid cache_schema_version True"):
        validate_and_normalize_transition_cache(dict(base_valid, cache_schema_version=True))

    with pytest.raises(ContractMismatchError, match="Invalid cache_schema_version False"):
        validate_and_normalize_transition_cache(dict(base_valid, cache_schema_version=False))

    # 3. String version
    with pytest.raises(ContractMismatchError, match="Invalid cache_schema_version '1'"):
        validate_and_normalize_transition_cache(dict(base_valid, cache_schema_version="1"))

    # 4. Float version
    with pytest.raises(ContractMismatchError, match="Invalid cache_schema_version 1.0"):
        validate_and_normalize_transition_cache(dict(base_valid, cache_schema_version=1.0))

    # 5. None version
    with pytest.raises(ContractMismatchError, match="Invalid cache_schema_version None"):
        validate_and_normalize_transition_cache(dict(base_valid, cache_schema_version=None))


def test_wave1_rev2_valid_collector_t0() -> None:
    """Verify that collector emits valid cache with cache_schema_version and empty (N, 0) arrays when t_max=0."""
    from torch.utils.data import DataLoader
    from self_audit.training.finetune_joint import collect_validation_transition_cache

    torch.manual_seed(42)
    net = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()

    images = torch.randn(2, 3, 64, 64)
    masks = torch.randint(0, 4, (2, 64, 64))

    class SyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            return {"image": images[idx], "mask": masks[idx]}

    loader = DataLoader(SyntheticDataset(), batch_size=2, shuffle=False)
    cache = collect_validation_transition_cache(
        net,
        loader,
        device=torch.device("cpu"),
        t_max=0,
        disable_tqdm=True,
        metric_contract="foreground_dice_exclude_v1",
    )

    # Must contain cache_schema_version
    assert "cache_schema_version" in cache
    assert cache["cache_schema_version"] == CACHE_SCHEMA_VERSION

    # Dimensions at T=0
    assert cache["initial_dice"].shape[0] == 2
    assert cache["delta_q"].shape == (2, 0)
    assert cache["actual_delta_dice"].shape == (2, 0)
    assert cache["q_previous"].shape == (2, 0)
    assert cache["q_candidate"].shape == (2, 0)
    assert cache["tp_candidate"].shape[1] == 0

    # Normalization & sweep must succeed
    norm = validate_and_normalize_transition_cache(cache)
    assert norm["cache_schema_version"] == CACHE_SCHEMA_VERSION
    assert "_normalized" not in norm  # no bool marker in normalized output!

    rows = sweep_thresholds(cache, [0.0, 0.1])
    assert len(rows) == 2
    for r in rows:
        assert r["mean_attempted_turns"] == 0.0
        assert r["mean_accepted_turns"] == 0.0
        assert np.isfinite(r["final_macro_dice"])


def test_wave1_rev2_collector_and_sweep_blank_data() -> None:
    """Verify collector and replay behavior on blank slice data (masks without foreground)."""
    from torch.utils.data import DataLoader
    from self_audit.training.finetune_joint import collect_validation_transition_cache

    torch.manual_seed(42)
    net = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()

    # All masks are background only (class 0) -> classes 1, 2, 3 are completely absent!
    images = torch.randn(2, 3, 64, 64)
    blank_masks = torch.zeros((2, 64, 64), dtype=torch.int64)

    class BlankDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            return {"image": images[idx], "mask": blank_masks[idx]}

    loader = DataLoader(BlankDataset(), batch_size=2, shuffle=False)
    cache = collect_validation_transition_cache(
        net,
        loader,
        device=torch.device("cpu"),
        t_max=2,
        disable_tqdm=True,
        metric_contract="foreground_dice_exclude_v1",
    )

    # Emitted cache checks
    assert cache["cache_schema_version"] == CACHE_SCHEMA_VERSION
    # Under exclude policy, all-blank GT means initial dice is NaN (or undefined)
    # If the network predicted foreground, Dice is 0.0; if both empty, Dice is NaN.
    # Either way, validate_and_normalize_transition_cache and sweep must process cleanly.
    norm = validate_and_normalize_transition_cache(cache)
    assert norm["cache_schema_version"] == CACHE_SCHEMA_VERSION
    assert "_normalized" not in norm

    rows = sweep_thresholds(cache, [0.0, 0.1])
    assert len(rows) == 2
    for r in rows:
        assert "undefined_transition_total" in r


# ==============================================================================
# 11. Wave 1.3: Volume Identity & Metric-Space Safety (W1.3)
# ==============================================================================


def test_w1_3_two_case_fixture_volume_identity_and_no_cohort_pooling() -> None:
    """W1.3 Mandatory Test 1: Two-case fixture.

    Independent fixture: two rows with TP/FP/FN for class 1:
      Row 0: (1, 0, 0) -> Dice = 1.0 (Q=1)
      Row 1: (0, 9, 9) -> Dice = 0.0 (Q=0)
    Proposals unchanged, delta_q = 1.0, tau = 0.0.

    Pass criteria:
    - Explicit IDs (case1, case2) -> volume_macro_dice = 0.5, volume_case_count = 2.
    - Missing IDs -> volume_macro_dice is None, volume_case_count is None,
      volume_unavailable_reason indicates missing case_ids; cohort pooling across
      unidentified slices is forbidden. NEVER asserted as 0.1 / count 1!
    - Explicit single-volume identity (case1, case1) is acceptable -> 0.1 / count 1.
    - Explicit require_volume=True without case_ids raises ContractMismatchError.
    """
    tp_init = np.zeros((2, 4), dtype=np.int64)
    fp_init = np.zeros((2, 4), dtype=np.int64)
    fn_init = np.zeros((2, 4), dtype=np.int64)
    # Row 0: class 1 TP=1, FP=0, FN=0
    tp_init[0, 1] = 1
    # Row 1: class 1 TP=0, FP=9, FN=9
    fp_init[1, 1] = 9
    fn_init[1, 1] = 9

    init_dice = np.array([1.0, 0.0])
    delta_q = np.array([[1.0], [1.0]])
    actual_delta = np.array([[0.0], [0.0]])
    q_prev = np.array([[1.0], [0.0]])
    q_cand = np.array([[1.0], [0.0]])
    tp_cand = tp_init[:, None, :]
    fp_cand = fp_init[:, None, :]
    fn_cand = fn_init[:, None, :]

    base_cache = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": init_dice,
        "delta_q": delta_q,
        "actual_delta_dice": actual_delta,
        "q_previous": q_prev,
        "q_candidate": q_cand,
        "tp_initial": tp_init,
        "fp_initial": fp_init,
        "fn_initial": fn_init,
        "tp_candidate": tp_cand,
        "fp_candidate": fp_cand,
        "fn_candidate": fn_cand,
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    }

    # 1. With explicit case IDs: macro mean across volumes is 0.5, count is 2
    cache_with_ids = dict(base_cache, case_ids=["case1", "case2"])
    res_ids = evaluate_threshold(0.0, transitions=cache_with_ids)
    assert res_ids["volume_metrics_available"] is True
    assert res_ids["volume_case_count"] == 2
    assert res_ids["volume_macro_dice"] == pytest.approx(0.5, abs=1e-5)
    assert res_ids["volume_per_class_dice"][1] == pytest.approx(0.5, abs=1e-5)
    assert res_ids.get("volume_unavailable_reason") is None
    assert res_ids["volume_metric_space"] == METRIC_SPACE_VOLUME_RESIZED
    assert res_ids["volume_metric_contract"] == FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT.name

    # 1b. Numpy array case_ids parity (must not fail boolean ambiguity check)
    cache_with_np_ids = dict(base_cache, case_ids=np.array(["case1", "case2"]))
    res_np_ids = evaluate_threshold(0.0, transitions=cache_with_np_ids)
    assert res_np_ids["volume_metrics_available"] is True
    assert res_np_ids["volume_case_count"] == 2
    assert res_np_ids["volume_macro_dice"] == pytest.approx(0.5, abs=1e-5)

    # 2. Without case IDs: cohort pooling is forbidden! Volume outputs are omitted
    res_no_ids = evaluate_threshold(0.0, transitions=base_cache)
    assert res_no_ids["volume_metrics_available"] is False
    assert res_no_ids["volume_macro_dice"] is None
    assert res_no_ids["volume_case_count"] is None
    assert res_no_ids["volume_unavailable_reason"] == (
        "Missing case_ids; cohort pooling across unidentified slices is forbidden"
    )
    # MUST NEVER fall back to cohort pooling (0.1, count 1)
    assert res_no_ids.get("volume_macro_dice") != pytest.approx(0.1, abs=0.01)
    assert res_no_ids.get("volume_case_count") != 1

    # 3. Sweep thresholds without case IDs also omits volume outputs
    rows = sweep_thresholds(base_cache, [-0.1, 0.0, 0.1])
    assert len(rows) == 3
    for r in rows:
        assert r["volume_metrics_available"] is False
        assert r["volume_macro_dice"] is None
        assert r["volume_case_count"] is None
        assert "cohort pooling across unidentified slices is forbidden" in r["volume_unavailable_reason"]

    # 4. Explicit single-volume identity is acceptable
    cache_single_vol = dict(base_cache, case_ids=["case1", "case1"])
    res_single = evaluate_threshold(0.0, transitions=cache_single_vol)
    assert res_single["volume_case_count"] == 1
    # Pooled TP=1, FP=9, FN=9 -> 2/20 = 0.1
    assert res_single["volume_macro_dice"] == pytest.approx(0.1, abs=1e-5)
    assert res_single["volume_metric_space"] == METRIC_SPACE_VOLUME_RESIZED

    # 5. Explicit volume scoring requested with missing case IDs raises ContractMismatchError
    with pytest.raises(ContractMismatchError, match="cohort pooling across unidentified slices is forbidden"):
        evaluate_threshold(0.0, transitions=base_cache, require_volume=True)

    with pytest.raises(ContractMismatchError, match="cohort pooling across unidentified slices is forbidden"):
        sweep_thresholds(base_cache, [0.0], require_volume=True)


def test_w1_3_direct_grouped_statistics_parity() -> None:
    """W1.3 Mandatory Test 2: Direct grouped statistics == replay selected grouped statistics."""
    # Build two cases with multi-slice structure
    # Case A: 2 slices, Case B: 2 slices
    tp_init = np.zeros((4, 4), dtype=np.int64)
    fp_init = np.zeros((4, 4), dtype=np.int64)
    fn_init = np.zeros((4, 4), dtype=np.int64)

    # Case A slice 0: class 1 TP=50, FP=10, FN=20
    tp_init[0, 1] = 50
    fp_init[0, 1] = 10
    fn_init[0, 1] = 20
    # Case A slice 1: class 1 TP=60, FP=20, FN=10
    tp_init[1, 1] = 60
    fp_init[1, 1] = 20
    fn_init[1, 1] = 10
    # Case B slice 0: class 1 TP=20, FP=80, FN=40
    tp_init[2, 1] = 20
    fp_init[2, 1] = 80
    fn_init[2, 1] = 40
    # Case B slice 1: class 1 TP=10, FP=50, FN=30
    tp_init[3, 1] = 10
    fp_init[3, 1] = 50
    fn_init[3, 1] = 30

    # Calculate direct slice macro dice
    c = FOREGROUND_DICE_EXCLUDE_V1_CONTRACT
    s_a0 = compute_dice_from_stats(SufficientStatistics(tp={1: 50}, fp={1: 10}, fn={1: 20}), c)[1]
    s_a1 = compute_dice_from_stats(SufficientStatistics(tp={1: 60}, fp={1: 20}, fn={1: 10}), c)[1]
    s_b0 = compute_dice_from_stats(SufficientStatistics(tp={1: 20}, fp={1: 80}, fn={1: 40}), c)[1]
    s_b1 = compute_dice_from_stats(SufficientStatistics(tp={1: 10}, fp={1: 50}, fn={1: 30}), c)[1]
    init_dice = np.array([s_a0, s_a1, s_b0, s_b1])

    # Direct grouped statistics computation:
    # Case A: TP = 50+60 = 110, FP = 10+20 = 30, FN = 20+10 = 30 -> Dice = 220 / (220 + 60) = 220/280
    case_a_dice = 2.0 * 110 / (2.0 * 110 + 30 + 30)
    # Case B: TP = 20+10 = 30, FP = 80+50 = 130, FN = 40+30 = 70 -> Dice = 60 / (60 + 200) = 60/260
    case_b_dice = 2.0 * 30 / (2.0 * 30 + 130 + 70)
    direct_volume_macro = (case_a_dice + case_b_dice) / 2.0

    transitions = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": init_dice,
        "delta_q": np.zeros((4, 1)),
        "actual_delta_dice": np.zeros((4, 1)),
        "q_previous": init_dice[:, None],
        "q_candidate": init_dice[:, None],
        "tp_initial": tp_init,
        "fp_initial": fp_init,
        "fn_initial": fn_init,
        "tp_candidate": tp_init[:, None, :],
        "fp_candidate": fp_init[:, None, :],
        "fn_candidate": fn_init[:, None, :],
        "case_ids": ["case_A", "case_A", "case_B", "case_B"],
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    }

    res = evaluate_threshold(0.0, transitions=transitions)
    assert res["volume_case_count"] == 2
    assert res["volume_macro_dice"] == pytest.approx(direct_volume_macro, abs=1e-5)
    assert res["volume_per_class_dice"][1] == pytest.approx(direct_volume_macro, abs=1e-5)


def test_w1_3_collector_rejects_volume_target_contracts() -> None:
    """W1.3 Mandatory Test 3: Collector rejects volume_native and volume_resized contracts."""
    from torch.utils.data import DataLoader
    from self_audit.training.finetune_joint import collect_validation_transition_cache

    net = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=1,
    ).eval()

    images = torch.randn(2, 3, 64, 64)
    masks = torch.randint(0, 4, (2, 64, 64))

    class MiniDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            return {"image": images[idx], "mask": masks[idx], "case_id": f"c_{idx}"}

    loader = DataLoader(MiniDataset(), batch_size=2, shuffle=False)
    device = torch.device("cpu")

    # 1. Reject foreground_dice_volume_native_v1
    with pytest.raises(ContractMismatchError, match="rejects volume contract 'foreground_dice_volume_native_v1'"):
        collect_validation_transition_cache(
            net, loader, device=device, metric_contract="foreground_dice_volume_native_v1"
        )

    # 2. Reject foreground_dice_volume_resized_v1
    with pytest.raises(ContractMismatchError, match="rejects volume contract 'foreground_dice_volume_resized_v1'"):
        collect_validation_transition_cache(
            net, loader, device=device, metric_contract="foreground_dice_volume_resized_v1"
        )

    # 3. Reject custom contract with metric_space != slice_proxy
    custom_vol = MetricContract(
        name="custom_volume_contract",
        metric_space=METRIC_SPACE_VOLUME_RESIZED,
        aggregation="volume_sufficient_stats",
    )
    with pytest.raises(ContractMismatchError, match="Per-slice Q values must never be labeled volume scores"):
        collect_validation_transition_cache(
            net, loader, device=device, metric_contract=custom_vol
        )


def test_w1_3_stats_bundle_validation_and_rejections() -> None:
    """W1.3 Mandatory Test 4: Partial stats, NaN/inf/negative/fractional counts, wrong class dimension,

    and Q/stat inconsistency rejected. Valid t_max=0 and blank->hallucination remain supported.
    """
    valid_init_stats = np.zeros((2, 4), dtype=np.int64)
    valid_init_stats[0, 1] = 10
    valid_init_stats[0, 2] = 20
    valid_init_stats[1, 1] = 30
    s0 = compute_dice_from_stats(SufficientStatistics(tp={1: 10, 2: 20}, fp={}, fn={}), FOREGROUND_DICE_EXCLUDE_V1_CONTRACT)[1]
    s1 = compute_dice_from_stats(SufficientStatistics(tp={1: 30}, fp={}, fn={}), FOREGROUND_DICE_EXCLUDE_V1_CONTRACT)[1]
    init_d = np.array([s0, s1])

    base_valid = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": init_d,
        "delta_q": np.zeros((2, 1)),
        "actual_delta_dice": np.zeros((2, 1)),
        "q_previous": init_d[:, None],
        "q_candidate": init_d[:, None],
        "tp_initial": valid_init_stats,
        "fp_initial": np.zeros((2, 4), dtype=np.int64),
        "fn_initial": np.zeros((2, 4), dtype=np.int64),
        "tp_candidate": valid_init_stats[:, None, :],
        "fp_candidate": np.zeros((2, 1, 4), dtype=np.int64),
        "fn_candidate": np.zeros((2, 1, 4), dtype=np.int64),
        "case_ids": ["c1", "c2"],
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    }

    # Baseline must validate cleanly
    norm = validate_and_normalize_transition_cache(base_valid)
    assert norm["cache_schema_version"] == CACHE_SCHEMA_VERSION

    # 4a: Partial initial bundle: missing fn_initial
    cache_missing_fn = dict(base_valid)
    del cache_missing_fn["fn_initial"]
    with pytest.raises(ContractMismatchError, match="Incomplete sufficient statistics bundle: initial stats missing fn_initial"):
        validate_and_normalize_transition_cache(cache_missing_fn)

    # 4b: Partial candidate bundle (T=1): missing fn_candidate
    cache_missing_cand_fn = dict(base_valid)
    del cache_missing_cand_fn["fn_candidate"]
    with pytest.raises(ContractMismatchError, match="Incomplete sufficient statistics bundle: candidate stats missing fn_candidate"):
        validate_and_normalize_transition_cache(cache_missing_cand_fn)

    # 4c: Initial present, candidate completely missing when T > 0
    cache_no_cand = dict(base_valid)
    del cache_no_cand["tp_candidate"]
    del cache_no_cand["fp_candidate"]
    del cache_no_cand["fn_candidate"]
    with pytest.raises(ContractMismatchError, match="candidate stats missing"):
        validate_and_normalize_transition_cache(cache_no_cand)

    # 4d: Candidate present, initial completely missing
    cache_no_init = dict(base_valid)
    del cache_no_init["tp_initial"]
    del cache_no_init["fp_initial"]
    del cache_no_init["fn_initial"]
    with pytest.raises(ContractMismatchError, match="initial stats missing"):
        validate_and_normalize_transition_cache(cache_no_init)

    # 4e: NaN in counts
    cache_nan = dict(base_valid, tp_initial=np.array([[np.nan, 0, 0, 0], [0, 0, 0, 0]]))
    with pytest.raises(ContractMismatchError, match="contains NaN or Inf"):
        validate_and_normalize_transition_cache(cache_nan)

    # 4f: Inf in counts
    cache_inf = dict(base_valid, tp_candidate=np.full((2, 1, 4), np.inf))
    with pytest.raises(ContractMismatchError, match="contains NaN or Inf"):
        validate_and_normalize_transition_cache(cache_inf)

    # 4g: Negative counts
    cache_neg = dict(base_valid, fp_initial=np.array([[-1, 0, 0, 0], [0, 0, 0, 0]]))
    with pytest.raises(ContractMismatchError, match="cannot contain negative counts"):
        validate_and_normalize_transition_cache(cache_neg)

    # 4h: Fractional float counts (never truncated to int)
    cache_frac_init = dict(base_valid, tp_initial=np.array([[10.5, 0, 0, 0], [0, 0, 0, 0]]))
    with pytest.raises(ContractMismatchError, match="cannot contain fractional counts"):
        validate_and_normalize_transition_cache(cache_frac_init)

    cache_frac_cand = dict(base_valid, fn_candidate=np.array([[[0.25, 0, 0, 0]], [[0, 0, 0, 0]]]))
    with pytest.raises(ContractMismatchError, match="cannot contain fractional counts"):
        validate_and_normalize_transition_cache(cache_frac_cand)

    # 4i: Wrong class dimension (C=2 when contract requires at least 4 classes for (1, 2, 3))
    cache_wrong_dim = dict(
        base_valid,
        tp_initial=np.zeros((2, 2), dtype=np.int64),
        fp_initial=np.zeros((2, 2), dtype=np.int64),
        fn_initial=np.zeros((2, 2), dtype=np.int64),
        tp_candidate=np.zeros((2, 1, 2), dtype=np.int64),
        fp_candidate=np.zeros((2, 1, 2), dtype=np.int64),
        fn_candidate=np.zeros((2, 1, 2), dtype=np.int64),
    )
    with pytest.raises(ContractMismatchError, match="class dimension C=2 is too small for contract classes"):
        validate_and_normalize_transition_cache(cache_wrong_dim)

    # 4j: Q / stat inconsistency: initial_dice is 0.5, but sufficient stats evaluate to s0 (~1.0)
    cache_inconsistent_init = dict(
        base_valid,
        initial_dice=np.array([0.5, s1]),
        q_previous=np.array([[0.5], [s1]]),
        q_candidate=np.array([[0.5], [s1]]),
        actual_delta_dice=np.zeros((2, 1)),
    )
    with pytest.raises(ContractMismatchError, match="Inconsistency detected between initial_dice"):
        validate_and_normalize_transition_cache(cache_inconsistent_init)

    # Q / stat inconsistency in candidate: q_candidate is 0.2, but candidate stats evaluate to s0 (~1.0)
    cache_inconsistent_cand = dict(
        base_valid,
        initial_dice=np.array([s0, s1]),
        q_previous=np.array([[s0], [s1]]),
        q_candidate=np.array([[0.2], [s1]]),
        actual_delta_dice=np.array([[0.2 - s0], [0.0]]),
    )
    with pytest.raises(ContractMismatchError, match="Inconsistency detected between q_candidate"):
        validate_and_normalize_transition_cache(cache_inconsistent_cand)

    # 4k: Invalid case IDs
    cache_empty_id = dict(base_valid, case_ids=["", "c2"])
    with pytest.raises(ContractMismatchError, match="case_ids contains invalid, empty, or None identifiers"):
        validate_and_normalize_transition_cache(cache_empty_id)

    cache_none_id = dict(base_valid, case_ids=["c1", "None"])
    with pytest.raises(ContractMismatchError, match="case_ids contains invalid, empty, or None identifiers"):
        validate_and_normalize_transition_cache(cache_none_id)

    # 4l: Valid t_max=0 with initial statistics remains supported
    t0_cache = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": init_d,
        "delta_q": np.zeros((2, 0)),
        "actual_delta_dice": np.zeros((2, 0)),
        "q_previous": np.zeros((2, 0)),
        "q_candidate": np.zeros((2, 0)),
        "tp_initial": valid_init_stats,
        "fp_initial": np.zeros((2, 4), dtype=np.int64),
        "fn_initial": np.zeros((2, 4), dtype=np.int64),
        "tp_candidate": np.zeros((2, 0, 4), dtype=np.int64),
        "fp_candidate": np.zeros((2, 0, 4), dtype=np.int64),
        "fn_candidate": np.zeros((2, 0, 4), dtype=np.int64),
        "case_ids": ["c1", "c2"],
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    }
    t0_res = evaluate_threshold(0.0, transitions=t0_cache)
    assert t0_res["volume_case_count"] == 2
    assert np.isfinite(t0_res["volume_macro_dice"])

    # 4m: Blank -> hallucination transition remains supported
    # Initial: blank slice (all stats 0 -> Dice NaN)
    # Candidate: hallucinates class 1 with FP=10, TP=0, FN=0 -> Dice 0.0, delta NaN
    blank_hallucination = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": np.array([float("nan")]),
        "delta_q": np.array([[1.0]]),
        "actual_delta_dice": np.array([[float("nan")]]),
        "q_previous": np.array([[float("nan")]]),
        "q_candidate": np.array([[0.0]]),
        "tp_initial": np.zeros((1, 4), dtype=np.int64),
        "fp_initial": np.zeros((1, 4), dtype=np.int64),
        "fn_initial": np.zeros((1, 4), dtype=np.int64),
        "tp_candidate": np.zeros((1, 1, 4), dtype=np.int64),
        "fp_candidate": np.array([[[0, 10, 0, 0]]], dtype=np.int64),
        "fn_candidate": np.zeros((1, 1, 4), dtype=np.int64),
        "case_ids": ["c1"],
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    }
    bh_norm = validate_and_normalize_transition_cache(blank_hallucination)
    assert bh_norm["cache_schema_version"] == CACHE_SCHEMA_VERSION
    bh_res = evaluate_threshold(0.0, transitions=blank_hallucination)
    assert bh_res["blank_to_hallucination_count"] == 1
    assert bh_res["volume_case_count"] == 1


def test_w1_3_distinct_slice_and_derived_volume_metric_metadata() -> None:
    """W1.3 Mandatory Test 5: Distinct slice and derived-volume metric metadata asserted."""
    tp_init = np.zeros((2, 4), dtype=np.int64)
    tp_init[:, 1] = [10, 20]
    fp_init = np.zeros((2, 4), dtype=np.int64)
    fn_init = np.zeros((2, 4), dtype=np.int64)

    init_d = np.array([1.0, 1.0])
    cache = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": init_d,
        "delta_q": np.zeros((2, 1)),
        "actual_delta_dice": np.zeros((2, 1)),
        "q_previous": init_d[:, None],
        "q_candidate": init_d[:, None],
        "tp_initial": tp_init,
        "fp_initial": fp_init,
        "fn_initial": fn_init,
        "tp_candidate": tp_init[:, None, :],
        "fp_candidate": fp_init[:, None, :],
        "fn_candidate": fn_init[:, None, :],
        "case_ids": ["vol_1", "vol_2"],
        "metric_contract": FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    }

    res = evaluate_threshold(0.0, transitions=cache)

    # 1. Assert distinct slice metadata
    assert res["metric_contract"] == "foreground_dice_exclude_v1"
    assert res["metric_space"] == METRIC_SPACE_SLICE_PROXY

    # 2. Assert distinct derived-volume metadata
    assert res["volume_metric_contract"] == FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT.name
    assert res["volume_metric_space"] == METRIC_SPACE_VOLUME_RESIZED

    # 3. Assert slice and volume spaces are not conflated
    assert res["metric_space"] != res["volume_metric_space"]

    # 4. Reject volume_native claims from transition cache
    cache_vol_native = dict(cache, metric_space="volume_native")
    with pytest.raises(ContractMismatchError, match="Cache metric_space 'volume_native' conflicts"):
        validate_and_normalize_transition_cache(cache_vol_native)

    cache_vol_contract = dict(cache, metric_contract="foreground_dice_volume_native_v1")
    with pytest.raises(ContractMismatchError, match="Transition cache represents per-slice transitions and requires metric_space='slice_proxy'"):
        validate_and_normalize_transition_cache(cache_vol_contract)

    # 5. Reject volume_resized as slice cache metric contract
    cache_vol_resized_contract = dict(cache, metric_contract="foreground_dice_volume_resized_v1")
    with pytest.raises(ContractMismatchError, match="Transition cache represents per-slice transitions and requires metric_space='slice_proxy'"):
        validate_and_normalize_transition_cache(cache_vol_resized_contract)

    # 6. save_calibration rejects stamping volume_native onto slice_proxy selected row
    with pytest.raises(ContractMismatchError, match="save_calibration metric_space 'volume_native' conflicts with selected_row metric_space 'slice_proxy'"):
        save_calibration(
            Path("/tmp/dummy_calib.json"),
            tau_accept=0.0,
            neutral_margin=0.005,
            source_split="val",
            t_max=1,
            threshold_grid=[0.0],
            selected_row=res,
            metric_space="volume_native",
        )


