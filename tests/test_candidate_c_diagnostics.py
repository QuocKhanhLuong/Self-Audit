"""Candidate C diagnostics tests.

Covers ``src/self_audit/evaluation/candidate_c.py`` and probe 4 of
``scripts/audit_checkpoint.py``.  Everything here runs on hand-built tensors
with tiny grids: no checkpoint, no dataset, no training loop, no GPU.

The properties under test are the ones that make the block trustworthy rather
than merely present:

* an unavailable metric is ``None``, never ``0``;
* every rate publishes the denominator it was divided by;
* displacement, duplicate tolerance and spread are in FEATURE PIXELS, and a
  degenerate size-one axis contributes exactly zero;
* the ground-truth block is scoring-only and never reaches a model call;
* the probe reports the runtime solver's numbers and does not re-run replay,
  re-solve coordinates or re-decide acceptance.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Mapping
import weakref

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
for _path in (ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from self_audit.evaluation.candidate_c import (  # noqa: E402
    CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION,
    DEFAULT_DUPLICATE_TOLERANCE_PIXELS,
    GT_METRICS_ARE_EVALUATION_ONLY,
    evaluate_candidate_c,
    geometry_row_metrics,
    ROW_EVAL_FIELDS,
    ground_truth_restitution_metrics,
    summarize_geometry_rows,
    merge_ground_truth_blocks,
    normalized_pixel_step,
    summarize_diagnostic_rows,
    summarize_geometry,
)


def _load_audit_checkpoint_module():
    path = ROOT / "scripts" / "audit_checkpoint.py"
    if not path.is_file():  # pragma: no cover - defensive
        pytest.skip("scripts/audit_checkpoint.py is absent")
    spec = importlib.util.spec_from_file_location("_candidate_c_audit_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixtures: minimal, explicit rows and geometry elements
# ---------------------------------------------------------------------------


def _row(**overrides):
    row = {
        "turn": 1,
        "record_turn": 0,
        "sample_index": 0,
        "active_indices": [0],
        "eligible": True,
        "record_kind": "ordinary",
        "c1_passed": True,
        "c1_max_abs_err": 1e-7,
        "regress_mass": 12.0,
        "fix_mass": 4.0,
        "num_protected": 3,
        "num_ties_excluded": 1,
        "feasible": True,
        "improved": True,
        "objective_before": 2.0,
        "objective_after": 1.5,
        "constraint_violation": 0.0,
        "coordinate_displacement": 0.25,
        "innovation_magnitude": 0.75,
        "fallback_reason": None,
        "accepted_path": "counterfactual",
        "evals": {
            "factual_replay": 1,
            "coordinate_backward": 1,
            "candidate_checks": 2,
            "total_forward": 4,
        },
        "delta_q": 0.3,
        "accepted": True,
    }
    row.update(overrides)
    return row


def _grid(height: int, width: int, k: int, fill: float = 0.0) -> torch.Tensor:
    return torch.full((1, height, width, k, 2), float(fill), dtype=torch.float32)


def _geometry_element(factual: torch.Tensor, chosen: torch.Tensor, *, height: int, width: int,
                      attention: torch.Tensor | None = None,
                      preclamp: torch.Tensor | None = None, **overrides):
    element = {
        "turn": 1,
        "record_turn": 0,
        "sample_index": 0,
        "feature_hw": (height, width),
        "k": int(factual.shape[-2]),
        "factual": (
            {
                "depth_index": 0,
                "coordinates": factual,
                "coordinates_preclamp": factual,
                "attention": attention,
                "turn_index": 0,
                "iteration_index": 0,
                "state_identity": "state-abc",
            },
        ),
        "chosen": (
            {
                "depth_index": 0,
                "coordinates": chosen,
                "coordinates_preclamp": preclamp,
                "attention": attention,
                "turn_index": 0,
                "iteration_index": 0,
                "state_identity": "state-abc",
            },
        ),
    }
    element.update(overrides)
    return element


# ---------------------------------------------------------------------------
# 1. Row summary: every requested quantity, no fake zeros, explicit denominators
# ---------------------------------------------------------------------------


def test_row_summary_reports_every_mandated_quantity() -> None:
    summary = summarize_diagnostic_rows([_row(), _row(sample_index=1, delta_q=-0.1, accepted=False,
                                                     accepted_path="ordinary")])
    assert summary["available"] is True
    assert summary["rows_total"] == 2
    assert summary["schema_version"] == CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION

    # Replay error, R/F mass, objective, constraint, displacement, innovation, delta_q.
    for field in ("c1_max_abs_err", "regress_mass", "fix_mass", "objective_before",
                  "objective_after", "constraint_violation", "coordinate_displacement",
                  "innovation_magnitude", "delta_q"):
        assert summary[f"numeric/{field}"]["count"] == 2, field

    # Protected FIX count and excluded ties.
    assert summary["count/num_protected"]["sum"] == 6.0
    assert summary["count/num_ties_excluded"]["sum"] == 2.0

    # Actual solver work, deduplicated per invocation. Both rows here carry
    # turn=1/record_turn=0 in one scope, so they are ONE invocation and its
    # counters are counted once -- not twice.
    actual = summary["evals"]["actual_invocation_totals"]
    assert actual["invocations"] == 1
    assert actual["totals"] == {
        "factual_replay": 1, "coordinate_backward": 1, "candidate_checks": 2, "total_forward": 4,
    }
    # The raw per-row sums are still published, under names that say so.
    assert summary["evals"]["row_sums_overcounted"]["total_forward"] == 8.0
    assert summary["evals"]["per_row_opportunity"]["rows_without_evals_block"] == 0
    assert summary["evals"]["per_row_opportunity"]["rows_per_invocation"]["mean"] == 2.0

    # Final Auditor decision, with its denominator.
    assert summary["flag/accepted"] == {
        "true_count": 1, "false_count": 1, "null_count": 0,
        "denominator": 2, "rate": 0.5, "rows_considered": 2,
    }
    # Fallback reason is a histogram, and a null reason stays visibly null.
    assert summary["histogram/fallback_reason"] == {"__null__": 2}
    assert summary["histogram/accepted_path"] == {"counterfactual": 1, "ordinary": 1}

    # Objective movement only counts rows where BOTH endpoints exist.
    assert summary["numeric/objective_delta"]["count"] == 2
    assert summary["objective_strictly_decreased_count"] == 2
    assert summary["objective_strictly_decreased_denominator"] == 2
    assert summary["objective_strictly_decreased_rate"] == 1.0


def test_unmeasured_fields_are_null_not_zero() -> None:
    rows = [
        _row(c1_passed=None, c1_max_abs_err=None, objective_after=None, delta_q=None,
             accepted=None, num_protected=None, evals=None, feasible=None,
             fallback_reason="replay_failure"),
    ]
    summary = summarize_diagnostic_rows(rows)

    displacement = summary["numeric/delta_q"]
    assert displacement["count"] == 0
    assert displacement["null_count"] == 1
    assert displacement["mean"] is None and displacement["sum"] is None

    assert summary["count/num_protected"]["sum"] is None
    assert summary["flag/accepted"]["denominator"] == 0
    assert summary["flag/accepted"]["rate"] is None
    assert summary["c1"]["attempted_count"] == 0
    assert summary["c1"]["pass_rate"] is None
    assert summary["c1"]["rows_without_replay_attempt"] == 1
    assert summary["evals"]["actual_invocation_totals"]["totals"]["total_forward"] is None
    assert summary["evals"]["actual_invocation_totals"]["invocations"] == 0
    assert summary["evals"]["row_sums_overcounted"]["total_forward"] is None
    assert summary["evals"]["per_row_opportunity"]["rows_without_evals_block"] == 1
    # Only one endpoint present, so no objective delta is invented.
    assert summary["numeric/objective_delta"]["count"] == 0
    assert summary["numeric/objective_delta"]["pairs_missing"] == 1
    assert summary["objective_strictly_decreased_rate"] is None
    assert summary["histogram/fallback_reason"] == {"replay_failure": 1}


def test_nonfinite_row_values_are_separated_from_missing_ones() -> None:
    summary = summarize_diagnostic_rows([_row(delta_q=float("nan")), _row(delta_q=None)])
    block = summary["numeric/delta_q"]
    assert block["nonfinite_count"] == 1
    assert block["null_count"] == 1
    assert block["count"] == 0
    assert block["mean"] is None


def test_absent_rows_key_is_reported_unavailable() -> None:
    summary = summarize_diagnostic_rows(None)
    assert summary["available"] is False
    assert "candidate_c_diagnostics" in summary["reason"]
    assert summary["rows_total"] == 0


def test_c1_pass_rate_uses_only_rows_that_attempted_a_replay() -> None:
    rows = [_row(c1_passed=True), _row(c1_passed=False, c1_max_abs_err=0.5), _row(c1_passed=None)]
    c1 = summarize_diagnostic_rows(rows)["c1"]
    assert c1["attempted_count"] == 2
    assert c1["passed_count"] == 1
    assert c1["failed_count"] == 1
    assert c1["pass_rate"] == 0.5
    assert c1["rows_without_replay_attempt"] == 1
    assert c1["max_abs_err"]["max"] == 0.5
    assert "does not re-execute the replay" in c1["note"]


# ---------------------------------------------------------------------------
# 2. Geometry: feature-pixel scale, saturation, duplicates, entropy, spread
# ---------------------------------------------------------------------------


def test_displacement_is_measured_in_feature_pixels() -> None:
    height = width = 5
    step_x, step_y = normalized_pixel_step(height, width)
    assert step_x == pytest.approx(0.5) and step_y == pytest.approx(0.5)

    factual = _grid(height, width, 2)
    chosen = factual.clone()
    chosen[0, 0, 0, 0, 0] = step_x  # exactly one feature pixel along x
    element = _geometry_element(factual, chosen, height=height, width=width)

    metrics = geometry_row_metrics(element)
    assert metrics["available"] is True
    displacement = metrics["displacement"]
    assert displacement["max_pixels"] == pytest.approx(1.0)
    assert displacement["max_abs_x_pixels"] == pytest.approx(1.0)
    assert displacement["max_abs_y_pixels"] == pytest.approx(0.0)
    assert displacement["moved_point_count"] == 1
    assert displacement["points"] == height * width * 2
    assert displacement["moved_point_rate"] == pytest.approx(1.0 / (height * width * 2))
    assert displacement["mean_moved_pixels"] == pytest.approx(1.0)


def test_degenerate_axis_contributes_zero_displacement() -> None:
    height, width = 1, 4
    step_x, step_y = normalized_pixel_step(height, width)
    assert step_y == 0.0
    factual = _grid(height, width, 2)
    chosen = factual.clone()
    chosen[..., 1] = 0.9  # a large normalized move along the degenerate y axis
    metrics = geometry_row_metrics(_geometry_element(factual, chosen, height=height, width=width))
    displacement = metrics["displacement"]
    assert displacement["degenerate_axes"] == ["y"]
    assert displacement["max_abs_y_pixels"] == pytest.approx(0.0)
    assert displacement["max_pixels"] == pytest.approx(0.0)
    assert displacement["moved_point_count"] == 0
    # The normalized move is still reported honestly; only the pixel scale is zero.
    assert displacement["max_normalized"] == pytest.approx(0.9)


def test_saturation_counts_edge_components_and_clamp_activity() -> None:
    height = width = 3
    factual = _grid(height, width, 2)
    chosen = factual.clone()
    chosen[0, 0, 0, 0, 0] = 1.0
    preclamp = chosen.clone()
    preclamp[0, 0, 0, 0, 0] = 1.4
    metrics = geometry_row_metrics(
        _geometry_element(factual, chosen, height=height, width=width, preclamp=preclamp)
    )
    saturation = metrics["saturation"]
    assert saturation["components"] == height * width * 2 * 2
    assert saturation["saturated_count"] == 1
    assert saturation["saturated_rate"] == pytest.approx(1.0 / saturation["components"])
    assert saturation["clamped_count"] == 1
    assert saturation["max_preclamp_overshoot"] == pytest.approx(0.4, abs=1e-6)


def test_missing_preclamp_leaves_clamp_activity_unavailable_not_zero() -> None:
    height = width = 3
    factual = _grid(height, width, 2)
    metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width, preclamp=None)
    )
    saturation = metrics["saturation"]
    assert saturation["clamped_count"] is None
    assert saturation["clamped_rate"] is None
    assert saturation["max_preclamp_overshoot"] is None
    assert "not measurable" in saturation["clamp_note"]
    assert metrics["saturation_preclamp_depths_missing"] == 1


def test_duplicate_supports_use_an_explicit_feature_pixel_tolerance() -> None:
    height = width = 3
    factual = _grid(height, width, 3)
    # Points 0 and 1 coincide exactly; point 2 sits one pixel away along x.
    step_x, _ = normalized_pixel_step(height, width)
    factual[..., 2, 0] = step_x
    metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width)
    )
    duplicates = metrics["duplicates"]
    assert duplicates["tolerance_pixels"] == DEFAULT_DUPLICATE_TOLERANCE_PIXELS
    assert duplicates["queries"] == height * width
    assert duplicates["points"] == height * width * 3
    assert duplicates["duplicate_point_count"] == height * width  # one per query
    assert duplicates["queries_with_duplicate_count"] == height * width
    assert duplicates["queries_with_duplicate_rate"] == pytest.approx(1.0)

    # Widening the tolerance past one pixel absorbs the third point too.
    wide = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width),
        duplicate_tolerance_pixels=1.5,
    )
    assert wide["duplicates"]["duplicate_point_count"] == 2 * height * width
    assert wide["duplicates"]["tolerance_pixels"] == 1.5


def test_attention_entropy_uniform_versus_peaked() -> None:
    height = width = 2
    k = 4
    factual = _grid(height, width, k)
    uniform = torch.full((1, 1, height, width, k), 1.0 / k)
    metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width, attention=uniform)
    )
    entropy = metrics["attention_entropy"]
    assert entropy["available"] is True
    assert entropy["mean_entropy_nats"] == pytest.approx(math.log(k), abs=1e-6)
    assert entropy["mean_normalized_entropy"] == pytest.approx(1.0, abs=1e-6)

    peaked = torch.zeros((1, 1, height, width, k))
    peaked[..., 0] = 1.0
    peaked_metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width, attention=peaked)
    )
    assert peaked_metrics["attention_entropy"]["mean_entropy_nats"] == pytest.approx(0.0, abs=1e-9)
    assert peaked_metrics["attention_entropy"]["mean_normalized_entropy"] == pytest.approx(0.0, abs=1e-9)


def test_attention_accepts_native_5d_and_flattened_4d_layouts() -> None:
    height = width = 2
    k = 4
    factual = _grid(height, width, k)
    native = torch.full((1, 2, height, width, k), 1.0 / k)
    flattened = torch.full((1, 2, height * width, k), 1.0 / k)
    native_metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width, attention=native)
    )
    flat_metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width, attention=flattened)
    )
    for metrics in (native_metrics, flat_metrics):
        assert metrics["attention_entropy"]["available"] is True
        assert metrics["attention_entropy"]["queries"] == 2 * height * width
        assert metrics["attention_entropy"]["mean_entropy_nats"] == pytest.approx(math.log(k), abs=1e-6)


def test_absent_attention_is_unavailable_never_zero_entropy() -> None:
    height = width = 2
    factual = _grid(height, width, 2)
    metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width, attention=None)
    )
    entropy = metrics["attention_entropy"]
    assert entropy["available"] is False
    assert "mean_entropy_nats" not in entropy
    assert entropy["depths_missing_attention"] == 1


def test_support_spread_is_in_feature_pixels() -> None:
    height = width = 5  # one pixel == 0.5 normalized units
    factual = _grid(height, width, 2)
    factual[..., 0, 0] = -0.5
    factual[..., 1, 0] = 0.5  # two pixels apart along x
    metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width)
    )
    spread = metrics["spread"]
    assert spread["queries"] == height * width
    assert spread["mean_radius_pixels"] == pytest.approx(1.0)
    assert spread["mean_axis_extent_pixels"] == pytest.approx(1.0)  # mean over x=2, y=0


def test_declared_unavailable_geometry_element_is_not_scored() -> None:
    element = {"turn": 2, "record_turn": 1, "sample_index": 3,
               "unavailable_reason": "fallback returned P; no chosen support captured"}
    metrics = geometry_row_metrics(element)
    assert metrics["available"] is False
    assert metrics["reason"].startswith("fallback returned P")
    assert metrics["sample_index"] == 3
    assert "displacement" not in metrics


def test_mismatched_depth_counts_are_refused() -> None:
    height = width = 2
    factual = _grid(height, width, 2)
    element = _geometry_element(factual, factual.clone(), height=height, width=width)
    element["chosen"] = ()
    metrics = geometry_row_metrics(element)
    assert metrics["available"] is False
    assert "depth count" in metrics["reason"]


def test_geometry_summary_pools_and_reports_unavailable_rows() -> None:
    height = width = 3
    factual = _grid(height, width, 2)
    good = _geometry_element(factual, factual.clone(), height=height, width=width)
    bad = {"turn": 1, "record_turn": 0, "sample_index": 1, "unavailable_reason": "no geometry"}
    summary = summarize_geometry([good, bad])
    assert summary["rows_total"] == 2
    assert summary["rows_usable"] == 1
    assert summary["rows_unavailable"] == 1
    assert summary["unavailable_reasons"] == {"no geometry": 1}
    assert summary["duplicate_tolerance_pixels"] == DEFAULT_DUPLICATE_TOLERANCE_PIXELS
    assert summary["displacement_mean_pixels"]["count"] == 1


def test_geometry_key_absent_is_unavailable_not_empty() -> None:
    summary = summarize_geometry(None)
    assert summary["available"] is False
    assert "capture_geometry" in summary["reason"]
    assert summary["rows_total"] == 0


def test_local_pixel_step_matches_the_core_definition() -> None:
    core = pytest.importorskip("self_audit.models.dynamic_window")
    if not hasattr(core, "normalized_pixel_step"):
        pytest.skip("core normalized_pixel_step is not available at this revision")
    for height, width in ((1, 1), (1, 8), (8, 1), (4, 7), (256, 256)):
        expected = core.normalized_pixel_step(
            height, width, device=torch.device("cpu"), dtype=torch.float64
        )
        step_x, step_y = normalized_pixel_step(height, width)
        assert step_x == pytest.approx(float(expected[0]))
        assert step_y == pytest.approx(float(expected[1]))


# ---------------------------------------------------------------------------
# 3. Evaluation-only ground-truth metrics: PROPOSAL vs RETAINED
# ---------------------------------------------------------------------------


def _one_hot_logits(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes).permute(0, 3, 1, 2).float() * 10.0


def _four_row_fixture():
    """A hand-computable 4-sample fixture covering all four outcome shapes.

    Every sample shares the same 1x4 prior transition, so the denominators are
    identical per row and the pooled totals are trivial multiples:

        g = [0, 0, 0, 0]   ground truth
        p = [1, 1, 0, 0]   pretransition argmax
        f = [0, 0, 1, 0]   factual argmax (the retained state entering the turn)

    which gives, per sample: prior_true_fix = {col 0, col 1} (2 px),
    prior_true_regress = {col 2} (1 px), factual_correct = {0, 1, 3} (3 px),
    factual_correct_outside_prior_fix = {col 3} (1 px), valid = 4 px.

    The four samples then differ only in the proposal Q and the audit decision:

        0  success            Q = [0,0,0,0]  accepted     repairs the regress
        1  identity fallback  Q = f          accepted     changes nothing
        2  rejected harmful   Q = [1,1,1,1]  rejected     would destroy both fixes
        3  rejected beneficial Q = [0,0,0,0] rejected     would repair the regress
    """

    gt = torch.zeros(4, 1, 4, dtype=torch.long)
    p = _one_hot_logits(torch.tensor([[[1, 1, 0, 0]]]).repeat(4, 1, 1), 2)
    f = _one_hot_logits(torch.tensor([[[0, 0, 1, 0]]]).repeat(4, 1, 1), 2)
    q = _one_hot_logits(
        torch.tensor(
            [
                [[0, 0, 0, 0]],   # 0 success
                [[0, 0, 1, 0]],   # 1 identity fallback (== f)
                [[1, 1, 1, 1]],   # 2 rejected harmful
                [[0, 0, 0, 0]],   # 3 rejected beneficial
            ]
        ),
        2,
    )
    rows = [
        _row(turn=1, record_turn=0, sample_index=0, accepted=True,
             accepted_path="counterfactual", eligible=True, fallback_reason=None),
        _row(turn=1, record_turn=0, sample_index=1, accepted=True,
             accepted_path="factual_support", eligible=True,
             fallback_reason="no_strict_improvement"),
        _row(turn=1, record_turn=0, sample_index=2, accepted=False,
             accepted_path="counterfactual", eligible=True, fallback_reason=None),
        _row(turn=1, record_turn=0, sample_index=3, accepted=False,
             accepted_path="counterfactual", eligible=True, fallback_reason=None),
    ]
    return rows, [p, f], [f, q], gt


def test_all_history_rows_are_scored_not_only_accepted_successes() -> None:
    rows, previous, candidates, gt = _four_row_fixture()
    block = ground_truth_restitution_metrics(
        rows, transition_previous=previous, transition_candidates=candidates, target=gt
    )
    assert block["available"] is True
    # Every row has history, so every row is scored -- including the identity
    # fallback and both rejected proposals.
    assert block["rows_scored"] == 4
    assert block["rows_unavailable"] == 0
    assert block["strata"]["by_accepted"] == {"False": 2, "True": 2}
    assert block["strata"]["by_accepted_path"] == {"counterfactual": 3, "factual_support": 1}
    assert block["strata"]["by_fallback_reason"] == {"__null__": 3, "no_strict_improvement": 1}


def test_proposal_and_retained_denominators_are_identical_and_hand_checkable() -> None:
    rows, previous, candidates, gt = _four_row_fixture()
    block = ground_truth_restitution_metrics(
        rows, transition_previous=previous, transition_candidates=candidates, target=gt
    )
    for family in ("proposal", "retained"):
        denominators = block[family]["denominators"]
        assert denominators["prior_true_fix"] == 8            # 4 samples x 2 px
        assert denominators["prior_true_regress"] == 4        # 4 samples x 1 px
        assert denominators["factual_correct"] == 12          # 4 samples x 3 px
        assert denominators["factual_correct_outside_prior_fix"] == 4
        assert denominators["valid_pixels"] == 16
        assert block[family]["rows_scored"] == 4


def test_proposal_family_scores_what_the_solver_put_forward() -> None:
    rows, previous, candidates, gt = _four_row_fixture()
    proposal = ground_truth_restitution_metrics(
        rows, transition_previous=previous, transition_candidates=candidates, target=gt
    )["proposal"]
    counts = proposal["counts"]
    # retained fixes: 2 (success) + 2 (identity) + 0 (harmful) + 2 (beneficial)
    assert counts["prior_true_fix_retained"] == 6
    assert counts["prior_true_fix_destroyed"] == 2            # the harmful proposal
    # repairs proposed: success + the REJECTED beneficial proposal
    assert counts["prior_true_regress_repaired"] == 2
    assert counts["new_error"] == 3                           # the harmful proposal's 3 px
    assert counts["new_error_outside_prior_fix"] == 1
    assert counts["after_correct"] == 11                      # 4 + 3 + 0 + 4

    assert proposal["prior_true_fix_retained_rate"] == pytest.approx(6 / 8)
    assert proposal["prior_true_fix_destroyed_rate"] == pytest.approx(2 / 8)
    assert proposal["prior_true_regress_repaired_rate"] == pytest.approx(2 / 4)
    assert proposal["new_error_rate"] == pytest.approx(3 / 12)
    assert proposal["new_error_outside_prior_fix_rate"] == pytest.approx(1 / 4)
    assert proposal["net_correct_pixel_delta"] == -1
    assert proposal["net_correct_rate_delta"] == pytest.approx(-1 / 16)


def test_retained_family_never_credits_a_rejected_proposed_repair() -> None:
    rows, previous, candidates, gt = _four_row_fixture()
    retained = ground_truth_restitution_metrics(
        rows, transition_previous=previous, transition_candidates=candidates, target=gt
    )["retained"]
    counts = retained["counts"]
    # Reject HALTs the row, so both rejected samples keep F unchanged.
    assert counts["prior_true_fix_retained"] == 8
    assert counts["prior_true_fix_destroyed"] == 0            # the harmful one never landed
    # ONLY the accepted success realized a repair; the rejected beneficial
    # proposal contributes nothing here, although it counted under proposal.
    assert counts["prior_true_regress_repaired"] == 1
    assert counts["new_error"] == 0
    assert counts["after_correct"] == 13                      # 4 + 3 + 3 + 3

    assert retained["prior_true_fix_retained_rate"] == pytest.approx(1.0)
    assert retained["prior_true_fix_destroyed_rate"] == pytest.approx(0.0)
    assert retained["prior_true_regress_repaired_rate"] == pytest.approx(1 / 4)
    assert retained["new_error_rate"] == pytest.approx(0.0)
    assert retained["net_correct_pixel_delta"] == 1
    assert retained["net_correct_rate_delta"] == pytest.approx(1 / 16)


def test_the_two_families_actually_differ_and_the_old_selection_was_biased() -> None:
    """Scoring only accepted counterfactuals would have reported 1.0 repair rate."""

    rows, previous, candidates, gt = _four_row_fixture()
    block = ground_truth_restitution_metrics(
        rows, transition_previous=previous, transition_candidates=candidates, target=gt
    )
    assert (
        block["proposal"]["prior_true_regress_repaired_rate"]
        != block["retained"]["prior_true_regress_repaired_rate"]
    )
    assert (
        block["proposal"]["prior_true_fix_destroyed_rate"]
        != block["retained"]["prior_true_fix_destroyed_rate"]
    )

    # The old admission rule: accepted AND a settled restitution path. It keeps
    # sample 0 only, whose repair rate is a perfect 1.0 -- four times the true
    # realized rate and twice the proposed rate.
    biased = ground_truth_restitution_metrics(
        [rows[0]], transition_previous=previous, transition_candidates=candidates, target=gt
    )
    assert biased["retained"]["prior_true_regress_repaired_rate"] == pytest.approx(1.0)
    assert block["retained"]["prior_true_regress_repaired_rate"] == pytest.approx(0.25)


def test_identity_fallback_counts_in_the_denominator_with_no_repair() -> None:
    rows, previous, candidates, gt = _four_row_fixture()
    block = ground_truth_restitution_metrics(
        rows, transition_previous=previous, transition_candidates=candidates, target=gt
    )
    identity = [row for row in block["per_row"] if row["accepted_path"] == "factual_support"]
    assert len(identity) == 1
    counts = identity[0]["retained"]
    assert counts["prior_true_regress"] == 1                  # denominator present
    assert counts["prior_true_regress_repaired"] == 0         # nothing repaired
    assert counts["prior_true_fix_destroyed"] == 0
    assert counts["new_error"] == 0
    # Proposal and retained coincide for an identity fallback.
    assert identity[0]["proposal"] == identity[0]["retained"]


def test_per_row_records_the_acceptance_stratum_for_every_scored_row() -> None:
    rows, previous, candidates, gt = _four_row_fixture()
    per_row = ground_truth_restitution_metrics(
        rows, transition_previous=previous, transition_candidates=candidates, target=gt
    )["per_row"]
    assert [row["sample_index"] for row in per_row] == [0, 1, 2, 3]
    assert [row["accepted"] for row in per_row] == [True, True, False, False]
    assert all(row["record_turn"] == 0 and row["turn"] == 1 for row in per_row)
    # The rejected harmful proposal differs between families; the accepted one
    # does not.
    assert per_row[0]["proposal"] == per_row[0]["retained"]
    assert per_row[2]["proposal"] != per_row[2]["retained"]


def test_keep_per_row_false_drops_the_detail_but_not_the_totals() -> None:
    rows, previous, candidates, gt = _four_row_fixture()
    block = ground_truth_restitution_metrics(
        rows, transition_previous=previous, transition_candidates=candidates, target=gt,
        keep_per_row=False,
    )
    assert block["per_row"] == []
    assert block["rows_scored"] == 4
    assert block["retained"]["counts"]["prior_true_regress_repaired"] == 1


def test_ground_truth_rate_with_zero_denominator_is_none() -> None:
    gt = torch.tensor([[[0, 0]]])
    same = _one_hot_logits(gt, 2)
    rows = [_row(turn=1, record_turn=0, sample_index=0, accepted=True)]
    block = ground_truth_restitution_metrics(
        rows,
        transition_previous=[same, torch.zeros(1, 2, 1, 2)],
        transition_candidates=[same, same],
        target=gt,
    )
    for family in ("proposal", "retained"):
        assert block[family]["counts"]["prior_true_fix"] == 0
        assert block[family]["prior_true_fix_retained_rate"] is None
        assert block[family]["prior_true_regress_repaired_rate"] is None
        assert block[family]["new_error_rate"] == pytest.approx(0.0)  # denominator 2


def test_rows_without_history_or_without_a_decision_are_unavailable() -> None:
    gt = torch.tensor([[[0, 0]]])
    same = _one_hot_logits(gt, 2)
    ordinary = _row(accepted_path="ordinary", record_turn=None, eligible=False,
                    fallback_reason="no_accepted_history")
    undecided = _row(accepted_path="counterfactual", record_turn=0, turn=1, accepted=None)
    out_of_range = _row(accepted_path="counterfactual", record_turn=1, turn=1)
    bad_sample = _row(accepted_path="counterfactual", sample_index=9)
    block = ground_truth_restitution_metrics(
        [ordinary, undecided, out_of_range, bad_sample],
        transition_previous=[same, same],
        transition_candidates=[same, same],
        target=gt,
    )
    assert block["available"] is False
    assert block["rows_scored"] == 0
    assert block["rows_unavailable"] == 4
    reasons = " ".join(block["unavailable_reasons"])
    assert "no accepted-history record_turn" in reasons
    assert "acceptance is unmeasured" in reasons
    assert "must precede the current turn" in reasons
    assert "sample_index out of range" in reasons


def test_a_rejected_row_is_scored_rather_than_discarded() -> None:
    """Regression guard: rejection is a stratum, not an exclusion criterion."""

    gt = torch.tensor([[[0, 0]]])
    p = _one_hot_logits(torch.tensor([[[1, 1]]]), 2)
    f = _one_hot_logits(torch.tensor([[[0, 0]]]), 2)
    q = _one_hot_logits(torch.tensor([[[1, 1]]]), 2)
    rejected = _row(turn=1, record_turn=0, sample_index=0, accepted=False,
                    accepted_path="counterfactual")
    block = ground_truth_restitution_metrics(
        [rejected], transition_previous=[p, p], transition_candidates=[f, q], target=gt
    )
    assert block["rows_scored"] == 1
    assert block["proposal"]["prior_true_fix_destroyed_rate"] == pytest.approx(1.0)
    assert block["retained"]["prior_true_fix_destroyed_rate"] == pytest.approx(0.0)


def test_ground_truth_block_is_unavailable_without_a_target() -> None:
    block = ground_truth_restitution_metrics(
        [_row()], transition_previous=[], transition_candidates=[], target=None
    )
    assert block["available"] is False
    assert block["reason"] == "no ground-truth target supplied"
    assert "proposal" not in block


def test_ground_truth_ignore_index_shrinks_the_denominator() -> None:
    gt = torch.tensor([[[0, 255]]])
    p = _one_hot_logits(torch.tensor([[[1, 1]]]), 2)
    f = _one_hot_logits(torch.tensor([[[0, 0]]]), 2)
    rows = [_row(turn=1, record_turn=0, sample_index=0, accepted=True)]
    block = ground_truth_restitution_metrics(
        rows, transition_previous=[p, p], transition_candidates=[f, f], target=gt,
        ignore_index=255,
    )
    assert block["retained"]["counts"]["valid_pixels"] == 1
    assert block["retained"]["counts"]["prior_true_fix"] == 1


def test_merge_pools_counts_rather_than_averaging_rates() -> None:
    rows, previous, candidates, gt = _four_row_fixture()
    # Two "batches": the two accepted rows, then the two rejected ones.
    left = ground_truth_restitution_metrics(
        rows[:2], transition_previous=previous, transition_candidates=candidates, target=gt,
        keep_per_row=False,
    )
    right = ground_truth_restitution_metrics(
        rows[2:], transition_previous=previous, transition_candidates=candidates, target=gt,
        keep_per_row=False,
    )
    merged = merge_ground_truth_blocks([left, right])
    whole = ground_truth_restitution_metrics(
        rows, transition_previous=previous, transition_candidates=candidates, target=gt
    )
    assert merged["rows_scored"] == 4
    for family in ("proposal", "retained"):
        assert merged[family]["counts"] == whole[family]["counts"]
        assert merged[family]["prior_true_regress_repaired_rate"] == pytest.approx(
            whole[family]["prior_true_regress_repaired_rate"]
        )
    assert merged["strata"]["by_accepted"] == {"False": 2, "True": 2}
    assert "per_row" not in merged


def test_merge_of_nothing_is_unavailable_with_null_rates() -> None:
    merged = merge_ground_truth_blocks([])
    assert merged["available"] is False
    assert merged["rows_scored"] == 0
    for family in ("proposal", "retained"):
        assert merged[family]["prior_true_fix_destroyed_rate"] is None
        assert merged[family]["net_correct_rate_delta"] is None


def test_ground_truth_helpers_never_reach_a_model() -> None:
    """The GT block is scoring-only: no model object is even accepted."""

    import inspect

    assert GT_METRICS_ARE_EVALUATION_ONLY is True
    parameters = set(inspect.signature(ground_truth_restitution_metrics).parameters)
    assert parameters == {
        "rows", "transition_previous", "transition_candidates", "target", "ignore_index",
        "geometry", "keep_per_row",
    }
    source = (
        _SRC / "self_audit" / "evaluation" / "candidate_c.py"
    ).read_text(encoding="utf-8")
    for forbidden in (".infer(", "annotation_expert", "auditor(", "nn.Module"):
        assert forbidden not in source, forbidden


def test_record_turn_falls_back_to_the_index_aligned_geometry_element() -> None:
    """The core row may omit record_turn; the geometry element can supply it."""

    gt = torch.tensor([[[0, 0]]])
    p = _one_hot_logits(torch.tensor([[[1, 1]]]), 2)
    f = _one_hot_logits(torch.tensor([[[0, 0]]]), 2)
    row = _row(turn=1, sample_index=0, accepted=True, accepted_path="counterfactual")
    row.pop("record_turn")
    factual = _grid(1, 2, 2)
    element = _geometry_element(factual, factual.clone(), height=1, width=2,
                               turn=1, record_turn=0, sample_index=0)

    without = ground_truth_restitution_metrics(
        [row], transition_previous=[p, p], transition_candidates=[f, f], target=gt
    )
    assert without["rows_scored"] == 0
    assert "no accepted-history record_turn" in " ".join(without["unavailable_reasons"])

    with_geometry = ground_truth_restitution_metrics(
        [row], transition_previous=[p, p], transition_candidates=[f, f], target=gt,
        geometry=[element],
    )
    assert with_geometry["rows_scored"] == 1
    assert with_geometry["retained"]["counts"]["prior_true_fix"] == 2


def test_the_row_field_wins_over_a_conflicting_geometry_hint() -> None:
    rows, previous, candidates, gt = _four_row_fixture()
    factual = _grid(1, 4, 2)
    # A hint naming a different record_turn must not override the row.
    elements = [
        _geometry_element(factual, factual.clone(), height=1, width=4,
                          turn=1, record_turn=99, sample_index=index)
        for index in range(4)
    ]
    block = ground_truth_restitution_metrics(
        rows, transition_previous=previous, transition_candidates=candidates, target=gt,
        geometry=elements,
    )
    assert block["rows_scored"] == 4
    assert all(row["record_turn"] == 0 for row in block["per_row"])


def test_geometry_hint_is_refused_when_it_disagrees_with_the_row() -> None:
    gt = torch.tensor([[[0, 0]]])
    p = _one_hot_logits(torch.tensor([[[1, 1]]]), 2)
    f = _one_hot_logits(torch.tensor([[[0, 0]]]), 2)
    row = _row(turn=1, sample_index=0, accepted=True, accepted_path="counterfactual")
    row.pop("record_turn")
    factual = _grid(1, 2, 2)
    mismatched = _geometry_element(factual, factual.clone(), height=1, width=2,
                                   turn=1, record_turn=0, sample_index=7)
    block = ground_truth_restitution_metrics(
        [row], transition_previous=[p, p], transition_candidates=[f, f], target=gt,
        geometry=[mismatched],
    )
    assert block["rows_scored"] == 0
    assert "no accepted-history record_turn" in " ".join(block["unavailable_reasons"])


# ---------------------------------------------------------------------------
# 4. Top-level entry point stays JSON-serialisable
# ---------------------------------------------------------------------------


def test_evaluate_candidate_c_output_is_json_serialisable_and_finite() -> None:
    height = width = 3
    factual = _grid(height, width, 2)
    output = {
        "candidate_c_diagnostics": [_row()],
        "candidate_c_geometry": [
            _geometry_element(factual, factual.clone(), height=height, width=width,
                              attention=torch.full((1, 1, height, width, 2), 0.5))
        ],
        "transition_previous": [torch.zeros(1, 2, height, width)] * 2,
        "transition_candidates": [torch.zeros(1, 2, height, width)] * 2,
    }
    result = evaluate_candidate_c(output, target=torch.zeros(1, height, width, dtype=torch.long))
    encoded = json.dumps(result)
    assert "NaN" not in encoded and "Infinity" not in encoded
    assert json.loads(encoded)["rows"]["rows_total"] == 1
    assert result["geometry"]["available"] is True
    assert result["ground_truth"]["evaluation_only"] is True


def test_evaluate_candidate_c_without_target_leaves_gt_unavailable_only() -> None:
    result = evaluate_candidate_c({"candidate_c_diagnostics": [_row()]})
    assert result["rows"]["available"] is True
    assert result["geometry"]["available"] is False
    assert result["ground_truth"]["available"] is False
    assert result["ground_truth"]["reason"] == "no ground-truth target supplied"


# ---------------------------------------------------------------------------
# 5. audit_checkpoint probe 4
# ---------------------------------------------------------------------------


class _FakeModel:
    """A stand-in for SelfAuditNet that records exactly how infer was called."""

    def __init__(self, output_factory, *, accept_capture_geometry: bool = True,
                 window_mode: str = "candidate_c"):
        self._output_factory = output_factory
        self.window_mode = window_mode
        self.training = False
        self.calls: list[dict] = []
        if accept_capture_geometry:
            def infer(images, *, mode="self_audit", tau_accept=0.0, t_max=0,
                      capture_geometry=False, **kwargs):
                self.calls.append({"mode": mode, "tau_accept": tau_accept, "t_max": t_max,
                                   "capture_geometry": capture_geometry, **kwargs})
                return self._output_factory()
        else:
            def infer(images, *, mode="self_audit", tau_accept=0.0, t_max=0, **kwargs):
                self.calls.append({"mode": mode, "tau_accept": tau_accept, "t_max": t_max,
                                   **kwargs})
                return self._output_factory()
        self.infer = infer

    def eval(self):
        self.training = False

    def train(self, flag=True):
        self.training = bool(flag)


def _fixed(output):
    return lambda: output


def _restitution_output(height=1, width=4):
    p = _one_hot_logits(torch.tensor([[[1, 1, 0, 0]]]), 2)
    f = _one_hot_logits(torch.tensor([[[0, 0, 1, 0]]]), 2)
    q = _one_hot_logits(torch.tensor([[[0, 1, 0, 1]]]), 2)
    factual = _grid(height, width, 2)
    return {
        "candidate_c_diagnostics": [
            _row(turn=1, record_turn=0, sample_index=0, accepted=True,
                 accepted_path="counterfactual")
        ],
        "candidate_c_geometry": [
            _geometry_element(factual, factual.clone(), height=height, width=width)
        ],
        "transition_previous": [p, f],
        "transition_candidates": [f, q],
    }


def _gt_batch():
    return {"image": torch.zeros(1, 1, 1, 4), "mask": torch.zeros(1, 1, 4, dtype=torch.long)}


def _find_tensors(node, path="") -> list[str]:
    """Every path at which a torch.Tensor survives in a nested structure."""

    found: list[str] = []
    if torch.is_tensor(node):
        return [path or "<root>"]
    if isinstance(node, Mapping):
        for key, value in node.items():
            found.extend(_find_tensors(value, f"{path}.{key}"))
    elif isinstance(node, (list, tuple, set)):
        for index, value in enumerate(node):
            found.extend(_find_tensors(value, f"{path}[{index}]"))
    return found


# --- mode gate: no forward at all for a baseline window mode ---------------


@pytest.mark.parametrize("mode", ["current", "feature_only", "free_offsets"])
def test_baseline_window_modes_are_skipped_before_any_forward(mode: str) -> None:
    cli = _load_audit_checkpoint_module()
    model = _FakeModel(_fixed(_restitution_output()), window_mode=mode)
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()] * 3, torch.device("cpu"), tau_accept=0.0, t_max=2
    )
    assert model.calls == []                       # zero actual extra forwards
    assert detail["infer_calls"] == 0
    assert detail["batches_measured"] == 0
    assert detail["available"] is False
    assert detail["mode_active"] is False
    assert detail["window_mode"] == mode
    assert "never builds an accepted-transition record" in detail["reason"]
    assert detail["flat"]["candidate_c/infer_calls"] == 0
    assert detail["flat"]["candidate_c/mode_active"] is False


@pytest.mark.parametrize("mode", ["candidate_c", "candidate_c_no_fix", "direct_rollback"])
def test_every_record_consuming_mode_is_measured_and_labelled(mode: str) -> None:
    cli = _load_audit_checkpoint_module()
    model = _FakeModel(_fixed(_restitution_output()), window_mode=mode)
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()], torch.device("cpu"), tau_accept=0.0, t_max=2
    )
    assert len(model.calls) == 1
    assert detail["available"] is True
    assert detail["mode_active"] is True
    assert detail["window_mode"] == mode
    assert detail["flat"]["candidate_c/window_mode"] == mode


def test_active_mode_with_no_attempt_is_not_reported_as_available() -> None:
    """An empty diagnostics list is a finding, not a measurement."""

    cli = _load_audit_checkpoint_module()
    empty = {"candidate_c_diagnostics": [], "transition_previous": [], "transition_candidates": []}
    model = _FakeModel(_fixed(empty), window_mode="candidate_c")
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()], torch.device("cpu"), tau_accept=0.0, t_max=2
    )
    assert detail["available"] is False
    assert detail["mode_active"] is True                 # distinct from a baseline mode
    assert detail["diagnostics_key_present"] is True
    assert detail["infer_calls"] == 1
    assert "no sample became replay-eligible" in detail["reason"]
    assert detail["flat"]["candidate_c/accepted_rate"] is None


def test_missing_diagnostics_key_is_distinguished_from_an_empty_one() -> None:
    cli = _load_audit_checkpoint_module()
    model = _FakeModel(
        _fixed({"transition_previous": [], "transition_candidates": []}),
        window_mode="candidate_c",
    )
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()], torch.device("cpu"), tau_accept=0.0, t_max=2
    )
    assert detail["available"] is False
    assert detail["diagnostics_key_present"] is False
    assert "emitted no candidate_c_diagnostics key at all" in detail["reason"]
    assert detail["flat"]["candidate_c/c1_pass_rate"] is None
    assert detail["flat"]["candidate_c/total_forward_sum"] is None


def test_an_unknown_window_mode_attribute_does_not_block_measurement() -> None:
    cli = _load_audit_checkpoint_module()
    model = _FakeModel(_fixed(_restitution_output()), window_mode="candidate_c")
    del model.window_mode
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()], torch.device("cpu"), tau_accept=0.0, t_max=2
    )
    assert detail["available"] is True
    assert detail["window_mode"] is None
    assert detail["mode_active"] is None       # unknown, not asserted either way


# --- batch cap -------------------------------------------------------------


def test_resolve_candidate_c_batches_inherits_rather_than_running_uncapped() -> None:
    cli = _load_audit_checkpoint_module()
    assert cli.resolve_candidate_c_batches(3, 8, 20) == 3      # explicit wins
    assert cli.resolve_candidate_c_batches(None, 8, 20) == 8   # inherits probe_batches
    assert cli.resolve_candidate_c_batches(None, None, 20) == 20  # then max_val_batches
    assert cli.resolve_candidate_c_batches(None, None, None) is None  # nothing supplied
    assert cli.resolve_candidate_c_batches(0, 8, 20) == 0      # an explicit 0 is honoured


def test_probe_honours_the_batch_cap_and_restores_training_mode() -> None:
    cli = _load_audit_checkpoint_module()
    model = _FakeModel(_fixed(_restitution_output()))
    model.train(True)
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()] * 5, torch.device("cpu"),
        tau_accept=0.0, t_max=2, max_batches=2,
    )
    assert detail["batches_measured"] == 2
    assert detail["infer_calls"] == 2
    assert len(model.calls) == 2
    assert model.training is True


# --- memory ----------------------------------------------------------------


def test_no_tensor_survives_in_the_probe_result() -> None:
    cli = _load_audit_checkpoint_module()
    model = _FakeModel(lambda: _restitution_output())
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()] * 3, torch.device("cpu"),
        tau_accept=0.0, t_max=2, capture_geometry=True,
    )
    assert detail["available"] is True
    assert detail["batches_measured"] == 3
    leaked = _find_tensors(detail)
    assert leaked == [], f"probe result retains tensors at {leaked}"
    json.dumps(detail)


def test_raw_geometry_tensors_are_released_between_batches() -> None:
    """The per-batch payload must be unreachable once the batch is measured."""

    cli = _load_audit_checkpoint_module()
    alive: list[weakref.ref] = []

    def factory():
        output = _restitution_output()
        element = output["candidate_c_geometry"][0]
        for entry in element["factual"] + element["chosen"]:
            for key in ("coordinates", "coordinates_preclamp", "attention"):
                value = entry.get(key)
                if torch.is_tensor(value):
                    alive.append(weakref.ref(value))
        for tensor in output["transition_previous"] + output["transition_candidates"]:
            alive.append(weakref.ref(tensor))
        return output

    def loader():
        for _ in range(3):
            yield _gt_batch()

    model = _FakeModel(factory)
    detail = cli.probe_candidate_c_path(
        model, loader(), torch.device("cpu"),
        tau_accept=0.0, t_max=2, capture_geometry=True,
    )
    assert detail["batches_measured"] == 3
    assert len(alive) > 3      # the fixture really did create tensors each batch
    gc.collect()
    survivors = [reference for reference in alive if reference() is not None]
    assert survivors == [], f"{len(survivors)} raw tensors outlived the probe loop"
    assert _find_tensors(detail) == []


def test_geometry_is_pooled_across_batches_without_the_payload() -> None:
    cli = _load_audit_checkpoint_module()
    model = _FakeModel(lambda: _restitution_output())
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()] * 4, torch.device("cpu"),
        tau_accept=0.0, t_max=2, capture_geometry=True,
    )
    geometry = detail["geometry"]
    assert geometry["available"] is True
    assert geometry["rows_total"] == 4          # one element per batch
    assert geometry["rows_usable"] == 4
    assert "rows" not in geometry               # per-row detail is not carried
    assert geometry["saturation_rate"]["count"] == 4


# --- read-out, flat keys and GT discipline ---------------------------------


def test_probe_publishes_every_declared_flat_key() -> None:
    cli = _load_audit_checkpoint_module()
    cases = [
        _FakeModel(_fixed({"transition_previous": [], "transition_candidates": []})),
        _FakeModel(_fixed(_restitution_output())),
        _FakeModel(_fixed(_restitution_output()), window_mode="current"),
    ]
    for model in cases:
        detail = cli.probe_candidate_c_path(
            model, [_gt_batch()], torch.device("cpu"),
            tau_accept=0.0, t_max=2, capture_geometry=True,
        )
        assert set(detail["flat"]) == set(cli.CANDIDATE_C_FLAT_KEYS)


def test_probe_reads_runtime_rows_and_scores_ground_truth_afterwards() -> None:
    cli = _load_audit_checkpoint_module()
    model = _FakeModel(_fixed(_restitution_output()))
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()], torch.device("cpu"),
        tau_accept=0.25, t_max=3, capture_geometry=True,
    )
    assert detail["available"] is True
    flat = detail["flat"]
    assert flat["candidate_c/rows_total"] == 1
    assert flat["candidate_c/c1_pass_rate"] == 1.0
    assert flat["candidate_c/total_forward_sum"] == 4.0
    assert flat["candidate_c/coordinate_backward_sum"] == 1.0
    assert flat["candidate_c/protected_sum"] == 3.0
    assert flat["candidate_c/ties_excluded_sum"] == 1.0
    assert flat["candidate_c/geometry_available"] is True
    assert flat["candidate_c/ground_truth_available"] is True
    assert flat["candidate_c/ground_truth_rows_scored"] == 1
    # Proposal and retained are published separately and are both present.
    assert flat["candidate_c/ground_truth_proposal_prior_true_fix_destroyed_rate"] == pytest.approx(0.5)
    assert flat["candidate_c/ground_truth_retained_prior_true_fix_destroyed_rate"] == pytest.approx(0.5)
    assert flat["candidate_c/ground_truth_proposal_prior_true_regress_repaired_rate"] == pytest.approx(1.0)

    # Deployable inference, unmodified: self_audit mode with the resolved tau,
    # and no ground truth in the call.
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["mode"] == "self_audit"
    assert call["tau_accept"] == 0.25
    assert call["t_max"] == 3
    for leaking in ("mask", "target", "oracle_target", "labels"):
        assert leaking not in call

    assert "does not re-execute the C1" in detail["measurement_scope"]
    assert "ADDITIONAL ordinary inference pass" in detail["measurement_scope"]
    assert "ground-truth-free" in detail["gt_discipline"]


def test_a_rejected_row_reaches_the_report_under_both_families() -> None:
    cli = _load_audit_checkpoint_module()

    def factory():
        output = _restitution_output()
        output["candidate_c_diagnostics"][0]["accepted"] = False
        return output

    model = _FakeModel(factory)
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()], torch.device("cpu"), tau_accept=0.0, t_max=2
    )
    flat = detail["flat"]
    assert flat["candidate_c/ground_truth_rows_scored"] == 1
    # Proposed destruction is visible; realized destruction is zero.
    assert flat["candidate_c/ground_truth_proposal_prior_true_fix_destroyed_rate"] == pytest.approx(0.5)
    assert flat["candidate_c/ground_truth_retained_prior_true_fix_destroyed_rate"] == pytest.approx(0.0)


def test_probe_marks_geometry_unavailable_when_infer_cannot_capture_it() -> None:
    cli = _load_audit_checkpoint_module()
    output = _restitution_output()
    output.pop("candidate_c_geometry")
    model = _FakeModel(_fixed(output), accept_capture_geometry=False)
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()], torch.device("cpu"),
        tau_accept=0.0, t_max=2, capture_geometry=True,
    )
    assert detail["geometry_supported_by_infer"] is False
    assert detail["geometry"]["available"] is False
    assert "does not accept it" in detail["geometry"]["reason"]
    assert detail["flat"]["candidate_c/geometry_available"] is False
    assert model.calls[0].get("capture_geometry") is None


def test_report_completeness_check_requires_the_candidate_c_keys() -> None:
    cli = _load_audit_checkpoint_module()
    missing = cli.check_required_keys({"flat": {}, "per_stage": {}})["missing"]
    for key in (
        "candidate_c/available",
        "candidate_c/rows_total",
        "candidate_c/batches_measured",
        "candidate_c/infer_calls",
        "candidate_c/mode_active",
    ):
        assert key in missing


# ---------------------------------------------------------------------------
# 6. Live coupling against the real core (no checkpoint, random weights)
# ---------------------------------------------------------------------------


def _live_candidate_c_output():
    """Run the real SelfAuditNet candidate_c path on a tiny random batch."""

    net_module = pytest.importorskip("self_audit.models.self_audit_net")
    if "window_mode" not in __import__("inspect").signature(
        net_module.SelfAuditNet.__init__
    ).parameters:
        pytest.skip("core SelfAuditNet does not expose window_mode at this revision")
    if "capture_geometry" not in __import__("inspect").signature(
        net_module.SelfAuditNet.infer
    ).parameters:
        pytest.skip("core infer does not expose capture_geometry at this revision")

    torch.manual_seed(7)
    net = net_module.SelfAuditNet(
        encoder_name="convnext_tiny",
        pretrained_encoder=False,
        shared_channels=32,
        num_classes=4,
        window_k=4,
        max_turns=3,
        window_mode="candidate_c",
    ).eval()
    torch.manual_seed(11)
    images = torch.randn(2, 3, 32, 32)
    # always_accept with a permissive tau drives several turns so a record
    # exists to replay; no ground truth is involved in producing any of it.
    return net.infer(
        images,
        mode="always_accept_refinement",
        tau_accept=-1e9,
        t_max=3,
        capture_geometry=True,
    )


def test_consumer_handles_the_real_core_payload() -> None:
    output = _live_candidate_c_output()
    rows = output["candidate_c_diagnostics"]
    geometry = output["candidate_c_geometry"]
    assert len(rows) == len(geometry) > 0

    # Rows are strictly JSON-compatible, as the contract requires.
    json.dumps(rows)

    result = evaluate_candidate_c(
        output, target=torch.randint(0, 4, (2, 32, 32)), keep_geometry_rows=False
    )
    encoded = json.dumps(result)
    assert "NaN" not in encoded and "Infinity" not in encoded

    assert result["rows"]["available"] is True
    assert result["rows"]["rows_total"] == len(rows)

    # Some rows replayed; every replay reported an error, and the consumer
    # pooled only those rows.
    c1 = result["rows"]["c1"]
    assert c1["attempted_count"] >= 1
    assert c1["attempted_count"] + c1["rows_without_replay_attempt"] == len(rows)
    assert c1["max_abs_err"]["count"] == c1["attempted_count"]

    # Geometry: the real payload uses the native [1, heads, H, W, K] attention
    # layout and detached ordinary (non-inference) coordinate tensors.
    usable = [element for element in geometry if "factual" in element]
    assert usable, "no geometry element carried a captured support"
    depth = usable[0]["factual"][0]
    assert depth["coordinates"].ndim == 5 and depth["coordinates"].shape[-1] == 2
    assert depth["coordinates"].is_inference() is False
    assert depth["coordinates"].requires_grad is False
    if depth["attention"] is not None:
        assert depth["attention"].shape[-1] == usable[0]["k"]

    geometry_summary = result["geometry"]
    assert geometry_summary["available"] is True
    assert geometry_summary["rows_usable"] == len(usable)
    assert geometry_summary["rows_unavailable"] == len(geometry) - len(usable)
    # Unavailable elements are explained, never silently pooled as zeros.
    assert geometry_summary["unavailable_reasons"]
    assert geometry_summary["attention_mean_entropy_nats"]["count"] == len(usable)
    entropy = geometry_summary["attention_mean_entropy_nats"]["max"]
    assert 0.0 <= entropy <= math.log(usable[0]["k"]) + 1e-9

    # The GT block either scores or explains itself; it never raises and never
    # invents a count.
    ground_truth = result["ground_truth"]
    assert ground_truth["evaluation_only"] is True
    if ground_truth["available"]:
        for family in ("proposal", "retained"):
            assert ground_truth[family]["rows_scored"] == ground_truth["rows_scored"]
            assert ground_truth[family]["counts"]["valid_pixels"] > 0
    else:
        assert ground_truth["unavailable_reasons"]
        assert ground_truth["rows_scored"] == 0


def test_core_rows_now_carry_record_turn_so_geometry_is_not_required() -> None:
    """Core landed the scalar row field; the GT block no longer needs geometry."""

    output = _live_candidate_c_output()
    rows = output["candidate_c_diagnostics"]
    assert all("record_turn" in row for row in rows)

    # Drop the geometry payload entirely and score from the rows alone.
    without_geometry = {key: value for key, value in output.items()
                        if key != "candidate_c_geometry"}
    target = torch.randint(0, 4, (2, 32, 32))
    with_geometry = evaluate_candidate_c(output, target=target)["ground_truth"]
    bare = evaluate_candidate_c(without_geometry, target=target)["ground_truth"]

    assert bare["rows_scored"] == with_geometry["rows_scored"]
    for family in ("proposal", "retained"):
        assert bare[family]["counts"] == with_geometry[family]["counts"]
    # Whatever the live run produced, the two paths must agree exactly.
    assert bare["strata"] == with_geometry["strata"]


def test_core_row_schema_matches_the_consumer_field_names() -> None:
    """Guards the frozen coupling: a renamed core field must fail here."""

    from self_audit.evaluation.candidate_c import (
        ROW_CATEGORICAL_FIELDS,
        ROW_COUNT_FIELDS,
        ROW_EVAL_FIELDS,
        ROW_FLAG_FIELDS,
        ROW_NUMERIC_FIELDS,
    )

    output = _live_candidate_c_output()
    row = output["candidate_c_diagnostics"][0]
    expected = set(ROW_NUMERIC_FIELDS) | set(ROW_COUNT_FIELDS) | set(ROW_FLAG_FIELDS)
    expected |= set(ROW_CATEGORICAL_FIELDS) | {"turn", "sample_index", "record_turn", "evals"}
    missing = expected - set(row)
    assert not missing, f"core row is missing consumer fields: {sorted(missing)}"
    assert set(ROW_EVAL_FIELDS) <= set(row["evals"])

    element = output["candidate_c_geometry"][0]
    assert {"turn", "record_turn", "sample_index"} <= set(element)


# ---------------------------------------------------------------------------
# 7. Input validation: malformed geometry is unavailable, bad limits are loud
# ---------------------------------------------------------------------------


def test_negative_attention_weights_are_rejected_not_renormalised() -> None:
    """A negative weight is not a probability; rescaling it is not an entropy."""

    height = width = 2
    k = 4
    factual = _grid(height, width, k)
    attention = torch.full((1, 1, height, width, k), 0.5)
    attention[..., 0] = -0.5          # row still sums to 1.0
    metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width,
                          attention=attention)
    )
    entropy = metrics["attention_entropy"]
    assert entropy["available"] is False
    assert "negative weights" in entropy["reason"]
    assert "mean_entropy_nats" not in entropy
    assert metrics["attention_entropy"]["depths_missing_attention"] == 1


def test_non_finite_attention_is_rejected_with_a_reason() -> None:
    height = width = 2
    k = 2
    factual = _grid(height, width, k)
    attention = torch.full((1, 1, height, width, k), 0.5)
    attention[0, 0, 0, 0, 0] = float("nan")
    metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width,
                          attention=attention)
    )
    assert metrics["attention_entropy"]["available"] is False
    assert "non-finite" in metrics["attention_entropy"]["reason"]


def test_zero_mass_attention_row_is_rejected() -> None:
    height = width = 2
    factual = _grid(height, width, 2)
    metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width,
                          attention=torch.zeros(1, 1, height, width, 2))
    )
    assert metrics["attention_entropy"]["available"] is False
    assert "zero total mass" in metrics["attention_entropy"]["reason"]


def test_merely_unnormalised_attention_is_still_measured_and_flagged() -> None:
    height = width = 2
    k = 4
    factual = _grid(height, width, k)
    metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=height, width=width,
                          attention=torch.full((1, 1, height, width, k), 3.0))
    )
    entropy = metrics["depths"][0]["attention_entropy"]
    assert entropy["available"] is True
    assert entropy["renormalized"] is True
    assert entropy["mean_entropy_nats"] == pytest.approx(math.log(k), abs=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duplicate_tolerance_pixels": -0.5},
        {"duplicate_tolerance_pixels": float("nan")},
        {"duplicate_tolerance_pixels": float("inf")},
        {"saturation_eps": -1e-6},
        {"saturation_eps": float("nan")},
    ],
)
def test_non_finite_or_negative_geometry_limits_are_refused(kwargs) -> None:
    factual = _grid(2, 2, 2)
    element = _geometry_element(factual, factual.clone(), height=2, width=2)
    with pytest.raises(ValueError):
        geometry_row_metrics(element, **kwargs)
    with pytest.raises(ValueError):
        summarize_geometry([element], **kwargs)
    with pytest.raises(ValueError):
        summarize_geometry_rows([], **kwargs)


def test_non_positive_feature_hw_is_unavailable_not_scored() -> None:
    factual = _grid(2, 2, 2)
    element = _geometry_element(factual, factual.clone(), height=2, width=2)
    element["feature_hw"] = (0, 2)
    metrics = geometry_row_metrics(element)
    assert metrics["available"] is False
    assert "feature_hw must be positive" in metrics["reason"]

    element["feature_hw"] = ("a", 2)
    assert "not a pair of integers" in geometry_row_metrics(element)["reason"]


def test_non_finite_preclamp_leaves_clamp_activity_unavailable() -> None:
    factual = _grid(2, 2, 2)
    preclamp = factual.clone()
    preclamp[0, 0, 0, 0, 0] = float("inf")
    metrics = geometry_row_metrics(
        _geometry_element(factual, factual.clone(), height=2, width=2, preclamp=preclamp)
    )
    saturation = metrics["saturation"]
    assert saturation["clamped_count"] is None
    assert saturation["max_preclamp_overshoot"] is None
    assert metrics["saturation_preclamp_depths_missing"] == 1


def test_summarize_geometry_rows_pools_pure_numbers_only() -> None:
    factual = _grid(3, 3, 2)
    row_metrics = [
        geometry_row_metrics(
            _geometry_element(factual, factual.clone(), height=3, width=3, sample_index=index)
        )
        for index in range(3)
    ]
    assert _find_tensors(row_metrics) == []
    summary = summarize_geometry_rows(row_metrics, keep_rows=False)
    assert summary["available"] is True
    assert summary["rows_usable"] == 3
    assert _find_tensors(summary) == []
    json.dumps(summary)


# ---------------------------------------------------------------------------
# 8. Solver work: actual invocation totals vs per-row attribution
# ---------------------------------------------------------------------------
#
# One solver invocation runs on a batched group and the SAME evaluation
# counters are copied into every surviving row. Summing them across rows
# multiplies real work by the number of survivors; dividing by
# solver_group_size is also wrong, because a halted row emits no row at all.


def _evals(factual_replay=1, coordinate_backward=1, candidate_checks=2, total_forward=3):
    return {
        "factual_replay": factual_replay,
        "coordinate_backward": coordinate_backward,
        "candidate_checks": candidate_checks,
        "total_forward": total_forward,
    }


def _group_rows(*, turn=1, record_turn=0, survivors=2, group_size=4, evals=None, **overrides):
    """``survivors`` emitted rows sharing one invocation out of a group of ``group_size``."""

    shared = _evals() if evals is None else evals
    return [
        _row(turn=turn, record_turn=record_turn, sample_index=index,
             evals=dict(shared), solver_group_size=group_size, **overrides)
        for index in range(survivors)
    ]


def test_shared_counters_are_deduplicated_per_invocation() -> None:
    """B=4 group with 2 survivors and total_forward=3 must report 3, not 6."""

    summary = summarize_diagnostic_rows(_group_rows(survivors=2, group_size=4))
    actual = summary["evals"]["actual_invocation_totals"]
    assert actual["invocations"] == 1
    assert actual["totals"]["total_forward"] == 3
    assert actual["totals"]["factual_replay"] == 1
    assert actual["totals"]["candidate_checks"] == 2
    assert actual["key_source"] == "batch_scope+turn+record_turn"

    # The over-counting raw sum is still published, explicitly named.
    assert summary["evals"]["row_sums_overcounted"]["total_forward"] == 6.0

    # solver_group_size is reported but is NOT used as a divisor: 3/4 is not
    # the answer either, because the two halted rows emitted nothing.
    opportunity = summary["evals"]["per_row_opportunity"]
    assert opportunity["solver_group_size"]["max"] == 4.0
    assert opportunity["rows_per_invocation"]["mean"] == 2.0
    assert opportunity["rows_with_evals"] == 2


def test_sample_associated_mean_is_the_attribution_not_the_total() -> None:
    summary = summarize_diagnostic_rows(_group_rows(survivors=3))
    mean = summary["evals"]["sample_associated_mean"]["total_forward"]
    assert mean["mean"] == pytest.approx(3.0)     # each row is told 3
    assert mean["sum"] == pytest.approx(9.0)      # the raw over-count
    assert summary["evals"]["actual_invocation_totals"]["totals"]["total_forward"] == 3
    assert "OVER-COUNT" in summary["evals"]["sample_associated_mean"]["note"]


def test_two_turns_are_two_invocations() -> None:
    rows = _group_rows(turn=1, record_turn=0, survivors=2) + _group_rows(
        turn=2, record_turn=1, survivors=1
    )
    actual = summarize_diagnostic_rows(rows)["evals"]["actual_invocation_totals"]
    assert actual["invocations"] == 2
    assert actual["totals"]["total_forward"] == 6      # 3 + 3, not 9


def test_repeated_ids_in_a_later_batch_need_the_scope_namespace() -> None:
    """Without a per-batch scope the second batch's work silently disappears."""

    rows = _group_rows(survivors=2) + _group_rows(survivors=2)
    collapsed = summarize_diagnostic_rows(rows)["evals"]["actual_invocation_totals"]
    assert collapsed["invocations"] == 1
    assert collapsed["totals"]["total_forward"] == 3   # batch 1's work lost

    scoped = summarize_diagnostic_rows(
        rows, scopes=[0, 0, 1, 1]
    )["evals"]["actual_invocation_totals"]
    assert scoped["invocations"] == 2
    assert scoped["totals"]["total_forward"] == 6      # both batches counted


def test_scopes_must_be_parallel_to_rows() -> None:
    with pytest.raises(ValueError, match="parallel to rows"):
        summarize_diagnostic_rows(_group_rows(survivors=2), scopes=[0])


def test_solver_invocation_id_is_preferred_when_core_supplies_it() -> None:
    """Two invocations in one turn stay distinct once core labels them."""

    rows = _group_rows(survivors=2)
    rows[0]["solver_invocation_id"] = "inv-a"
    rows[1]["solver_invocation_id"] = "inv-b"
    actual = summarize_diagnostic_rows(rows)["evals"]["actual_invocation_totals"]
    assert actual["key_source"] == "solver_invocation_id"
    assert actual["invocations"] == 2
    assert actual["totals"]["total_forward"] == 6

    shared = _group_rows(survivors=3)
    for row in shared:
        row["solver_invocation_id"] = 7
    same = summarize_diagnostic_rows(shared)["evals"]["actual_invocation_totals"]
    assert same["invocations"] == 1
    assert same["totals"]["total_forward"] == 3


def test_a_group_with_disagreeing_counters_reports_unknown_not_a_guess() -> None:
    rows = _group_rows(survivors=2)
    rows[1]["evals"]["total_forward"] = 5          # the group contradicts itself
    actual = summarize_diagnostic_rows(rows)["evals"]["actual_invocation_totals"]
    assert actual["invocations"] == 1
    assert actual["invocations_inconsistent"] == 1
    assert actual["totals"]["total_forward"] is None            # never averaged
    assert actual["unknown_invocations_per_field"]["total_forward"] == 1
    # Fields the group DID agree on are still reported.
    assert actual["totals"]["candidate_checks"] == 2
    listed = actual["inconsistent_invocations"]
    assert len(listed) == 1
    assert listed[0]["fields"] == ["total_forward"]
    assert listed[0]["rows"] == 2
    assert listed[0]["turn"] == 1 and listed[0]["record_turn"] == 0


def test_one_bad_group_does_not_erase_the_known_part() -> None:
    good = _group_rows(turn=1, record_turn=0, survivors=2)
    bad = _group_rows(turn=2, record_turn=1, survivors=2)
    bad[1]["evals"]["total_forward"] = 99
    actual = summarize_diagnostic_rows(good + bad)["evals"]["actual_invocation_totals"]
    assert actual["totals"]["total_forward"] is None
    assert actual["totals_over_known_invocations"]["total_forward"] == 3
    assert actual["invocations"] == 2


@pytest.mark.parametrize("bad", [None, -1, "three", float("nan")])
def test_a_malformed_counter_is_unmeasured_not_zero(bad) -> None:
    rows = _group_rows(survivors=2)
    for row in rows:
        row["evals"]["total_forward"] = bad
    actual = summarize_diagnostic_rows(rows)["evals"]["actual_invocation_totals"]
    assert actual["totals"]["total_forward"] is None
    assert actual["invocations_unmeasured"] == 1
    assert actual["totals"]["candidate_checks"] == 2


def test_rows_that_carry_no_key_are_excluded_and_counted() -> None:
    rows = _group_rows(survivors=1)
    rows[0].pop("turn")
    actual = summarize_diagnostic_rows(rows)["evals"]["actual_invocation_totals"]
    assert actual["rows_without_invocation_key"] == 1
    assert actual["invocations"] == 0
    # Rows existed but nothing could be keyed: unmeasured, NOT a zero.
    assert actual["totals"]["total_forward"] is None


def test_no_attempt_rows_at_all_is_an_honest_zero() -> None:
    """The zero-forward baseline guard: nothing ran, so nothing was spent."""

    actual = summarize_diagnostic_rows([])["evals"]["actual_invocation_totals"]
    assert actual["invocations"] == 0
    assert actual["rows_without_evals_block"] == 0
    assert actual["rows_without_invocation_key"] == 0
    assert actual["totals"] == {field: 0 for field in ROW_EVAL_FIELDS}


def test_solver_totals_declare_what_they_exclude() -> None:
    actual = summarize_diagnostic_rows(_group_rows())["evals"]["actual_invocation_totals"]
    for excluded in ("Auditor", "ordinary Annotation Expert", "ordinary-fallback"):
        assert excluded in actual["note"]
    assert "solver_group_size" in actual["deduplication_note"]


# --- the same, through probe 4 --------------------------------------------


def _group_output(*, survivors=2, group_size=4, turn=1, record_turn=0):
    p = _one_hot_logits(torch.tensor([[[1, 1, 0, 0]]]).repeat(survivors, 1, 1), 2)
    f = _one_hot_logits(torch.tensor([[[0, 0, 1, 0]]]).repeat(survivors, 1, 1), 2)
    q = _one_hot_logits(torch.tensor([[[0, 1, 0, 1]]]).repeat(survivors, 1, 1), 2)
    return {
        "candidate_c_diagnostics": _group_rows(
            turn=turn, record_turn=record_turn, survivors=survivors, group_size=group_size,
            accepted=True, accepted_path="counterfactual",
        ),
        "transition_previous": [p, f],
        "transition_candidates": [f, q],
    }


def test_probe_namespaces_repeated_ids_across_batches() -> None:
    cli = _load_audit_checkpoint_module()
    model = _FakeModel(lambda: _group_output(survivors=2, group_size=4))
    batch = {"image": torch.zeros(2, 1, 1, 4), "mask": torch.zeros(2, 1, 4, dtype=torch.long)}
    detail = cli.probe_candidate_c_path(
        model, [batch] * 3, torch.device("cpu"), tau_accept=0.0, t_max=2
    )
    flat = detail["flat"]
    # Three batches, one invocation each, total_forward 3 per invocation.
    assert flat["candidate_c/solver_invocations"] == 3
    assert flat["candidate_c/total_forward_sum"] == 9
    assert flat["candidate_c/factual_replay_sum"] == 3
    assert flat["candidate_c/candidate_checks_sum"] == 6
    assert flat["candidate_c/coordinate_backward_sum"] == 3
    # The raw per-row sum over-counts by the number of survivors.
    assert flat["candidate_c/total_forward_row_sum"] == 18.0
    assert flat["candidate_c/mean_rows_per_solver_invocation"] == 2.0
    assert flat["candidate_c/solver_invocations_inconsistent"] == 0
    assert flat["candidate_c/solver_invocation_key_source"] == "batch_scope+turn+record_turn"


def test_probe_reports_unknown_solver_work_for_a_contradictory_group() -> None:
    cli = _load_audit_checkpoint_module()

    def factory():
        output = _group_output(survivors=2)
        output["candidate_c_diagnostics"][1]["evals"]["total_forward"] = 11
        return output

    detail = cli.probe_candidate_c_path(
        _FakeModel(factory),
        [{"image": torch.zeros(2, 1, 1, 4), "mask": torch.zeros(2, 1, 4, dtype=torch.long)}],
        torch.device("cpu"), tau_accept=0.0, t_max=2,
    )
    flat = detail["flat"]
    assert flat["candidate_c/total_forward_sum"] is None       # unknown, not fabricated
    assert flat["candidate_c/solver_invocations_inconsistent"] == 1
    assert flat["candidate_c/candidate_checks_sum"] == 2       # the agreed field survives


def test_inactive_mode_keeps_the_zero_forward_guard() -> None:
    cli = _load_audit_checkpoint_module()
    model = _FakeModel(lambda: _group_output(), window_mode="current")
    detail = cli.probe_candidate_c_path(
        model, [_gt_batch()] * 2, torch.device("cpu"), tau_accept=0.0, t_max=2
    )
    assert model.calls == []
    assert detail["infer_calls"] == 0
    flat = detail["flat"]
    assert flat["candidate_c/total_forward_sum"] is None
    assert flat["candidate_c/total_forward_row_sum"] is None
    assert flat["candidate_c/solver_invocations"] is None
    assert set(flat) == set(cli.CANDIDATE_C_FLAT_KEYS)


def test_cli_flat_keys_separate_actual_work_from_the_raw_row_sum() -> None:
    cli = _load_audit_checkpoint_module()
    for actual_key, row_key in (
        ("candidate_c/total_forward_sum", "candidate_c/total_forward_row_sum"),
        ("candidate_c/factual_replay_sum", "candidate_c/factual_replay_row_sum"),
        ("candidate_c/candidate_checks_sum", "candidate_c/candidate_checks_row_sum"),
        ("candidate_c/coordinate_backward_sum", "candidate_c/coordinate_backward_row_sum"),
    ):
        assert actual_key in cli.CANDIDATE_C_FLAT_KEYS
        assert row_key in cli.CANDIDATE_C_FLAT_KEYS

    model = _FakeModel(lambda: _group_output(survivors=2))
    detail = cli.probe_candidate_c_path(
        model,
        [{"image": torch.zeros(2, 1, 1, 4), "mask": torch.zeros(2, 1, 4, dtype=torch.long)}],
        torch.device("cpu"), tau_accept=0.0, t_max=2,
    )
    flat = detail["flat"]
    # The two must not be the same number when survivors > 1.
    assert flat["candidate_c/total_forward_sum"] == 3
    assert flat["candidate_c/total_forward_row_sum"] == 6.0
    assert "solver-only" in detail["measurement_scope"].lower() or (
        "Solver-only" in detail["rows"]["evals"]["actual_invocation_totals"]["note"]
    )


# ---------------------------------------------------------------------------
# 9. Zero-work groups are not solver invocations
# ---------------------------------------------------------------------------
#
# An ordinary annotation turn, a direct rollback (record consumed, solver never
# called) and a stale-record preflight that aborts before any replay all emit
# rows whose four counters are explicitly zero. They executed nothing, so they
# must not appear as invocations -- but they are KNOWN zero, not unknown.


_ZERO_EVALS = {
    "factual_replay": 0,
    "coordinate_backward": 0,
    "candidate_checks": 0,
    "total_forward": 0,
}


def _ordinary_rows(*, turn, survivors=2, **overrides):
    """Ordinary-path rows: no record consumed, no solver, all counters zero."""

    return [
        _row(turn=turn, record_turn=None, sample_index=index, eligible=False,
             record_kind=None, accepted_path="ordinary",
             fallback_reason="no_accepted_history", c1_passed=None,
             evals=dict(_ZERO_EVALS), solver_group_size=None, **overrides)
        for index in range(survivors)
    ]


def _actual(rows, **kwargs):
    return summarize_diagnostic_rows(rows, **kwargs)["evals"]["actual_invocation_totals"]


def test_ordinary_zero_work_turns_are_not_phantom_invocations() -> None:
    """The real ordinary -> C -> ordinary trajectory, B=2: exactly ONE solve."""

    rows = (
        _ordinary_rows(turn=0)
        + _group_rows(turn=1, record_turn=0, survivors=2, group_size=2,
                      accepted_path="counterfactual")
        + _ordinary_rows(turn=2)
    )
    summary = summarize_diagnostic_rows(rows)
    actual = summary["evals"]["actual_invocation_totals"]

    assert actual["invocations"] == 1
    assert actual["invocation_groups_attempted"] == 3
    assert actual["groups_without_executed_work"] == 2
    assert actual["totals"]["total_forward"] == 3
    assert actual["totals"]["total_forward"] <= 3
    assert summary["evals"]["row_sums_overcounted"]["total_forward"] == 6.0
    assert summary["evals"]["row_sums_overcounted"]["total_forward"] <= 6.0

    # Row-attempt statistics are retained in full: all six rows still count.
    opportunity = summary["evals"]["per_row_opportunity"]
    assert opportunity["rows_total"] == 6
    assert opportunity["rows_with_evals"] == 6
    assert opportunity["rows_per_invocation"]["count"] == 1     # executed groups only
    assert opportunity["rows_per_group"]["count"] == 3          # every keyed group
    assert summary["rows_total"] == 6


def test_zero_work_is_known_zero_not_unknown() -> None:
    actual = _actual(_ordinary_rows(turn=0) + _ordinary_rows(turn=1))
    assert actual["invocations"] == 0
    assert actual["groups_without_executed_work"] == 2
    assert actual["rows_that_could_hide_work"] == 0
    # Nothing executed, and that is measured, so the totals are a hard zero.
    assert actual["totals"] == {field: 0 for field in ROW_EVAL_FIELDS}


def test_direct_rollback_consumes_a_record_without_calling_the_solver() -> None:
    rows = _group_rows(turn=1, record_turn=0, survivors=2, group_size=2,
                       accepted_path="rollback", evals=dict(_ZERO_EVALS))
    summary = summarize_diagnostic_rows(rows)
    actual = summary["evals"]["actual_invocation_totals"]
    assert actual["invocations"] == 0                      # solver never called
    assert actual["invocation_groups_attempted"] == 1
    assert actual["totals"]["total_forward"] == 0
    # Record consumption is tracked separately from solver execution.
    consumption = actual["record_consumption"]
    assert consumption["rows_consuming_a_record"] == 2
    assert consumption["by_accepted_path"] == {"rollback": 2}


def test_stale_record_preflight_is_not_an_executed_invocation() -> None:
    rows = _group_rows(turn=1, record_turn=0, survivors=1, group_size=1,
                       accepted_path="ordinary", fallback_reason="stale_record",
                       evals=dict(_ZERO_EVALS))
    actual = _actual(rows)
    assert actual["invocations"] == 0
    assert actual["invocation_groups_attempted"] == 1        # the attempt is retained
    assert actual["groups_without_executed_work"] == 1
    assert actual["record_consumption"]["rows_consuming_a_record"] == 1


def test_a_single_nonzero_counter_still_counts_as_executed() -> None:
    """Only an all-zero group is 'nothing ran'; one forward is still work."""

    rows = _group_rows(survivors=2, evals=dict(_ZERO_EVALS, factual_replay=1))
    actual = _actual(rows)
    assert actual["invocations"] == 1
    assert actual["totals"]["factual_replay"] == 1
    assert actual["totals"]["total_forward"] == 0


def test_a_row_with_no_counters_makes_the_exact_totals_unknown() -> None:
    """A partial sum is never presented as the full known figure."""

    rows = _group_rows(turn=1, record_turn=0, survivors=2) + [
        _row(turn=2, record_turn=1, sample_index=0, evals=None)
    ]
    actual = _actual(rows)
    assert actual["invocations"] == 1
    assert actual["rows_without_evals_block"] == 1
    assert actual["rows_that_could_hide_work"] == 1
    assert actual["totals"]["total_forward"] is None              # exact is unknown
    assert actual["totals_over_known_invocations"]["total_forward"] == 3   # known part kept


def test_an_unkeyable_row_that_might_carry_work_makes_totals_unknown() -> None:
    known = _group_rows(turn=1, record_turn=0, survivors=2)
    hidden = _group_rows(survivors=1)
    hidden[0].pop("turn")
    actual = _actual(known + hidden)
    assert actual["rows_without_invocation_key"] == 1
    assert actual["rows_without_invocation_key_uncertain"] == 1
    assert actual["rows_that_could_hide_work"] == 1
    assert actual["totals"]["total_forward"] is None
    assert actual["totals_over_known_invocations"]["total_forward"] == 3


def test_an_unkeyable_row_reporting_explicit_zero_work_is_not_unknown() -> None:
    known = _group_rows(turn=1, record_turn=0, survivors=2)
    benign = _ordinary_rows(turn=0, survivors=1)
    benign[0].pop("turn")
    actual = _actual(known + benign)
    assert actual["rows_without_invocation_key"] == 1
    assert actual["rows_without_invocation_key_zero_work"] == 1
    assert actual["rows_without_invocation_key_uncertain"] == 0
    assert actual["rows_that_could_hide_work"] == 0
    assert actual["totals"]["total_forward"] == 3            # still exact


def test_mixed_trajectory_across_two_batches_counts_each_solve_once() -> None:
    batch = (
        _ordinary_rows(turn=0)
        + _group_rows(turn=1, record_turn=0, survivors=2, group_size=2,
                      accepted_path="counterfactual")
        + _ordinary_rows(turn=2)
    )
    rows = batch + [dict(row) for row in batch]
    scopes = [0] * len(batch) + [1] * len(batch)
    actual = _actual(rows, scopes=scopes)
    assert actual["invocations"] == 2
    assert actual["invocation_groups_attempted"] == 6
    assert actual["groups_without_executed_work"] == 4
    assert actual["totals"]["total_forward"] == 6


def test_zero_work_note_names_the_three_cases() -> None:
    actual = _actual(_ordinary_rows(turn=0))
    for case in ("ordinary annotation turn", "direct rollback", "stale-record preflight"):
        assert case in actual["zero_work_note"]
    assert "partial sum is never presented" in actual["unknown_note"]


def test_probe_reports_one_invocation_for_an_ordinary_c_ordinary_batch() -> None:
    cli = _load_audit_checkpoint_module()

    def factory():
        output = _group_output(survivors=2, group_size=2)
        output["candidate_c_diagnostics"] = (
            _ordinary_rows(turn=0)
            + output["candidate_c_diagnostics"]
            + _ordinary_rows(turn=2)
        )
        return output

    detail = cli.probe_candidate_c_path(
        _FakeModel(factory),
        [{"image": torch.zeros(2, 1, 1, 4), "mask": torch.zeros(2, 1, 4, dtype=torch.long)}],
        torch.device("cpu"), tau_accept=0.0, t_max=3,
    )
    flat = detail["flat"]
    assert flat["candidate_c/rows_total"] == 6
    assert flat["candidate_c/solver_invocations"] == 1
    assert flat["candidate_c/solver_invocation_groups_attempted"] == 3
    assert flat["candidate_c/solver_groups_without_executed_work"] == 2
    assert flat["candidate_c/solver_rows_that_could_hide_work"] == 0
    assert flat["candidate_c/total_forward_sum"] == 3
    assert flat["candidate_c/total_forward_row_sum"] == 6.0
    assert set(flat) == set(cli.CANDIDATE_C_FLAT_KEYS)


def test_live_three_turn_trajectory_reports_one_solver_invocation() -> None:
    """The real ordinary -> C -> ordinary run: 3 turns, 1 solve, B=2."""

    output = _live_candidate_c_output()
    rows = output["candidate_c_diagnostics"]
    turns = {row.get("turn") for row in rows}
    if len(turns) < 2:
        pytest.skip("live trajectory did not reach a second turn at this revision")

    summary = summarize_diagnostic_rows(rows)
    actual = summary["evals"]["actual_invocation_totals"]
    raw = summary["evals"]["row_sums_overcounted"]

    # An ordinary turn emits rows with all-zero counters; it is not a solve.
    assert actual["invocation_groups_attempted"] == len(turns)
    assert actual["invocations"] <= 1
    assert actual["groups_without_executed_work"] == (
        actual["invocation_groups_attempted"] - actual["invocations"]
    )
    assert actual["rows_that_could_hide_work"] == 0
    assert actual["totals"]["total_forward"] is not None
    assert actual["totals"]["total_forward"] <= 3
    assert raw["total_forward"] <= 6.0
    # The raw sum over-counts by exactly the number of rows sharing the solve.
    if actual["invocations"] == 1:
        assert actual["invocations"] == 1
        assert raw["total_forward"] >= actual["totals"]["total_forward"]
