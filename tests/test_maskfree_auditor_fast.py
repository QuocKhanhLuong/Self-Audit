"""Focused scalar-vs-bulk W2 audit equivalence checks.

These are CPU synthetic software checks only.  They intentionally compare the
independent public ``audit_bank`` reference against ``audit_banks``' flattened
fit/score path and verify that custom ObservationModel subclasses stay scalar.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from self_audit_maskfree.auditor import (
    MAX_REGION_SCORE_CALLS,
    audit_bank,
    audit_banks,
)
from self_audit_maskfree.contracts import (
    EvidenceScore,
    FittedHypothesis,
    FittingView,
    ScoringView,
    hypothesis_from_labels,
)
from self_audit_maskfree.observation import ObservationModel

SIZE = 16


def _labels() -> torch.Tensor:
    labels = torch.zeros(SIZE, SIZE, dtype=torch.long)
    labels[2:14, 2:14] = 1
    labels[5:11, 5:11] = 2
    labels[7:9, 7:9] = 3
    return labels


def _views(unit_id: str = "unit-0") -> tuple[FittingView, ScoringView]:
    labels = _labels()
    image = torch.where(labels == 0, -1.0, torch.where(labels == 1, 0.0,
                         torch.where(labels == 2, 1.0, 2.0)))
    by, bx = torch.meshgrid(torch.arange(SIZE) // 4, torch.arange(SIZE) // 4, indexing="ij")
    selection_support = ((by + bx) % 4) == 0
    fitting_support = ~selection_support
    fitting_image = torch.where(fitting_support, image, torch.zeros_like(image)).unsqueeze(0)
    fitting = FittingView(
        image=fitting_image,
        support=fitting_support,
        context=fitting_image.repeat(3, 1, 1),
        study_id="audit-fast",
        unit_id=unit_id,
        partition_id="audit-fast-partition",
    )
    selection = ScoringView(
        image=torch.where(selection_support, image, torch.zeros_like(image)).unsqueeze(0),
        support=selection_support,
        study_id="audit-fast",
        unit_id=unit_id,
        role="select",
        partition_id="audit-fast-partition",
    )
    return fitting, selection


def _bank() -> list:
    labels = _labels()
    return [
        hypothesis_from_labels(labels, "incumbent", "initializer"),
        hypothesis_from_labels(torch.zeros_like(labels), "all-bg", "all-bg"),
        hypothesis_from_labels(labels.roll(2, dims=1), "shifted", "boundary-edit"),
        hypothesis_from_labels((labels + 1) % 4, "permuted", "semantic-alternative"),
    ]


def _logical_trace(trace: dict) -> dict:
    keys = (
        "bank_size", "rounds_requested", "round_slots", "fits_performed",
        "primary_score_calls", "score_calls_total", "regional_score_calls",
        "regional_score_observation_count", "regional_score_counts", "regional_refits",
        "accepted", "selected_candidate_id", "selected_index", "search_inconclusive",
        "semantic_unresolved", "semantic_alternatives", "class_pixel_counts", "class_absent",
    )
    return {key: trace[key] for key in keys}


def test_audit_banks_matches_scalar_decision_and_logical_trace() -> None:
    fitting, selection = _views()
    bank = _bank()
    scalar = audit_bank(bank, fitting, selection, ObservationModel())
    bulk = audit_banks([bank], [fitting], [selection], ObservationModel())[0]

    assert _logical_trace(bulk.trace) == _logical_trace(scalar.trace)
    assert [entry["candidate_id"] for entry in bulk.trace["evaluation_order"]] == [
        entry["candidate_id"] for entry in scalar.trace["evaluation_order"]
    ]
    assert [entry["round"] for entry in bulk.trace["evaluation_order"]] == [
        entry["round"] for entry in scalar.trace["evaluation_order"]
    ]
    assert [entry["cached"] for entry in bulk.trace["evaluation_order"]] == [
        entry["cached"] for entry in scalar.trace["evaluation_order"]
    ]
    assert bulk.selected.candidate_id == scalar.selected.candidate_id
    assert torch.equal(bulk.selected.labels, scalar.selected.labels)
    assert torch.equal(bulk.selected.probabilities, scalar.selected.probabilities)
    assert torch.equal(bulk.selected.validity, scalar.selected.validity)
    torch.testing.assert_close(bulk.regional_margin, scalar.regional_margin, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(bulk.validity, scalar.validity, rtol=2e-11, atol=2e-11)
    assert bulk.trace["physical_work"]["scope"] == "unit"
    assert "fit_many_calls" not in bulk.trace["physical_work"]


def test_audit_banks_matches_an_accepted_edit_and_semantic_trace() -> None:
    """The bulk path also preserves an actual accepted challenger decision."""
    size = 32
    row, col = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    labels = torch.zeros(size, size, dtype=torch.long)
    labels[(col >= size // 4) & (col < size // 2)] = 1
    labels[(col >= size // 2) & (col < 3 * size // 4)] = 2
    labels[col >= 3 * size // 4] = 3
    image = torch.where(
        labels == 0,
        torch.tensor(-2.0),
        torch.where(
            labels == 1,
            torch.tensor(-0.5),
            torch.where(labels == 2, torch.tensor(0.5), torch.tensor(1.5)),
        ),
    )
    selection_support = ((row // 4 + col // 4) % 4) == 0
    fitting_support = ~selection_support
    fitting = FittingView(
        image=torch.where(fitting_support, image, torch.zeros_like(image)).unsqueeze(0),
        support=fitting_support,
        context=torch.where(fitting_support, image, torch.zeros_like(image))
        .unsqueeze(0)
        .repeat(3, 1, 1),
        study_id="audit-accepted",
        unit_id="unit-accepted",
        partition_id="audit-accepted-partition",
    )
    selection = ScoringView(
        image=torch.where(selection_support, image, torch.zeros_like(image)).unsqueeze(0),
        support=selection_support,
        study_id="audit-accepted",
        unit_id="unit-accepted",
        role="select",
        partition_id="audit-accepted-partition",
    )
    bank = [
        hypothesis_from_labels(torch.zeros_like(labels), "incumbent", "bad"),
        hypothesis_from_labels(labels, "good", "good"),
        hypothesis_from_labels(labels.roll(1, dims=1), "shifted", "boundary-edit"),
        hypothesis_from_labels(torch.ones_like(labels), "all-rv", "all-rv"),
    ]
    scalar = audit_bank(bank, fitting, selection, ObservationModel())
    bulk = audit_banks([bank], [fitting], [selection], ObservationModel())[0]
    assert scalar.trace["accepted"] is bulk.trace["accepted"] is True
    assert scalar.selected.candidate_id == bulk.selected.candidate_id == "good"
    scalar_rejected, bulk_rejected = scalar.trace["rejected_edits"], bulk.trace["rejected_edits"]
    assert [row["candidate_id"] for row in scalar_rejected] == [
        row["candidate_id"] for row in bulk_rejected
    ]
    assert [row["reason"] for row in scalar_rejected] == [row["reason"] for row in bulk_rejected]
    for scalar_row, bulk_row in zip(scalar_rejected, bulk_rejected):
        assert math.isclose(
            scalar_row["gain_nats_per_pixel"], bulk_row["gain_nats_per_pixel"],
            rel_tol=2e-11, abs_tol=2e-11,
        )
    assert scalar.trace["search_inconclusive"] is bulk.trace["search_inconclusive"] is False
    assert _logical_trace(scalar.trace) == _logical_trace(bulk.trace)
    assert [candidate.candidate_id for candidate in scalar.bank] == [
        candidate.candidate_id for candidate in bulk.bank
    ]
    torch.testing.assert_close(scalar.regional_margin, bulk.regional_margin, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(scalar.validity, bulk.validity, rtol=2e-11, atol=2e-11)


def test_audit_banks_flattens_multiple_units_and_keeps_unit_order() -> None:
    fitting0, selection0 = _views("unit-0")
    fitting1, selection1 = _views("unit-1")
    banks = [_bank(), _bank()]
    results = audit_banks(
        banks, [fitting0, fitting1], [selection0, selection1], ObservationModel()
    )
    references = [
        audit_bank(banks[0], fitting0, selection0, ObservationModel()),
        audit_bank(banks[1], fitting1, selection1, ObservationModel()),
    ]
    assert [result.trace["unit_id"] for result in results] == ["unit-0", "unit-1"]
    assert [result.selected.candidate_id for result in results] == [
        reference.selected.candidate_id for reference in references
    ]
    for result, reference in zip(results, references):
        assert _logical_trace(result.trace) == _logical_trace(reference.trace)
        assert result.trace["physical_work"]["scope"] == "unit"
        assert "fit_many_calls" not in result.trace["physical_work"]


class _ScalarOnlyObservation(ObservationModel):
    """Custom scalar double used to guard the callback compatibility path."""

    def __init__(self, values: dict[str, float]) -> None:
        super().__init__(max_iterations=1)
        self.values = values
        self.calls: list[str] = []

    def fit(self, hypothesis, fitting_view):  # type: ignore[no-untyped-def]
        self.calls.append(f"fit:{hypothesis.candidate_id}")
        count = int(fitting_view.support.sum().item())
        score = EvidenceScore(count, count, 1.0, 0.0, 0.0, 1.0, "fit")
        return FittedHypothesis(
            hypothesis=hypothesis,
            parameters={},
            fit_score=score,
            fitting_steps=1,
            capacity=self.capacity,
            study_id=fitting_view.study_id,
            unit_id=fitting_view.unit_id,
            partition_id=fitting_view.partition_id,
        )

    def score(self, fitted, scoring_view):  # type: ignore[no-untyped-def]
        self.calls.append(f"score:{fitted.hypothesis.candidate_id}:{scoring_view.role}")
        value = self.values[fitted.hypothesis.candidate_id]
        count = int(scoring_view.support.sum().item())
        return EvidenceScore(
            value * count, count, value if count else None, 0.0, 0.0,
            value if count else None, scoring_view.role,
            available=bool(count), reason=None if count else "empty",
        )


def test_custom_subclass_with_inherited_bulk_methods_stays_scalar() -> None:
    fitting, selection = _views()
    bank = _bank()[:3]
    model = _ScalarOnlyObservation({"incumbent": 1.0, "all-bg": 0.8, "shifted": 0.7})
    result = audit_banks([bank], [fitting], [selection], model)[0]
    assert result.selected.candidate_id == "shifted"
    assert not any(name.startswith(("fit_many", "score_many")) for name in model.calls)
    assert result.trace["physical_work"]["mode"] == "scalar"


def test_scalar_threshold_and_first_candidate_tie_boundary() -> None:
    fitting, selection = _views()
    bank = _bank()[:3]
    # Binary-exact 0.125 gain is accepted at equality and the first equal-gain
    # challenger wins; a nextafter perturbation below the threshold is rejected
    # so the later exact challenger can still be selected.
    exact = {"incumbent": 1.0, "all-bg": 0.875, "shifted": 0.875}
    result = audit_bank(
        bank, fitting, selection, _ScalarOnlyObservation(exact), improvement_threshold=0.125
    )
    assert result.selected.candidate_id == "all-bg"
    below = {
        "incumbent": 1.0,
        "all-bg": torch.nextafter(torch.tensor(0.875), torch.tensor(1.0)).item(),
        "shifted": 0.875,
    }
    result = audit_bank(
        bank, fitting, selection, _ScalarOnlyObservation(below), improvement_threshold=0.125
    )
    assert result.selected.candidate_id == "shifted"


def test_regional_midbudget_exhaustion_matches_scalar_and_bulk() -> None:
    """A many-component partition hits the unchanged regional budget guards."""
    size = 64
    labels = (torch.arange(size * size).reshape(size, size) % 2).long()
    image = labels.float()
    by, bx = torch.meshgrid(
        torch.arange(size) // 4, torch.arange(size) // 4, indexing="ij"
    )
    selection_support = ((by + bx) % 4) == 0
    fitting_support = ~selection_support
    fitting = FittingView(
        image=torch.where(fitting_support, image, torch.zeros_like(image)).unsqueeze(0),
        support=fitting_support,
        context=torch.where(fitting_support, image, torch.zeros_like(image))
        .unsqueeze(0)
        .repeat(3, 1, 1),
        study_id="audit-budget",
        unit_id="unit-budget",
        partition_id="audit-budget-partition",
    )
    selection = ScoringView(
        image=torch.where(selection_support, image, torch.zeros_like(image)).unsqueeze(0),
        support=selection_support,
        study_id="audit-budget",
        unit_id="unit-budget",
        role="select",
        partition_id="audit-budget-partition",
    )
    bank = [
        hypothesis_from_labels(labels, "incumbent", "checkerboard"),
        hypothesis_from_labels(torch.zeros_like(labels), "all-bg", "all-bg"),
        hypothesis_from_labels(torch.ones_like(labels), "all-rv", "all-rv"),
        hypothesis_from_labels(labels.roll(1, dims=0), "shifted", "boundary-edit"),
    ]
    scalar = audit_bank(bank, fitting, selection, ObservationModel())
    bulk = audit_banks([bank], [fitting], [selection], ObservationModel())[0]

    assert scalar.selected.candidate_id == bulk.selected.candidate_id == "incumbent"
    assert scalar.trace["search_notes"]["regions_over_budget"] > 0
    assert scalar.trace["search_notes"]["score_calls"] <= MAX_REGION_SCORE_CALLS
    assert scalar.trace["search_notes"]["regions_over_budget"] == bulk.trace["search_notes"]["regions_over_budget"]
    assert scalar.trace["regional_score_calls"] == bulk.trace["regional_score_calls"]
    assert scalar.trace["regional_score_counts"] == bulk.trace["regional_score_counts"]
    torch.testing.assert_close(scalar.regional_margin, bulk.regional_margin, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(scalar.validity, bulk.validity, rtol=2e-11, atol=2e-11)
