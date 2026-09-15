"""Focused checks for mask-free hypotheses, ontology and the evidence auditor (W2).

Focused checks matching the W2 acceptance criteria:

1. concrete anatomical role rules, missing anatomy, and ambiguity that is
   reported rather than guessed or silently mapped to background;
2. tied semantic likelihood, incumbent retention, and the anti-starvation rule
   that a rejected edit in an early round does not end the budget;
3. a non-trivial end-to-end bank/audit path plus the sealed-observation firewall;
4. content-addressed bank identity and explicit orientation-rule execution.

Fixtures are CPU synthetic phantoms. They are software evidence about the
implemented rules only; this checkout contains no real cardiac data, so nothing
here says anything about ACDC or M&Ms behaviour.
"""
from __future__ import annotations

import pytest
import torch

from self_audit_maskfree.auditor import AuditError, audit_bank
from self_audit_maskfree.contracts import (
    FittingView,
    ScoringView,
    hypothesis_from_labels,
)
from self_audit_maskfree.hypotheses import BANK_SIZE, bank_identity, generate_bank
from self_audit_maskfree.observation import ObservationModel
from self_audit_maskfree.ontology import (
    BG,
    LV,
    MYO,
    RV,
    UNRESOLVED_VALIDITY,
    apply_alternative,
    resolve_roles,
)

SIZE = 32
STUDY = "synthetic_study"
UNIT = "synthetic_study_z04"
PARTITION = "partition_hash_abc"
#: Well separated appearances so the phantom is genuinely explainable.
REGION_MEANS = torch.tensor([-1.0, 0.5, 0.0, 1.0])


class CountingObservationModel(ObservationModel):
    """W1 model spy for proving regional work scores frozen fits only."""

    def __init__(self) -> None:
        super().__init__()
        self.fit_calls = 0
        self.score_calls = 0
        self.score_support_counts: list[int] = []

    def fit(self, hypothesis, fitting_view):  # type: ignore[no-untyped-def]
        self.fit_calls += 1
        return super().fit(hypothesis, fitting_view)

    def score(self, fitted, scoring_view):  # type: ignore[no-untyped-def]
        self.score_calls += 1
        self.score_support_counts.append(int(scoring_view.support.sum().item()))
        return super().score(fitted, scoring_view)


def _radius() -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, dtype=torch.float32),
        torch.arange(SIZE, dtype=torch.float32),
        indexing="ij",
    )
    return ((ys - (SIZE - 1) / 2).pow(2) + (xs - (SIZE - 1) / 2).pow(2)).sqrt()


def _concentric_partition() -> torch.Tensor:
    """Anonymous groups: 0 outside, 1 annulus, 2 inner disc. No anatomy yet."""
    radius = _radius()
    partition = torch.zeros(SIZE, SIZE, dtype=torch.long)
    partition[radius < 9.0] = 1
    partition[radius < 5.0] = 2
    return partition


def _two_blob_partition() -> torch.Tensor:
    """Two congruent, non-enclosing blobs: the rules genuinely cannot name them."""
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, dtype=torch.float32),
        torch.arange(SIZE, dtype=torch.float32),
        indexing="ij",
    )
    partition = torch.zeros(SIZE, SIZE, dtype=torch.long)
    left = ((ys - 16.0).pow(2) + (xs - 10.0).pow(2)).sqrt() < 4.0
    right = ((ys - 16.0).pow(2) + (xs - 22.0).pow(2)).sqrt() < 4.0
    partition[left] = 1
    partition[right] = 2
    return partition


def _role_masks() -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic block split into fitting and selection roles."""
    block = torch.arange(SIZE) // 4
    block_y, block_x = torch.meshgrid(block, block, indexing="ij")
    select = ((block_y + block_x) % 4) == 0
    return ~select, select


def _views(image: torch.Tensor, role: str = "select") -> tuple[FittingView, ScoringView]:
    fit_mask, select_mask = _role_masks()
    fit_image = torch.where(fit_mask, image, torch.zeros_like(image)).unsqueeze(0)
    fitting = FittingView(
        image=fit_image,
        support=fit_mask,
        context=fit_image.repeat(3, 1, 1),
        study_id=STUDY,
        unit_id=UNIT,
        partition_id=PARTITION,
    )
    scoring = ScoringView(
        image=torch.where(select_mask, image, torch.zeros_like(image)).unsqueeze(0),
        support=select_mask,
        study_id=STUDY,
        unit_id=UNIT,
        role=role,
        partition_id=PARTITION,
    )
    return fitting, scoring


def _phantom_from(labels: torch.Tensor, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return REGION_MEANS[labels] + torch.randn(SIZE, SIZE, generator=generator) * 0.15


def _anatomical_labels() -> torch.Tensor:
    """Concentric MYO ring around an LV cavity, plus an adjacent RV cavity."""
    radius = _radius()
    labels = torch.zeros(SIZE, SIZE, dtype=torch.long)
    labels[radius < 9.0] = MYO
    labels[radius < 5.0] = LV
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, dtype=torch.float32),
        torch.arange(SIZE, dtype=torch.float32),
        indexing="ij",
    )
    rv = ((ys - 16.0).pow(2) + (xs - 25.0).pow(2)).sqrt() < 4.0
    labels[rv] = RV
    return labels


# ----------------------------------------------------------------------
# check 1: concrete role rules, missing anatomy, reported ambiguity
# ----------------------------------------------------------------------
def test_role_rules_resolve_enclosure_report_ambiguity_and_never_write_background() -> None:
    partition = _concentric_partition()
    image = _phantom_from(
        torch.where(partition == 2, torch.full_like(partition, LV),
                    torch.where(partition == 1, torch.full_like(partition, MYO), partition))
    )
    fitting, _ = _views(image)

    resolved = resolve_roles(partition, fitting, candidate_id="c", source="test")
    trace = resolved.metadata["ontology_trace"]

    # Enclosure names the ring MYO and the cavity LV. No GT was consulted.
    assert "R3_enclosure" in trace["rules_fired"]
    assert resolved.labels[16, 16].item() == LV
    assert resolved.labels[16, 16 + 7].item() == MYO
    assert resolved.labels[0, 0].item() == BG
    assert not resolved.semantic_unresolved
    # R7: RV is simply absent here, and an absent class costs no prior penalty.
    assert trace["absent_roles"] == ["RV"]
    assert resolved.metadata["prior_penalty"] == 0.0
    assert trace["prior_violations"] == {}

    # Ambiguous geometry: two congruent non-enclosing blobs.
    ambiguous_partition = _two_blob_partition()
    ambiguous_image = _phantom_from(
        torch.where(ambiguous_partition > 0, torch.full_like(ambiguous_partition, LV),
                    torch.zeros_like(ambiguous_partition))
    )
    ambiguous_fit, _ = _views(ambiguous_image)
    ambiguous = resolve_roles(ambiguous_partition, ambiguous_fit, candidate_id="a", source="test")
    ambiguous_trace = ambiguous.metadata["ontology_trace"]

    assert ambiguous.semantic_unresolved
    assert ambiguous.alternatives, "an unresolved naming must record its competitors"
    assert not ambiguous_trace["orientation_available"]
    assert ambiguous_trace["orientation_rule_executed"] is False
    assert ambiguous_trace["orientation_resolution"] == "unresolved_metadata_unavailable"
    assert "orientation_metadata_absent_left_right_unresolved" in ambiguous_trace["notes"]
    foreground = ambiguous_partition > 0
    # Unresolved pixels keep a draft anatomical label and lose their validity.
    # They are never rewritten to background, which would delete the structure.
    assert bool((ambiguous.labels[foreground] != BG).all())
    assert float(ambiguous.validity[foreground].max().item()) == UNRESOLVED_VALIDITY
    assert float(ambiguous.validity[~foreground].min().item()) == 1.0

    # An all-background partition names no anatomy, so no pixel may supervise.
    empty = resolve_roles(torch.zeros_like(partition), fitting, candidate_id="e", source="test")
    assert empty.semantic_unresolved
    assert float(empty.validity.max().item()) == 0.0
    assert sorted(empty.metadata["ontology_trace"]["absent_roles"]) == ["LV", "MYO", "RV"]


def test_orientation_rule_requires_and_executes_spatial_metadata() -> None:
    """Metadata availability is separate from executing the spatial rule."""
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, dtype=torch.float32),
        torch.arange(SIZE, dtype=torch.float32),
        indexing="ij",
    )
    partition = torch.zeros(SIZE, SIZE, dtype=torch.long)
    # Three non-enclosing groups: the intensity prior drafts roles, then the
    # affine-derived +x direction must swap the contradictory blood-pool order.
    left = ((ys - 16.0).pow(2) + (xs - 9.0).pow(2)).sqrt() < 3.0
    right = ((ys - 16.0).pow(2) + (xs - 23.0).pow(2)).sqrt() < 3.0
    middle = ((ys - 8.0).pow(2) + (xs - 16.0).pow(2)).sqrt() < 3.0
    partition[left] = 3
    partition[right] = 2
    partition[middle] = 1
    anonymous_intensity = torch.zeros_like(ys)
    anonymous_intensity[partition == 1] = 0.0  # darkest draft: MYO
    anonymous_intensity[partition == 2] = 1.0  # brightest draft: RV before rule
    anonymous_intensity[partition == 3] = 0.5  # middle draft: LV before rule
    fitting, _ = _views(anonymous_intensity)
    oriented = FittingView(
        image=fitting.image,
        support=fitting.support,
        context=fitting.context,
        study_id=fitting.study_id,
        unit_id=fitting.unit_id,
        protocol=fitting.protocol,
        metadata={
            "view": "short_axis",
            "inplane_left_axis": "+x",
            "geometry_valid": True,
        },
        partition_id=fitting.partition_id,
    )
    resolved = resolve_roles(partition, oriented, candidate_id="oriented", source="test")
    trace = resolved.metadata["ontology_trace"]
    assert trace["orientation_available"] is True
    assert trace["orientation_rule_executed"] is True
    assert trace["orientation_resolution"] == "resolved_blood_pool_order"
    assert "R5b_orientation" in trace["rules_fired"]
    assert resolved.labels[16, 23].item() == LV
    assert resolved.labels[16, 9].item() == RV

    # With only two groups, metadata can be valid while the rule is inapplicable
    # because there is no drafted RV role. The trace must say that explicitly.
    two = resolve_roles(_two_blob_partition(), oriented, candidate_id="two", source="test")
    two_trace = two.metadata["ontology_trace"]
    assert two_trace["orientation_available"] is True
    assert two_trace["orientation_rule_executed"] is False
    assert two_trace["orientation_resolution"] == "not_applicable_missing_blood_pool_role"
    assert "R5b_orientation" not in two_trace["rules_fired"]


# ----------------------------------------------------------------------
# check 2: tied likelihood, incumbent retention, no starvation after rejection
# ----------------------------------------------------------------------
def test_tied_semantics_incumbent_retention_and_no_starvation_after_rejection() -> None:
    partition = _two_blob_partition()
    image = _phantom_from(
        torch.where(partition > 0, torch.full_like(partition, LV), torch.zeros_like(partition))
    )
    fitting, selection = _views(image)
    model = ObservationModel()

    draft = resolve_roles(partition, fitting, candidate_id="draft", source="test")
    permuted = apply_alternative(draft, partition, draft.alternatives[-1], "permuted")

    # A semantic permutation of the same geometry must not gain or lose evidence.
    assert not torch.equal(draft.labels, permuted.labels)
    assert permuted.metadata["prior_penalty"] == draft.metadata["prior_penalty"]
    draft_score = model.score(model.fit(draft, fitting), selection)
    permuted_score = model.score(model.fit(permuted, fitting), selection)
    assert draft_score.total == pytest.approx(permuted_score.total, abs=1e-9)

    # Bank: incumbent, two deliberately degenerate edits, then the tied permutation.
    truth = _anatomical_labels()
    phantom = _phantom_from(truth)
    fitting, selection = _views(phantom)
    incumbent = hypothesis_from_labels(truth, "incumbent", "test")
    all_background = hypothesis_from_labels(torch.zeros_like(truth), "all_background", "test")
    shuffled = hypothesis_from_labels(
        truth.reshape(-1)[torch.randperm(truth.numel(), generator=torch.Generator().manual_seed(0))]
        .reshape(truth.shape),
        "shuffled",
        "test",
    )
    tied = hypothesis_from_labels(truth.clone(), "tied", "test")

    result = audit_bank(
        [incumbent, all_background, shuffled, tied], fitting, selection, model
    )
    assert result.trace["fits_performed"] == 4, "every preregistered slot must be spent"
    assert result.trace["rounds_requested"] == 2
    assert len(result.trace["round_slots"]) == 2
    assert all(result.trace["round_slots"]), "rounds must partition the bank, not multiply it"
    # Degenerate edits are rejected and the incumbent is retained on the tie.
    assert result.trace["accepted"] is False
    assert result.selected.candidate_id == "incumbent"
    assert result.trace["rejection_is_edit_scoped"] is True
    assert result.trace["learning_sample_retained"] is True
    rejected = {row["candidate_id"] for row in result.trace["rejected_edits"]}
    assert {"all_background", "shuffled", "tied"} <= rejected

    # A genuine improvement banked in the *last* round is still reached, which is
    # the concrete anti-starvation check: two earlier rejections did not stop it.
    eroded = truth.clone()
    eroded[_radius() < 6.5] = MYO  # incumbent loses most of the LV cavity
    weak = hypothesis_from_labels(eroded, "weak_incumbent", "test")
    improved = audit_bank([weak, all_background, shuffled, incumbent], fitting, selection, model)
    assert improved.trace["fits_performed"] == 4
    assert improved.trace["accepted"] is True
    assert improved.selected.candidate_id == "incumbent"
    assert improved.trace["improvement_nats_per_pixel"] >= 0.001
    assert improved.trace["evaluation_order"][-1]["round"] == 2


# ----------------------------------------------------------------------
# check 3: non-trivial bank/audit path and the sealed-observation firewall
# ----------------------------------------------------------------------
def test_bank_audit_path_and_sealed_verification_firewall() -> None:
    phantom = _phantom_from(_anatomical_labels())
    fitting, selection = _views(phantom)
    model = CountingObservationModel()

    bank = generate_bank(fitting, seed=42)
    assert len(bank) == BANK_SIZE
    sources = [candidate.source for candidate in bank]
    assert sources[0] == "grouping"
    assert sources[1] == "anatomical_initializer"
    assert sources[2] == "boundary_edit"
    assert sources[3] in ("split_merge", "semantic_alternative")
    # A real search: at least one challenger genuinely differs from the incumbent.
    assert any(
        bool((candidate.labels != bank[0].labels).any()) for candidate in bank[1:]
    ), "a bank of identical candidates is not a search"
    for candidate in bank:
        assert candidate.metadata["prior_penalty"] >= 0.0
        assert "ontology_trace" in candidate.metadata
        assert candidate.metadata["generation"]["seed"] == 42

    # Determinism: the same fitting view must rebuild the identical bank.
    assert all(
        torch.equal(a.labels, b.labels)
        for a, b in zip(bank, generate_bank(fitting, seed=42))
    )

    result = audit_bank(bank, fitting, selection, model)
    assert result.trace["bank_size"] == BANK_SIZE
    assert result.trace["fits_performed"] == BANK_SIZE
    assert result.validity.shape == fitting.support.shape
    assert float(result.validity.min()) >= 0.0 and float(result.validity.max()) <= 1.0
    assert result.regional_margin.shape == fitting.support.shape
    assert set(result.trace["class_pixel_counts"]) == {"BG", "RV", "MYO", "LV"}
    assert result.trace["selection_observations_consumed"] == int(selection.support.sum())
    assert result.trace["primary_score_calls"] == BANK_SIZE
    assert result.trace["regional_score_calls"] > 0
    assert result.trace["regional_refits"] == 0
    assert result.trace["score_calls_total"] == (
        result.trace["primary_score_calls"] + result.trace["regional_score_calls"]
    )
    assert model.fit_calls == result.trace["fits_performed"]
    assert model.score_calls == result.trace["score_calls_total"]
    assert len(result.trace["regional_score_counts"]) == result.trace["regional_score_calls"]
    # Regional scores use the same O_select support for all candidates within a
    # region, and the support is narrower than the unit-level primary score.
    assert all(
        count > 0 and count < int(selection.support.sum())
        for count in result.trace["regional_score_counts"]
    )
    # Every candidate was fitted at the same capacity and iteration budget.
    assert len({fit.capacity for fit in result.fitted}) == 1
    assert len({fit.fitting_steps for fit in result.fitted}) == 1
    # Validity may never resurrect a pixel the ontology marked unresolved.
    assert bool((result.validity[result.selected.validity == 0.0] == 0.0).all())

    # An unchallenged incumbent is not a confident incumbent.
    single = audit_bank([bank[0]], fitting, selection, model)
    assert single.trace["search_inconclusive"] is True
    assert float(single.regional_margin.abs().max().item()) == 0.0

    # Firewall: sealed verification observations are refused outright.
    verify = ScoringView(
        image=selection.image,
        support=selection.support,
        study_id=STUDY,
        unit_id=UNIT,
        role="verify",
        partition_id=PARTITION,
    )
    with pytest.raises(AuditError, match="role='select' only"):
        audit_bank(bank, fitting, verify, model)
    # And a bank from another observation partition cannot be scored here.
    foreign = ScoringView(
        image=selection.image,
        support=selection.support,
        study_id=STUDY,
        unit_id=UNIT,
        role="select",
        partition_id="other_partition",
    )
    with pytest.raises(AuditError, match="observation partitions"):
        audit_bank(bank, fitting, foreign, model)


def test_bank_identity_is_content_based_and_latency_independent() -> None:
    """The bank join covers partition/input/features/candidates, never latency."""
    fitting, _ = _views(_phantom_from(_anatomical_labels()))
    features = torch.randn(1, 4, SIZE, SIZE, requires_grad=True)
    bank = generate_bank(fitting, features=features, seed=42)
    before = bank_identity(fitting, bank, features=features, seed=999)
    for candidate in bank:
        candidate.metadata["generation_latency"] = {"bank_wall_seconds": 999999.0}
    after = bank_identity(fitting, bank, features=features, seed=123)
    assert after["bank_id"] == before["bank_id"]
    assert after["partition_hash"]
    assert after["fit_input_hash"]
    assert after["feature_hash"]
    assert after["candidate_content_hash"]
    assert all(candidate.metadata["generation"]["features_detached_cpu"] for candidate in bank)
