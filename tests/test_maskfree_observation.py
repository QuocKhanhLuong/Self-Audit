"""Focused checks for the mask-free observation model (W1).

Four checks, matching the W1 acceptance criteria:

1. target blindness plus positive score sensitivity;
2. partition sensitivity (mask shuffle and all-background counterexamples);
3. matched fitting budgets across a bank, and the no-refit-at-scoring rule;
4. frozen fitted identity under caller mutation, and recipe-mismatch rejection.

Run with the package importable under its canonical name, so that no second
copy of the module is created under a different top-level name::

    PYTHONPATH=src python3 -m pytest tests/test_maskfree_observation.py -v

Fixtures are CPU synthetic phantoms. They are software evidence about the
implemented equations only; they say nothing about real cardiac data, which this
checkout does not contain.
"""
from __future__ import annotations

import pytest
import torch

from self_audit_maskfree.contracts import (
    FittingView,
    ScoringView,
    hypothesis_from_labels,
)
from self_audit_maskfree.observation import ObservationContractError, ObservationModel

SIZE = 32
STUDY = "synthetic_study"
UNIT = "synthetic_study_z04"
PARTITION = "partition_hash_abc"


def _true_labels() -> torch.Tensor:
    """Concentric phantom in the frozen semantic order 0=BG, 1=RV, 2=MYO, 3=LV."""
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, dtype=torch.float32),
        torch.arange(SIZE, dtype=torch.float32),
        indexing="ij",
    )
    radius = ((ys - (SIZE - 1) / 2).pow(2) + (xs - (SIZE - 1) / 2).pow(2)).sqrt()
    labels = torch.zeros(SIZE, SIZE, dtype=torch.long)
    labels[radius < 12.0] = 1
    labels[radius < 8.0] = 2
    labels[radius < 4.0] = 3
    return labels


#: Well separated region appearances so that the correct partition is genuinely
#: the better explanation, not a coin flip.
REGION_MEANS = torch.tensor([-1.0, 0.0, 1.0, 2.0])


def _phantom(seed: int = 0) -> torch.Tensor:
    labels = _true_labels()
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(SIZE, SIZE, generator=generator) * 0.2
    return REGION_MEANS[labels] + noise


def _role_masks() -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic checkerboard-of-blocks split into fit and select roles."""
    block = torch.arange(SIZE) // 8
    block_y, block_x = torch.meshgrid(block, block, indexing="ij")
    select = ((block_y + block_x) % 4) == 0
    fit = ~select
    return fit, select


def _views(image: torch.Tensor | None = None) -> tuple[FittingView, ScoringView]:
    image = _phantom() if image is None else image
    fit_mask, select_mask = _role_masks()
    fit_image = torch.where(fit_mask, image, torch.zeros_like(image)).unsqueeze(0)
    context = fit_image.repeat(3, 1, 1)
    fitting = FittingView(
        image=fit_image,
        support=fit_mask,
        context=context,
        study_id=STUDY,
        unit_id=UNIT,
        partition_id=PARTITION,
    )
    scoring = ScoringView(
        image=torch.where(select_mask, image, torch.zeros_like(image)).unsqueeze(0),
        support=select_mask,
        study_id=STUDY,
        unit_id=UNIT,
        role="select",
        partition_id=PARTITION,
    )
    return fitting, scoring


def test_target_blind_fit_and_positive_score_sensitivity() -> None:
    """Fitted parameters ignore the scored intensities; the score still reacts to them.

    The negative control is the pair of fits: changing only the withheld
    intensities must leave every fitted parameter bit-identical. The positive
    control is the pair of scores: the same frozen fit must score a clean
    withheld image better than a corrupted one, otherwise the score is inert and
    could not carry evidence at all.
    """
    clean = _phantom(seed=0)
    corrupted = clean.clone()
    _, select_mask = _role_masks()
    corrupted[select_mask] = -corrupted[select_mask] * 3.0

    fitting_clean, scoring_clean = _views(clean)
    fitting_corrupt, scoring_corrupt = _views(corrupted)

    labels = _true_labels()
    hypothesis = hypothesis_from_labels(labels, "incumbent", "phantom_truth")
    model = ObservationModel()

    fitted_clean = model.fit(hypothesis, fitting_clean)
    fitted_corrupt = model.fit(hypothesis, fitting_corrupt)

    # Target blindness: withheld intensities changed drastically, the fit did not.
    assert fitted_clean.parameters == fitted_corrupt.parameters
    assert fitted_clean.fit_score.nll_sum == fitted_corrupt.fit_score.nll_sum

    # Positive control: the frozen fit does respond to what it is scored against.
    good = model.score(fitted_clean, scoring_clean)
    bad = model.score(fitted_clean, scoring_corrupt)
    assert good.available and bad.available
    assert good.count == bad.count == int(select_mask.sum())
    assert good.total is not None and bad.total is not None
    assert good.total < bad.total

    # Empty scoring support is unavailable with a reason, never a zero score.
    empty = ScoringView(
        image=torch.zeros(1, SIZE, SIZE),
        support=torch.zeros(SIZE, SIZE, dtype=torch.bool),
        study_id=STUDY,
        unit_id=UNIT,
        role="select",
        partition_id=PARTITION,
    )
    unavailable = model.score(fitted_clean, empty)
    assert unavailable.available is False
    assert unavailable.normalized_nll is None and unavailable.total is None
    assert unavailable.reason

    # Provenance is enforced: a fit may not be scored against another unit.
    foreign = ScoringView(
        image=scoring_clean.image,
        support=scoring_clean.support,
        study_id=STUDY,
        unit_id="other_unit",
        role="select",
        partition_id=PARTITION,
    )
    with pytest.raises(ObservationContractError):
        model.score(fitted_clean, foreign)


def test_partition_sensitivity_against_degenerate_candidates() -> None:
    """The partition must actually drive the predictive distribution.

    A shuffled partition and an all-background partition are the two standard
    counterexamples: if either matched the true partition's selection score, the
    likelihood would be ignoring the partition and the whole evidence argument
    would collapse.
    """
    fitting, scoring = _views()
    model = ObservationModel()
    labels = _true_labels()

    truth = model.fit(hypothesis_from_labels(labels, "truth", "phantom_truth"), fitting)

    shuffled_flat = labels.reshape(-1)[torch.randperm(labels.numel(),
                                                      generator=torch.Generator().manual_seed(7))]
    shuffled = model.fit(
        hypothesis_from_labels(shuffled_flat.reshape(SIZE, SIZE), "shuffled", "mask_shuffle"),
        fitting,
    )
    background = model.fit(
        hypothesis_from_labels(torch.zeros_like(labels), "all_bg", "all_background"),
        fitting,
    )

    truth_score = model.score(truth, scoring)
    shuffled_score = model.score(shuffled, scoring)
    background_score = model.score(background, scoring)

    assert truth_score.total < shuffled_score.total
    assert truth_score.total < background_score.total
    # The shuffle also pays a large complexity penalty; check the raw predictive
    # term alone separates them, so the win is not an artifact of the penalty.
    assert truth_score.normalized_nll < shuffled_score.normalized_nll
    assert truth_score.normalized_nll < background_score.normalized_nll
    # All-background leaves three regions empty; that must be reported, not hidden.
    assert background.parameters["fallback_region_count"] == 3
    assert background.parameters["fallback_regions"] == [1, 2, 3]


def test_matched_budgets_and_no_refit_during_scoring() -> None:
    """Every bank member gets identical capacity and steps, and scoring is read-only."""
    fitting, scoring = _views()
    labels = _true_labels()
    bank = [
        hypothesis_from_labels(labels, "incumbent", "grouping"),
        hypothesis_from_labels(torch.zeros_like(labels), "all_bg", "degenerate"),
        hypothesis_from_labels((labels + 1) % 4, "permuted", "semantic_alternative"),
        hypothesis_from_labels(labels.roll(3, dims=1), "shifted", "boundary_edit"),
    ]
    model = ObservationModel(background_modes=2)
    fitted = [model.fit(hypothesis, fitting) for hypothesis in bank]

    assert {f.fitting_steps for f in fitted} == {model.max_iterations}
    assert model.capacity == 5  # 3 foreground components + 2 nuisance background modes
    assert {f.capacity for f in fitted} == {model.capacity}
    assert all(len(f.parameters["components"]) == model.capacity for f in fitted)
    assert all(f.partition_id == PARTITION and f.study_id == STUDY for f in fitted)
    # No shared mutable nuisance state between candidates.
    assert len({id(f.parameters) for f in fitted}) == len(fitted)
    assert fitted[0].parameters["components"] != fitted[1].parameters["components"]

    # Scoring is pure: identical results, untouched parameters, no step growth.
    before = [dict(f.parameters) for f in fitted]
    first = [model.score(f, scoring) for f in fitted]
    second = [model.score(f, scoring) for f in fitted]
    assert [f.parameters for f in fitted] == before
    assert all(f.fitting_steps == model.max_iterations for f in fitted)
    assert [s.total for s in first] == [s.total for s in second]

    # A fit made at one capacity may not be scored by a model of another.
    mismatched = ObservationModel(background_modes=1)
    with pytest.raises(ObservationContractError):
        mismatched.score(fitted[0], scoring)

    # Temporal support is not implied by the spatial implementation.
    cine = FittingView(
        image=fitting.image,
        support=fitting.support,
        context=fitting.context,
        study_id=STUDY,
        unit_id=UNIT,
        protocol="cine_predictive",
        partition_id=PARTITION,
    )
    with pytest.raises(ObservationContractError):
        model.fit(bank[0], cine)


def test_frozen_fit_identity_and_recipe_mismatch() -> None:
    """A stored fit keeps the partition it was fitted on, and refuses foreign recipes.

    Both failures this guards against are silent ones. If the fit held the
    caller's ``Hypothesis`` by reference, a challenger round that rewrote labels
    in place would retroactively change which partition every earlier stored fit
    appears to describe, while its appearance parameters still belonged to the
    old one. If ``score`` only checked capacity, two candidates fitted under
    different variance floors or density families would be ranked against each
    other as though the comparison were matched.
    """
    fitting, scoring = _views()
    labels = _true_labels()
    model = ObservationModel()

    caller_hypothesis = hypothesis_from_labels(labels.clone(), "incumbent", "grouping")
    fitted = model.fit(caller_hypothesis, fitting)
    baseline = model.score(fitted, scoring)

    # The caller mutates its own object in place after fitting.
    caller_hypothesis.labels.zero_()
    caller_hypothesis.probabilities.zero_()
    caller_hypothesis.probabilities[0] = 1.0
    caller_hypothesis.validity.zero_()
    caller_hypothesis.metadata["prior_penalty"] = 9.0

    assert not torch.equal(fitted.hypothesis.labels, caller_hypothesis.labels)
    assert torch.equal(fitted.hypothesis.labels, labels)
    assert "prior_penalty" not in fitted.hypothesis.metadata
    after = model.score(fitted, scoring)
    assert after.total == baseline.total
    assert after.prior == baseline.prior == 0.0

    # Mutating the frozen snapshot itself is detected rather than silently rescored.
    fitted.hypothesis.labels.zero_()
    with pytest.raises(ObservationContractError, match="mutated after fitting"):
        model.score(fitted, scoring)

    # Recipe mismatch: same capacity, different likelihood knobs, refused.
    reference = model.fit(hypothesis_from_labels(labels, "incumbent", "grouping"), fitting)
    assert model.score(reference, scoring).total is not None
    for other in (
        ObservationModel(variance_floor=0.2),
        ObservationModel(beta=0.5),
        ObservationModel(distribution="student_t"),
        ObservationModel(bias_ridge=0.5),
        ObservationModel(max_iterations=3),
    ):
        assert other.capacity == model.capacity
        with pytest.raises(ObservationContractError, match="recipe mismatch"):
            other.score(reference, scoring)

    # Probabilities that do not sum to one over the four classes are rejected.
    broken = hypothesis_from_labels(labels, "broken", "grouping")
    broken.probabilities[0] += 0.5
    with pytest.raises(ObservationContractError, match="sum to one"):
        model.fit(broken, fitting)
