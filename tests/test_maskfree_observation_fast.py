"""Focused checks for the W1 batched observation and prepared-score paths.

These are CPU synthetic fixtures and software-equivalence checks only.  They do
not claim real ACDC/M&Ms, CUDA, clinical, or throughput evidence.  The scalar
``ObservationModel.fit``/``score`` methods remain the independent reference;
the tests deliberately compare the new bulk paths against those methods.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch

from self_audit_maskfree.contracts import (
    FittingView,
    ScoringView,
    hypothesis_from_labels,
)
from self_audit_maskfree.observation import (
    _Component,
    _batch_log_mixture,
    _log_density,
    ObservationContractError,
    ObservationModel,
    PreparedScore,
)

SIZE = 16
STUDY = "fast_synthetic_study"
UNIT = "fast_synthetic_unit"
PARTITION = "fast_partition"


def _labels() -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, dtype=torch.float32),
        torch.arange(SIZE, dtype=torch.float32),
        indexing="ij",
    )
    radius = ((ys - (SIZE - 1) / 2).pow(2) + (xs - (SIZE - 1) / 2).pow(2)).sqrt()
    labels = torch.zeros(SIZE, SIZE, dtype=torch.long)
    labels[radius < 6.5] = 1
    labels[radius < 4.0] = 2
    labels[radius < 2.0] = 3
    return labels


def _views(image: torch.Tensor | None = None) -> tuple[FittingView, ScoringView]:
    labels = _labels()
    if image is None:
        image = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)[labels]
        image = image + torch.randn(
            image.shape, generator=torch.Generator().manual_seed(7)
        ) * 0.03
    blocks = torch.arange(SIZE) // 4
    by, bx = torch.meshgrid(blocks, blocks, indexing="ij")
    select_support = ((by + bx) % 4) == 0
    fit_support = ~select_support
    fit_image = torch.where(fit_support, image, torch.zeros_like(image)).unsqueeze(0)
    fitting = FittingView(
        image=fit_image,
        support=fit_support,
        context=fit_image.repeat(3, 1, 1),
        study_id=STUDY,
        unit_id=UNIT,
        partition_id=PARTITION,
    )
    scoring = ScoringView(
        image=torch.where(select_support, image, torch.zeros_like(image)).unsqueeze(0),
        support=select_support,
        study_id=STUDY,
        unit_id=UNIT,
        role="select",
        partition_id=PARTITION,
    )
    return fitting, scoring


def _restricted(view: ScoringView, region: torch.Tensor) -> ScoringView:
    support = view.support & region
    return ScoringView(
        image=torch.where(support, view.image[0], torch.zeros_like(view.image[0])).unsqueeze(0),
        support=support,
        study_id=view.study_id,
        unit_id=view.unit_id,
        role=view.role,
        partition_id=view.partition_id,
    )


def _bank() -> list:
    labels = _labels()
    return [
        hypothesis_from_labels(labels, "truth", "truth"),
        hypothesis_from_labels(torch.zeros_like(labels), "all_bg", "all_bg"),
        hypothesis_from_labels(labels.roll(2, dims=1), "shifted", "shifted"),
        hypothesis_from_labels((labels + 1) % 4, "permuted", "permuted"),
    ]


def _mixture_components(batch: int, max_modes: int) -> list[tuple[_Component, ...]]:
    """Deterministic fitted-like components for direct density-path checks."""
    result: list[tuple[_Component, ...]] = []
    for index in range(batch):
        components = [
            _Component(
                region=0,
                mean=-0.75 + 0.01 * index,
                variance=0.21 + 0.002 * index,
                log_weight=0.0 if max_modes == 1 else math.log(0.65),
                fallback=False,
            )
        ]
        if max_modes == 2:
            components.append(
                _Component(
                    region=0,
                    mean=-0.10 + 0.01 * index,
                    variance=0.47 + 0.003 * index,
                    log_weight=math.log(0.35),
                    fallback=False,
                )
            )
        for region in range(1, 4):
            components.append(
                _Component(
                    region=region,
                    mean=0.55 * region + 0.01 * index,
                    variance=0.18 + 0.03 * region + 0.001 * index,
                    log_weight=0.0,
                    fallback=False,
                )
            )
        result.append(tuple(components))
    return result


def _explicit_region_reference(
    residual: torch.Tensor,
    labels: torch.Tensor,
    components: list[tuple[_Component, ...]],
    distribution: str,
) -> torch.Tensor:
    """Independent four-region reference mirroring the frozen scalar helper."""
    output = torch.full_like(residual, float("nan"))
    for batch_index, candidate_components in enumerate(components):
        for region in range(4):
            mask = labels[batch_index] == region
            if not bool(mask.any()):
                continue
            region_components = [
                component for component in candidate_components if component.region == region
            ]
            values = residual[batch_index][mask]
            stacked = torch.stack(
                [
                    _log_density(values, component.mean, component.variance, distribution)
                    + component.log_weight
                    for component in region_components
                ]
            )
            output[batch_index, mask] = torch.logsumexp(stacked, dim=0)
    if torch.isnan(output).any():
        raise ObservationContractError("unscored pixels remain in the mixture density")
    return output


@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("max_modes", [1, 2])
@pytest.mark.parametrize("distribution", ["gaussian", "student_t"])
def test_batch_log_mixture_gather_matches_explicit_regions(
    batch: int, max_modes: int, distribution: str
) -> None:
    """Gathered hard-label density is bitwise equal to the four-region reference."""
    generator = torch.Generator().manual_seed(7000 + batch * 10 + max_modes)
    components = _mixture_components(batch, max_modes)

    all_labels = torch.arange(17, dtype=torch.long).remainder(4).repeat(batch, 1)
    all_residual = torch.randn((batch, 17), generator=generator, dtype=torch.float64)
    missing_labels = torch.arange(17, dtype=torch.long).remainder(3).repeat(batch, 1)
    missing_residual = torch.randn((batch, 17), generator=generator, dtype=torch.float64)

    # A strided [B,N] view exercises the same tiny/non-contiguous shapes used
    # by score_many without changing the component fixture or dtype.
    residual_base = torch.randn((batch, 10), generator=generator, dtype=torch.float64)
    labels_base = torch.arange(10, dtype=torch.long).remainder(4).repeat(batch, 1)
    tiny_residual = residual_base[:, ::2]
    tiny_labels = labels_base[:, ::2]

    for residual, labels in (
        (all_residual, all_labels),
        (missing_residual, missing_labels),
        (tiny_residual, tiny_labels),
    ):
        expected = _explicit_region_reference(residual, labels, components, distribution)
        actual = _batch_log_mixture(residual, labels, components, distribution, max_modes)
        assert torch.equal(actual, expected)
        assert actual.shape == residual.shape
        assert actual.dtype == torch.float64
        assert torch.isfinite(actual).all()


def test_batch_log_mixture_preserves_validation_and_nan_guards() -> None:
    components = _mixture_components(batch=1, max_modes=2)
    residual = torch.tensor([[0.0, 0.2, -0.4, 1.1]], dtype=torch.float64)
    labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    # A missing mixture mode is represented by -inf and must still produce a
    # finite density for regions whose labels do not use that mode.
    result = _batch_log_mixture(residual, labels, components, "gaussian", 2)
    assert torch.isfinite(result).all()

    omitted_region = [components[0][:-1]]
    with pytest.raises(ObservationContractError, match="omit a semantic region"):
        _batch_log_mixture(residual, labels, omitted_region, "gaussian", 2)

    too_many_modes = [
        components[0]
        + (
            _Component(
                region=0,
                mean=0.9,
                variance=0.4,
                log_weight=math.log(0.1),
                fallback=False,
            ),
        )
    ]
    with pytest.raises(ObservationContractError, match="exceeds model capacity"):
        _batch_log_mixture(residual, labels, too_many_modes, "gaussian", 2)

    with pytest.raises(ObservationContractError, match="unscored pixels"):
        _batch_log_mixture(
            residual, torch.tensor([[0, 1, 4, 3]], dtype=torch.long), components, "gaussian", 2
        )
    with pytest.raises(ObservationContractError, match="unscored pixels"):
        nan_residual = residual.clone()
        nan_residual[0, 1] = float("nan")
        _batch_log_mixture(
            nan_residual,
            labels,
            components,
            "gaussian",
            2,
        )


def _large_unit(index: int) -> tuple[FittingView, ScoringView, torch.Tensor]:
    """A modest production-grid (128x128) unit with a unit-specific role map."""
    size = 128
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    radius = ((ys - 63.5).pow(2) + (xs - 63.5).pow(2)).sqrt()
    labels = torch.zeros(size, size, dtype=torch.long)
    labels[radius < 48.0] = 1
    labels[radius < 32.0] = 2
    labels[radius < 16.0] = 3
    image = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)[labels]
    image = image + 0.01 * index
    # Distinct support counts/IDs ensure the bulk API does not accidentally
    # assume one shared unit or one partition.
    block = torch.arange(size) // 16
    by, bx = torch.meshgrid(block, block, indexing="ij")
    select = ((by + 2 * bx + index) % (5 + index)) == 0
    if index == 2:
        select[:8] = False
    fit = ~select
    fit_image = torch.where(fit, image, torch.zeros_like(image)).unsqueeze(0)
    fitting = FittingView(
        image=fit_image,
        support=fit,
        context=fit_image.repeat(3, 1, 1),
        study_id=f"large-study-{index}",
        unit_id=f"large-unit-{index}",
        partition_id=f"large-partition-{index}",
    )
    scoring = ScoringView(
        image=torch.where(select, image, torch.zeros_like(image)).unsqueeze(0),
        support=select,
        study_id=f"large-study-{index}",
        unit_id=f"large-unit-{index}",
        role="select",
        partition_id=f"large-partition-{index}",
    )
    return fitting, scoring, labels


def _assert_fit_equivalent(reference, candidate) -> None:
    assert candidate.hypothesis.candidate_id == reference.hypothesis.candidate_id
    assert candidate.hypothesis.source == reference.hypothesis.source
    assert torch.equal(candidate.hypothesis.labels, reference.hypothesis.labels)
    assert torch.equal(candidate.hypothesis.probabilities, reference.hypothesis.probabilities)
    assert torch.equal(candidate.hypothesis.validity, reference.hypothesis.validity)
    assert candidate.hypothesis.metadata == reference.hypothesis.metadata
    assert candidate.parameters["fit_support_shape"] == reference.parameters["fit_support_shape"]
    assert candidate.parameters["fit_support_bits"] == reference.parameters["fit_support_bits"]
    assert candidate.parameters["region_counts"] == reference.parameters["region_counts"]
    assert candidate.parameters["fallback_regions"] == reference.parameters["fallback_regions"]
    assert candidate.parameters["fallback_region_count"] == reference.parameters["fallback_region_count"]
    for key in ("global_mean", "global_variance"):
        assert candidate.parameters[key] == pytest.approx(reference.parameters[key], rel=2e-14, abs=2e-14)
    for got, expected in zip(candidate.parameters["bias_coefficients"], reference.parameters["bias_coefficients"]):
        assert got == pytest.approx(expected, rel=2e-14, abs=2e-14)
    assert len(candidate.parameters["components"]) == len(reference.parameters["components"])
    for got, expected in zip(candidate.parameters["components"], reference.parameters["components"]):
        assert got["region"] == expected["region"]
        assert got["fallback"] == expected["fallback"]
        for key in ("mean", "variance", "log_weight"):
            assert got[key] == pytest.approx(expected[key], rel=2e-14, abs=2e-14)
    assert candidate.metadata["hypothesis_digest"] == reference.metadata["hypothesis_digest"]
    assert candidate.fit_score.count == reference.fit_score.count
    assert candidate.fit_score.nll_sum == pytest.approx(reference.fit_score.nll_sum, rel=2e-12, abs=2e-12)
    assert candidate.fit_score.normalized_nll == pytest.approx(
        reference.fit_score.normalized_nll, rel=2e-12, abs=2e-12
    )
    assert candidate.fit_score.complexity == reference.fit_score.complexity
    assert candidate.fit_score.prior == reference.fit_score.prior
    assert candidate.fit_score.total == pytest.approx(reference.fit_score.total, rel=2e-12, abs=2e-12)


def test_fit_many_multiple_units_and_128_grid_fields() -> None:
    cases = [_large_unit(index) for index in range(3)]
    fitting_views = [case[0] for case in cases]
    scoring_views = [case[1] for case in cases]
    candidates = [
        hypothesis_from_labels(case[2], f"large-{index}", "large")
        for index, case in enumerate(cases)
    ]
    model = ObservationModel()
    scalar_fits = [model.fit(candidate, view) for candidate, view in zip(candidates, fitting_views)]
    batch_fits = model.fit_many(candidates, fitting_views)
    assert [fit.unit_id for fit in batch_fits] == [view.unit_id for view in fitting_views]
    for reference, candidate in zip(scalar_fits, batch_fits):
        _assert_fit_equivalent(reference, candidate)

    scalar_scores = [model.score(fit, view) for fit, view in zip(scalar_fits, scoring_views)]
    batch_scores = model.score_many(batch_fits, scoring_views)
    for reference, candidate in zip(scalar_scores, batch_scores):
        assert candidate.nll_sum == pytest.approx(reference.nll_sum, rel=2e-12, abs=2e-12)
        assert candidate.normalized_nll == pytest.approx(
            reference.normalized_nll, rel=2e-12, abs=2e-12
        )
        assert candidate.complexity == reference.complexity
        assert candidate.prior == reference.prior
        assert candidate.total == pytest.approx(reference.total, rel=2e-12, abs=2e-12)

    untouched = copy.deepcopy(batch_fits[1].parameters)
    batch_fits[0].parameters["components"][0]["mean"] += 999.0
    assert batch_fits[1].parameters == untouched


def test_score_many_empty_batch_and_guard_failures() -> None:
    fitting, scoring = _views()
    model = ObservationModel()
    fits = model.fit_many(_bank()[:2], fitting)
    empty = ScoringView(
        image=torch.zeros_like(scoring.image),
        support=torch.zeros_like(scoring.support),
        study_id=STUDY,
        unit_id=UNIT,
        role="select",
        partition_id=PARTITION,
    )
    scores = model.score_many(fits, [empty, empty])
    assert len(scores) == 2
    assert all(
        (score.available is False)
        and score.count == 0
        and score.normalized_nll is None
        and score.total is None
        and score.reason == "no observed pixels in this scoring role for this unit"
        for score in scores
    )

    overlap = ScoringView(
        image=fitting.image,
        support=fitting.support,
        study_id=STUDY,
        unit_id=UNIT,
        role="select",
        partition_id=PARTITION,
    )
    with pytest.raises(ObservationContractError, match="overlap"):
        model.score_many(fits, [overlap, overlap])

    wrong_recipe = ObservationModel(background_modes=2)
    with pytest.raises(ObservationContractError, match="recipe mismatch"):
        wrong_recipe.score_many(fits, [scoring, scoring])

    fits[0].hypothesis.labels.zero_()
    with pytest.raises(ObservationContractError, match="mutated after fitting"):
        model.score_many(fits, [scoring, scoring])


@pytest.mark.parametrize("background_modes", [1, 2])
@pytest.mark.parametrize("distribution", ["gaussian", "student_t"])
def test_fit_many_and_score_many_match_reference_and_selection(
    background_modes: int, distribution: str
) -> None:
    fitting, scoring = _views()
    model = ObservationModel(background_modes=background_modes, distribution=distribution)
    bank = _bank()

    scalar_fits = [model.fit(candidate, fitting) for candidate in bank]
    batch_fits = model.fit_many(bank, [fitting] * len(bank))
    assert [fit.hypothesis.candidate_id for fit in batch_fits] == [
        fit.hypothesis.candidate_id for fit in scalar_fits
    ]
    assert len(batch_fits) == len(bank)
    for scalar, batched in zip(scalar_fits, batch_fits):
        assert batched.fitting_steps == scalar.fitting_steps == 5
        assert batched.capacity == scalar.capacity == model.capacity
        _assert_fit_equivalent(scalar, batched)

    scalar_scores = [model.score(fit, scoring) for fit in scalar_fits]
    batch_scores = model.score_many(batch_fits, [scoring] * len(batch_fits))
    assert [score.count for score in batch_scores] == [score.count for score in scalar_scores]
    for scalar, batched in zip(scalar_scores, batch_scores):
        assert batched.nll_sum == pytest.approx(scalar.nll_sum, rel=2e-11, abs=2e-11)
        assert batched.normalized_nll == pytest.approx(scalar.normalized_nll, rel=2e-11, abs=2e-11)
        assert batched.total == pytest.approx(scalar.total, rel=2e-11, abs=2e-11)
    # Floating reordering must never change the discrete winner or strict edit
    # comparison.  This is the non-negotiable part of equivalence.
    scalar_order = sorted(range(len(scalar_scores)), key=lambda i: scalar_scores[i].total)
    batch_order = sorted(range(len(batch_scores)), key=lambda i: batch_scores[i].total)
    assert batch_order == scalar_order


def test_shared_view_convenience_and_candidate_independence() -> None:
    fitting, scoring = _views()
    model = ObservationModel()
    bank = _bank()[:3]
    batch = model.fit_many(bank, fitting)
    assert len(batch) == 3
    before_other = copy.deepcopy(batch[1].parameters)
    batch[0].parameters["components"][0]["mean"] += 123.0
    assert batch[1].parameters == before_other

    scores = model.score_many(batch, scoring)
    assert len(scores) == 3
    # A corresponding sequence across units is accepted, while a length
    # mismatch is rejected before any partial result is returned.
    fitting_other = FittingView(
        image=fitting.image,
        support=fitting.support,
        context=fitting.context,
        study_id=STUDY,
        unit_id="other_unit",
        partition_id=PARTITION,
    )
    with pytest.raises(ObservationContractError, match="unit mismatch"):
        model.score_many(batch, [scoring, scoring, ScoringView(
            image=scoring.image,
            support=scoring.support,
            study_id=STUDY,
            unit_id="other_unit",
            role="select",
            partition_id=PARTITION,
        )])
    with pytest.raises(ObservationContractError, match="length"):
        model.fit_many(bank, [fitting, fitting_other])

    original_group = model._fit_many_group
    model._fit_many_group = lambda *_args, **_kwargs: []  # type: ignore[method-assign]
    with pytest.raises(ObservationContractError, match="unexpected result count"):
        model.fit_many(bank, fitting)
    model._fit_many_group = original_group  # type: ignore[method-assign]


def test_prepare_score_exact_regions_and_snapshot_mutation_guard() -> None:
    fitting, scoring = _views()
    model = ObservationModel()
    fitted = model.fit(_bank()[0], fitting)
    prepared = model.prepare_score(fitted, scoring)
    assert isinstance(prepared, PreparedScore)
    scalar = model.score(fitted, scoring)
    assert prepared.available == scalar.available
    assert prepared.count == scalar.count
    assert prepared.nll_sum == pytest.approx(scalar.nll_sum, rel=2e-12, abs=2e-12)
    assert prepared.total == pytest.approx(scalar.total, rel=2e-12, abs=2e-12)

    regions = [scoring.support, _labels() == 0, _labels() == 1, _labels() == 2, _labels() == 3]
    for region in regions:
        regional = model.score_region(prepared, region)
        reference = model.score(fitted, _restricted(scoring, region))
        assert regional.available == reference.available
        assert regional.count == reference.count
        assert regional.nll_sum == pytest.approx(reference.nll_sum, rel=2e-12, abs=2e-12)
        if reference.normalized_nll is None:
            assert regional.normalized_nll is None
        else:
            assert regional.normalized_nll == pytest.approx(
                reference.normalized_nll, rel=2e-12, abs=2e-12
            )
        assert regional.complexity == pytest.approx(reference.complexity, rel=0.0, abs=0.0)
        assert regional.prior == pytest.approx(reference.prior, rel=0.0, abs=0.0)
        assert regional.reason == reference.reason
        if reference.total is None:
            assert regional.total is None
        else:
            assert regional.total == pytest.approx(reference.total, rel=2e-12, abs=2e-12)

    arbitrary = torch.rand(
        scoring.support.shape, generator=torch.Generator().manual_seed(19)
    ) > 0.63
    arbitrary_score = model.score_region(prepared, arbitrary)
    arbitrary_reference = model.score(fitted, _restricted(scoring, arbitrary))
    assert arbitrary_score.count == arbitrary_reference.count
    assert arbitrary_score.nll_sum == pytest.approx(
        arbitrary_reference.nll_sum, rel=2e-12, abs=2e-12
    )

    # Original fitted/view mutations cannot affect the detached prepared copy.
    baseline = model.score_region(prepared, scoring.support)
    fitted.hypothesis.labels.zero_()
    scoring.image.add_(17.0)
    assert model.score_region(prepared, scoring.support).total == baseline.total
    # Prepared reductions retain the fit-time recipe even if the live model is
    # changed after preparation; the public scalar path independently rejects
    # that changed recipe.
    model.beta = 9.0
    assert model.score_region(prepared, scoring.support).total == baseline.total
    prepared._nll_per_pixel[0, 0] += 1.0
    with pytest.raises(ObservationContractError, match="prepared score"):
        model.score_region(prepared, scoring.support)


def test_prepare_empty_support_and_tiny_degenerate_shapes() -> None:
    fitting, scoring = _views()
    empty = ScoringView(
        image=torch.zeros_like(scoring.image),
        support=torch.zeros_like(scoring.support),
        study_id=STUDY,
        unit_id=UNIT,
        role="select",
        partition_id=PARTITION,
    )
    model = ObservationModel()
    fitted = model.fit(_bank()[0], fitting)
    scalar = model.score(fitted, empty)
    prepared = model.prepare_score(fitted, empty)
    region = model.score_region(prepared, torch.ones_like(empty.support))
    for score in (scalar, prepared, region):
        assert score.available is False
        assert score.count == 0
        assert score.normalized_nll is None and score.total is None
        assert score.reason == "no observed pixels in this scoring role for this unit"

    # One pixel exercises zero neighbour pairs and all fallback components.
    one = torch.tensor([[0.25]], dtype=torch.float32)
    one_fit = FittingView(
        image=one.unsqueeze(0),
        support=torch.ones(1, 1, dtype=torch.bool),
        context=one.unsqueeze(0).repeat(3, 1, 1),
        study_id="tiny",
        unit_id="tiny-0",
        partition_id="tiny-partition",
    )
    one_score = ScoringView(
        image=torch.zeros(1, 1, 1),
        support=torch.zeros(1, 1, dtype=torch.bool),
        study_id="tiny",
        unit_id="tiny-0",
        role="select",
        partition_id="tiny-partition",
    )
    tiny_hypothesis = hypothesis_from_labels(torch.zeros(1, 1, dtype=torch.long), "tiny", "tiny")
    tiny = ObservationModel(background_modes=2)
    tiny_fit = tiny.fit_many([tiny_hypothesis], one_fit)[0]
    assert tiny_fit.parameters["fallback_regions"] == [0, 1, 2, 3]
    assert len(tiny_fit.parameters["components"]) == tiny.capacity == 5
    assert tiny.score_many([tiny_fit], one_score)[0].available is False


def test_fit_only_firewall_and_foreign_prepared_provenance() -> None:
    fitting, scoring = _views()
    model = ObservationModel()
    bank = _bank()[:2]
    fits_a = model.fit_many(bank, fitting)
    changed_scoring = ScoringView(
        image=scoring.image * -5.0,
        support=scoring.support,
        study_id=STUDY,
        unit_id=UNIT,
        role="select",
        partition_id=PARTITION,
    )
    fits_b = model.fit_many(bank, fitting)
    assert [fit.parameters for fit in fits_a] == [fit.parameters for fit in fits_b]
    assert [score.total for score in model.score_many(fits_a, scoring)] != [
        score.total for score in model.score_many(fits_a, changed_scoring)
    ]

    prepared = model.prepare_score(fits_a[0], scoring)
    foreign_support = ScoringView(
        image=scoring.image,
        support=scoring.support,
        study_id=STUDY,
        unit_id="foreign",
        role="select",
        partition_id=PARTITION,
    )
    with pytest.raises(ObservationContractError, match="unit mismatch"):
        model.prepare_score(fits_a[0], foreign_support)
    foreign_partition = ScoringView(
        image=scoring.image,
        support=scoring.support,
        study_id=STUDY,
        unit_id=UNIT,
        role="select",
        partition_id="foreign-partition",
    )
    with pytest.raises(ObservationContractError, match="partition mismatch"):
        model.prepare_score(fits_a[0], foreign_partition)
    assert model.score_region(prepared, scoring.support).available
