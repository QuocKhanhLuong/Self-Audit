"""Execution-device routing checks for the mask-free observation model.

These fixtures are CPU synthetic software checks.  The CUDA test is skipped
when this checkout has no CUDA runtime; a skipped test is not GPU evidence.
"""

from __future__ import annotations

import math

import pytest
import torch

from self_audit_maskfree.contracts import (
    FittingView,
    ScoringView,
    hypothesis_from_labels,
)
from self_audit_maskfree.observation import ObservationContractError, ObservationModel

SIZE = 32


def _case(
    *, phase: int = 0, intensity_scale: float = 1.0, intensity_offset: float = 0.0
) -> tuple[FittingView, ScoringView, torch.Tensor]:
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, dtype=torch.float32),
        torch.arange(SIZE, dtype=torch.float32),
        indexing="ij",
    )
    radius = ((ys - 15.5).pow(2) + (xs - 15.5).pow(2)).sqrt()
    labels = torch.zeros(SIZE, SIZE, dtype=torch.long)
    labels[radius < 12.0] = 1
    labels[radius < 8.0] = 2
    labels[radius < 4.0] = 3
    image = torch.tensor([-1.0, 0.0, 1.0, 2.0])[labels]
    image = intensity_offset + intensity_scale * (
        image + 0.05 * torch.sin(xs / 3.0) + 0.03 * torch.cos(ys / 5.0)
    )
    block = torch.arange(SIZE) // 8
    by, bx = torch.meshgrid(block, block, indexing="ij")
    scoring_support = ((by + bx + phase) % 4) == 0
    fitting_support = ~scoring_support
    fitting = FittingView(
        image=torch.where(fitting_support, image, torch.zeros_like(image)).unsqueeze(0),
        support=fitting_support,
        context=torch.where(fitting_support, image, torch.zeros_like(image))
        .unsqueeze(0)
        .repeat(3, 1, 1),
        study_id="device-study",
        unit_id="device-unit",
        partition_id="device-partition",
    )
    scoring = ScoringView(
        image=torch.where(scoring_support, image, torch.zeros_like(image)).unsqueeze(0),
        support=scoring_support,
        study_id="device-study",
        unit_id="device-unit",
        role="select",
        partition_id="device-partition",
    )
    return fitting, scoring, labels


def _bank(labels: torch.Tensor) -> list:
    return [
        hypothesis_from_labels(labels, "truth", "device-test"),
        hypothesis_from_labels(torch.zeros_like(labels), "all-bg", "device-test"),
        hypothesis_from_labels(labels.roll(3, dims=1), "shifted", "device-test"),
        hypothesis_from_labels((labels + 1) % 4, "permuted", "device-test"),
    ]


def test_execution_device_validation_and_default_reference() -> None:
    assert ObservationModel().execution_device is None
    assert ObservationModel(execution_device="cpu").execution_device == torch.device(
        "cpu"
    )
    assert ObservationModel(
        execution_device=torch.device("cpu")
    ).execution_device == torch.device("cpu")
    with pytest.raises(ObservationContractError, match="execution_device"):
        ObservationModel(execution_device=object())  # type: ignore[arg-type]
    with pytest.raises(ObservationContractError, match="valid torch device"):
        ObservationModel(execution_device="not-a-device")
    for unsupported in ("mps", "meta"):
        with pytest.raises(ObservationContractError, match="only CPU and CUDA"):
            ObservationModel(execution_device=unsupported)
    if not torch.cuda.is_available():
        with pytest.raises(ObservationContractError, match="CUDA is unavailable"):
            ObservationModel(execution_device="cuda")


@pytest.mark.parametrize("background_modes", [1, 2])
@pytest.mark.parametrize("distribution", ["gaussian", "student_t"])
def test_explicit_cpu_matches_legacy_discrete_budget(
    background_modes: int, distribution: str
) -> None:
    fitting, scoring, labels = _case()
    bank = _bank(labels)
    legacy = ObservationModel(
        background_modes=background_modes, distribution=distribution
    )
    device_model = ObservationModel(
        background_modes=background_modes,
        distribution=distribution,
        execution_device="cpu",
    )
    legacy_fits = legacy.fit_many(bank, fitting)
    device_fits = device_model.fit_many(bank, fitting)
    assert [fit.fitting_steps for fit in device_fits] == [5] * len(bank)
    assert [fit.capacity for fit in device_fits] == [legacy.capacity] * len(bank)
    # Explicit CPU is intentionally the exact legacy arithmetic path, not the
    # reordered tensorized CUDA emulation.
    assert [fit.parameters for fit in device_fits] == [
        fit.parameters for fit in legacy_fits
    ]
    assert [fit.fit_score for fit in device_fits] == [
        fit.fit_score for fit in legacy_fits
    ]
    assert [fit.parameters["fallback_regions"] for fit in device_fits] == [
        fit.parameters["fallback_regions"] for fit in legacy_fits
    ]
    legacy_scores = legacy.score_many(legacy_fits, scoring)
    device_scores = device_model.score_many(device_fits, scoring)
    assert sorted(
        range(len(legacy_scores)), key=lambda i: legacy_scores[i].total
    ) == sorted(range(len(device_scores)), key=lambda i: device_scores[i].total)
    for reference, candidate in zip(legacy_scores, device_scores):
        assert candidate.count == reference.count
        assert candidate.nll_sum == pytest.approx(
            reference.nll_sum, rel=2e-10, abs=2e-10
        )
        assert candidate.total == pytest.approx(reference.total, rel=2e-10, abs=2e-10)
    # Explicit-device fits are portable snapshots, not CUDA-owned hypotheses.
    for fitted in device_fits:
        assert fitted.hypothesis.labels.device.type == "cpu"
        assert fitted.hypothesis.probabilities.device.type == "cpu"
        assert fitted.hypothesis.validity.device.type == "cpu"
        assert all(
            math.isfinite(float(component[key]))
            for component in fitted.parameters["components"]
            for key in ("mean", "variance", "log_weight")
        )


def test_explicit_cpu_prepared_snapshot_is_device_local_and_frozen() -> None:
    fitting, scoring, labels = _case()
    model = ObservationModel(execution_device="cpu")
    fitted = model.fit(hypothesis_from_labels(labels, "truth", "device-test"), fitting)
    prepared = model.prepare_score(fitted, scoring)
    assert prepared.nll_per_pixel.device.type == "cpu"
    assert prepared.support.device.type == "cpu"
    full = model.score(fitted, scoring)
    assert prepared.total == pytest.approx(full.total, rel=2e-12, abs=2e-12)
    assert model.score_region(prepared, scoring.support).total == pytest.approx(
        full.total, rel=2e-12, abs=2e-12
    )
    with pytest.raises(ObservationContractError, match="prepared score device"):
        model.score_region(prepared, scoring.support.to(device="meta"))


def test_tensorized_helper_cpu_emulation_preserves_empty_background_mode_equations() -> (
    None
):
    size = 16
    image = torch.full((size, size), 3.0, dtype=torch.float32)
    support = torch.ones(size, size, dtype=torch.bool)
    view = FittingView(
        image=image.unsqueeze(0),
        support=support,
        context=image.unsqueeze(0).repeat(3, 1, 1),
        study_id="constant-study",
        unit_id="constant-unit",
        partition_id="constant-partition",
    )
    labels = torch.zeros(size, size, dtype=torch.long)
    candidate = hypothesis_from_labels(labels, "constant-bg", "device-test")
    reference = ObservationModel(background_modes=2).fit(candidate, view)
    tensorized_model = ObservationModel(background_modes=2, execution_device="cpu")
    tensorized = tensorized_model._fit_many_group_on_device(
        [candidate], [view], device=torch.device("cpu")
    )[0]
    assert tensorized.parameters["components"][1]["fallback"] is True
    assert tensorized.parameters["components"][1]["log_weight"] == pytest.approx(
        -math.log(size * size), rel=0.0, abs=1e-14
    )
    assert tensorized.parameters["bias_coefficients"] == pytest.approx(
        reference.parameters["bias_coefficients"], rel=1e-13, abs=1e-13
    )
    assert tensorized.fit_score.normalized_nll == pytest.approx(
        reference.fit_score.normalized_nll, rel=1e-13, abs=1e-13
    )


@pytest.mark.parametrize("background_modes", [1, 2])
@pytest.mark.parametrize("distribution", ["gaussian", "student_t"])
def test_tensorized_helper_cpu_emulation_covers_distinct_units_and_rare_regions(
    background_modes: int, distribution: str
) -> None:
    fitting_a, scoring_a, labels_a = _case()
    fitting_b, scoring_b, labels_b = _case(
        phase=1, intensity_scale=1.0e-3, intensity_offset=1.0e6
    )
    candidates = [_bank(labels_a)[0], _bank(labels_b)[1]]
    views = [fitting_a, fitting_b]
    reference_model = ObservationModel(
        background_modes=background_modes, distribution=distribution
    )
    tensorized_model = ObservationModel(
        background_modes=background_modes,
        distribution=distribution,
        execution_device="cpu",
    )
    references = [
        reference_model.fit(candidate, view)
        for candidate, view in zip(candidates, views)
    ]
    tensorized = tensorized_model._fit_many_group_on_device(
        candidates, views, device=torch.device("cpu")
    )

    def assert_nested(got, expected) -> None:  # type: ignore[no-untyped-def]
        if isinstance(expected, dict):
            assert isinstance(got, dict)
            assert got.keys() == expected.keys()
            for key in expected:
                assert_nested(got[key], expected[key])
        elif isinstance(expected, list):
            assert isinstance(got, list)
            assert len(got) == len(expected)
            for got_item, expected_item in zip(got, expected):
                assert_nested(got_item, expected_item)
        elif isinstance(expected, (bool, int)):
            assert got == expected
        elif isinstance(expected, float):
            # CPU tensorized reductions may reorder float64 additions by a few
            # ulps.  The 1e6-offset low-variance fixture reaches 3.73e-9 in
            # component means from FP64 reduction/subtraction ordering and
            # cancellation near 1e6; both paths receive identical float32 input.
            assert got == pytest.approx(expected, rel=1e-12, abs=1e-8)
        else:
            assert got == expected

    for got, expected in zip(tensorized, references):
        assert_nested(got.parameters, expected.parameters)
        assert got.fit_score.count == expected.fit_score.count
        assert got.fit_score.nll_sum == pytest.approx(
            expected.fit_score.nll_sum, rel=1e-12, abs=1e-8
        )
        assert got.fit_score.normalized_nll == pytest.approx(
            expected.fit_score.normalized_nll, rel=1e-12, abs=1e-8
        )
        assert (
            got.parameters["fallback_regions"]
            == expected.parameters["fallback_regions"]
        )
    # Scores are evaluated through the public CPU reference path, so compare
    # the tensorized fit snapshots under both distinct scoring supports.
    got_scores = tensorized_model.score_many(tensorized, [scoring_a, scoring_b])
    expected_scores = [
        reference_model.score(fitted, view)
        for fitted, view in zip(references, [scoring_a, scoring_b])
    ]
    for got, expected in zip(got_scores, expected_scores):
        assert got.count == expected.count
        assert got.available == expected.available
        assert got.reason == expected.reason
        assert got.nll_sum == pytest.approx(expected.nll_sum, rel=1e-12, abs=1e-8)


def test_explicit_cpu_rejects_non_cpu_capability_view() -> None:
    size = 2
    image = torch.empty(1, size, size, device="meta")
    support = torch.empty(size, size, dtype=torch.bool, device="meta")
    context = torch.empty(3, size, size, device="meta")
    view = FittingView(image, support, context, "meta-study", "meta-unit")
    candidate = hypothesis_from_labels(
        torch.zeros(size, size, dtype=torch.long), "meta", "device"
    )
    with pytest.raises(ObservationContractError, match="requires CPU capability"):
        ObservationModel(execution_device="cpu").fit(candidate, view)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable; GPU evidence NOT RUN"
)
def test_cuda_residency_equivalence_and_portable_parameters() -> None:
    fitting, scoring, labels = _case()
    bank = _bank(labels)
    cpu_model = ObservationModel(
        background_modes=2, distribution="student_t", execution_device="cpu"
    )
    cuda_model = ObservationModel(
        background_modes=2, distribution="student_t", execution_device="cuda"
    )
    cpu_fits = cpu_model.fit_many(bank, fitting)
    cuda_fits = cuda_model.fit_many(bank, fitting)
    cpu_scores = cpu_model.score_many(cpu_fits, scoring)
    cuda_scores = cuda_model.score_many(cuda_fits, scoring)
    assert [fit.fitting_steps for fit in cuda_fits] == [5] * len(bank)
    assert [fit.parameters["fallback_regions"] for fit in cuda_fits] == [
        fit.parameters["fallback_regions"] for fit in cpu_fits
    ]
    for fitted in cuda_fits:
        assert fitted.hypothesis.labels.device.type == "cpu"
        assert all(
            isinstance(value, (int, float, str, bool))
            for value in fitted.parameters["bias_coefficients"]
        )
    for reference, candidate in zip(cpu_scores, cuda_scores):
        assert candidate.nll_sum == pytest.approx(reference.nll_sum, rel=2e-8, abs=2e-8)
        assert candidate.total == pytest.approx(reference.total, rel=2e-8, abs=2e-8)
    prepared = cuda_model.prepare_score(cuda_fits[0], scoring)
    assert prepared.nll_per_pixel.device.type == "cpu"
    assert prepared.support.device.type == "cpu"
    regional = cuda_model.score_region(prepared, scoring.support)
    assert regional.nll_sum == pytest.approx(cuda_scores[0].nll_sum, rel=2e-8, abs=2e-8)
