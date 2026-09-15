"""Exact-equivalence checks for the maskfree150 connected-components fast path.

The reference below deliberately spells out the frozen synchronous recurrence
instead of calling any production helper.  These tests are CPU software checks:
they establish bitwise label/flag parity and input/device contracts, not a
clinical or GPU performance claim.
"""
from __future__ import annotations

from collections import deque
from itertools import product

import pytest
import torch

from self_audit_maskfree.ontology import OntologyError, connected_components


def _reference_connected_components(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, bool, int]:
    """Independent frozen recurrence with an iteration count for boundary tests."""
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise AssertionError("reference expects bool [H,W]")
    height, width = mask.shape
    if not bool(mask.any()):
        return torch.zeros_like(mask, dtype=torch.long), True, 0

    ids = torch.arange(
        1, height * width + 1, dtype=torch.float64, device=mask.device
    ).reshape(height, width)
    ids = torch.where(mask, ids, torch.zeros_like(ids))
    budget = 2 * (height + width)
    for iteration in range(1, budget + 1):
        # Spell out the zero-padded self-plus-four-neighbour maximum
        # independently of the production ``_neighbor_max`` helper.  Updates are
        # synchronous because every value is read from ``ids`` and written to
        # ``propagated``.
        propagated = ids.clone()
        for row, column in product(range(height), range(width)):
            if not bool(mask[row, column]):
                continue
            neighbours: list[torch.Tensor] = [ids[row, column]]
            if row > 0:
                neighbours.append(ids[row - 1, column])
            if row + 1 < height:
                neighbours.append(ids[row + 1, column])
            if column > 0:
                neighbours.append(ids[row, column - 1])
            if column + 1 < width:
                neighbours.append(ids[row, column + 1])
            if neighbours:
                propagated[row, column] = torch.stack(neighbours).max()
        if bool(torch.equal(propagated, ids)):
            return ids.long(), True, iteration
        ids = propagated
    return ids.long(), False, budget


def _assert_matches_reference(mask: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Check labels, convergence, dtype/device, and non-mutation together."""
    original = mask.clone()
    expected, expected_converged, _ = _reference_connected_components(mask)
    actual, actual_converged = connected_components(mask)
    assert torch.equal(actual, expected)
    assert actual_converged is expected_converged
    assert actual.dtype == torch.int64
    assert actual.device == mask.device
    assert torch.equal(mask, original), "connected_components mutated its input"
    return actual, actual_converged


def _max_id_eccentricity(mask: torch.Tensor) -> int:
    """Independent graph distance from a component's maximum raster-ID pixel."""
    height, width = mask.shape
    ids = torch.arange(1, height * width + 1, dtype=torch.long).reshape(height, width)
    masked_ids = torch.where(
        mask.reshape(-1), ids.reshape(-1), torch.zeros_like(ids).reshape(-1)
    )
    source_flat = masked_ids.argmax()
    source = (int(source_flat) // width, int(source_flat) % width)
    distances = [[-1] * width for _ in range(height)]
    distances[source[0]][source[1]] = 0
    queue: deque[tuple[int, int]] = deque([source])
    while queue:
        row, column = queue.popleft()
        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if not (0 <= next_row < height and 0 <= next_column < width):
                continue
            if not bool(mask[next_row, next_column]) or distances[next_row][next_column] >= 0:
                continue
            distances[next_row][next_column] = distances[row][column] + 1
            queue.append((next_row, next_column))
    return max(distance for row in distances for distance in row)


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (2, 3), (3, 3)])
def test_exhaustive_small_masks_match_frozen_reference(shape: tuple[int, int]) -> None:
    height, width = shape
    for bits in range(1 << (height * width)):
        values = [(bits >> index) & 1 for index in range(height * width)]
        mask = torch.tensor(values, dtype=torch.bool).reshape(height, width)
        _assert_matches_reference(mask)


@pytest.mark.parametrize("shape", [(3, 4), (5, 7), (8, 9)])
def test_seeded_random_masks_match_frozen_reference(shape: tuple[int, int]) -> None:
    generator = torch.Generator().manual_seed(150_447)
    for probability in (0.1, 0.5, 0.9):
        for _ in range(8):
            mask = torch.rand(shape, generator=generator) < probability
            _assert_matches_reference(mask)


def test_seeded_128_masks_dense_sparse_and_noncontiguous_match_reference() -> None:
    """Exercise the CPU shortcut on representative 128-pixel mask layouts."""
    generator = torch.Generator().manual_seed(128_150)
    sparse = torch.rand((1, 128), generator=generator) < 0.05
    dense = torch.rand((1, 128), generator=generator) < 0.95
    base = torch.rand((1, 256), generator=generator) < 0.5
    noncontiguous = base[:, ::2]
    assert not noncontiguous.is_contiguous()

    for mask in (sparse, dense, noncontiguous):
        _assert_matches_reference(mask)


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros((4, 5), dtype=torch.bool),
        torch.ones((4, 5), dtype=torch.bool),
        torch.tensor(
            [
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
            ],
            dtype=torch.bool,
        ),
        torch.tensor([[True, False, True, True, False, True]], dtype=torch.bool),
        torch.tensor([[True], [False], [True], [True], [False], [True]], dtype=torch.bool),
    ],
)
def test_edge_shapes_and_masks_match_frozen_reference(mask: torch.Tensor) -> None:
    _assert_matches_reference(mask)


def test_noncontiguous_mask_matches_reference_and_is_not_repacked_in_place() -> None:
    base = torch.zeros((7, 12), dtype=torch.bool)
    base[1:6, 1::2] = torch.tensor(
        [
            [True, False, True, True, False, True],
            [False, True, True, False, True, False],
            [True, True, False, True, False, True],
            [False, True, False, True, True, False],
            [True, False, True, False, True, True],
        ],
        dtype=torch.bool,
    )
    mask = base[:, ::2]
    assert not mask.is_contiguous()
    before = base.clone()
    _assert_matches_reference(mask)
    assert torch.equal(base, before)


def test_converged_components_keep_the_maximum_raster_id_per_component() -> None:
    mask = torch.tensor(
        [
            [True, True, False],
            [False, True, False],
            [False, False, True],
        ],
        dtype=torch.bool,
    )
    labels, converged = _assert_matches_reference(mask)
    assert converged is True
    assert labels.tolist() == [[5, 5, 0], [0, 5, 0], [0, 0, 9]]


def _serpentine_mask(height: int = 17, width: int = 17) -> torch.Tensor:
    """One-cell-wide path whose graph diameter exceeds the finite budget."""
    mask = torch.zeros((height, width), dtype=torch.bool)
    for row in range(1, height, 2):
        mask[row, :] = True
        if row + 1 < height:
            connector_column = width - 1 if ((row - 1) // 2) % 2 == 0 else 0
            mask[row + 1, connector_column] = True
    return mask


def test_long_serpentine_exercises_finite_budget_nonconvergence() -> None:
    mask = _serpentine_mask()
    expected, expected_converged, iterations = _reference_connected_components(mask)
    assert expected_converged is False
    assert iterations == 2 * sum(mask.shape)
    actual, converged = connected_components(mask)
    assert converged is False
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("width", "expected_eccentricity", "expected_converged"),
    [
        # The max-ID source eccentricity is cap-1: the stability check at the
        # cap sees the already-complete state and must return converged=True.
        (10, 31, True),
        # Eccentricity cap and cap+1 both exhaust the cap before the next
        # stability check; both must retain the exact partial labels and False.
        (11, 34, False),
        (12, 37, False),
    ],
)
def test_serpentine_convergence_cap_boundaries(
    width: int, expected_eccentricity: int, expected_converged: bool
) -> None:
    mask = _serpentine_mask(6, width)
    expected, reference_converged, iterations = _reference_connected_components(mask)
    budget = 2 * sum(mask.shape)
    assert _max_id_eccentricity(mask) == expected_eccentricity
    assert iterations == budget
    assert reference_converged is expected_converged

    actual, converged = connected_components(mask)
    assert torch.equal(actual, expected)
    assert converged is expected_converged


def test_disconnected_components_with_different_eccentricity_keep_each_max_id() -> None:
    # A length-five horizontal component (eccentricity four) and a singleton
    # (eccentricity zero) force independent component maxima/seeds.
    mask = torch.zeros((6, 10), dtype=torch.bool)
    mask[0, :5] = True
    mask[5, 9] = True
    labels, converged = _assert_matches_reference(mask)

    assert converged is True
    assert labels[0, :5].tolist() == [5, 5, 5, 5, 5]
    assert labels[5, 9].item() == 60
    assert int((labels > 0).sum()) == 6


def test_convergence_boundary_returns_stable_prior_state() -> None:
    # A singleton is stable on the first update; this specifically exercises the
    # pre-assignment convergence check while preserving its raster ID.
    mask = torch.tensor([[True]], dtype=torch.bool)
    expected, expected_converged, iterations = _reference_connected_components(mask)
    actual, converged = connected_components(mask)
    assert expected_converged is True
    assert iterations == 1
    assert converged is True
    assert torch.equal(actual, expected)
    assert actual.item() == 1


def test_input_validation_remains_unchanged() -> None:
    with pytest.raises(OntologyError, match=r"bool \[H,W\]"):
        connected_components(torch.zeros((2, 2), dtype=torch.float32))
    with pytest.raises(OntologyError, match=r"bool \[H,W\]"):
        connected_components(torch.zeros((2, 2, 1), dtype=torch.bool))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fallback_matches_reference_with_deterministic_algorithms() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        cpu_mask = _serpentine_mask(9, 9)
        expected, expected_converged, _ = _reference_connected_components(cpu_mask)
        actual, actual_converged = connected_components(cpu_mask.cuda())
        assert torch.equal(actual.cpu(), expected)
        assert actual_converged is expected_converged
        assert actual.device.type == "cuda"
    finally:
        torch.use_deterministic_algorithms(previous)
