"""Unit tests for bounded CPU candidate execution and optional spawn multiprocessing (W3).

Covers:
1. Canonical serial execution (workers=0) against single-unit generate_bank.
2. Persistent spawn pool (workers=2 and workers=4, threads=1) exact numerical equality.
3. Original batch ordering preservation.
4. Clean shutdown, lifecycle, and context manager operations.
5. Strict validation of worker counts, devices, thread bounds, and inputs.
6. Error propagation when workers encounter invalid inputs.
7. Explicit per-unit seeds and determinism.
"""
from __future__ import annotations

import pytest
import torch
import os
from concurrent.futures.process import BrokenProcessPool

from scripts.check_maskfree_audit_device import build_synthetic_units
from self_audit_maskfree.candidate_execution import (
    CandidateExecutionError,
    CandidatePoolExecutor,
    generate_candidate_banks,
)
from self_audit_maskfree.contracts import FittingView
from self_audit_maskfree.hypotheses import BANK_SIZE, generate_bank


@pytest.fixture(scope="module")
def synthetic_batch() -> tuple[list[FittingView], list[torch.Tensor], list[int]]:
    """Fixture producing a deterministic batch of 8 units at 224x224."""
    count = 8
    size = 224
    units = build_synthetic_units(size=size, seed=42, count=count)
    fitting_views = [u.fitting for u in units]
    # 16-channel producer features (only MAX_FEATURE_CHANNELS=8 consumed)
    generator = torch.Generator().manual_seed(1234)
    features = [torch.randn(16, size, size, generator=generator) for _ in range(count)]
    seeds = [1000 + i for i in range(count)]
    return fitting_views, features, seeds


def test_serial_execution_matches_generate_bank(
    synthetic_batch: tuple[list[FittingView], list[torch.Tensor], list[int]]
) -> None:
    """Canonical serial execution must match sequential generate_bank exactly."""
    fitting_views, features, seeds = synthetic_batch
    count = len(fitting_views)

    expected = [
        generate_bank(fitting_views[i], features=features[i], seed=seeds[i])
        for i in range(count)
    ]

    actual = generate_candidate_banks(
        fitting_views, features=features, seeds=seeds, workers=0
    )

    assert len(actual) == count
    for i in range(count):
        assert len(actual[i]) == BANK_SIZE
        for c in range(BANK_SIZE):
            assert torch.equal(actual[i][c].labels, expected[i][c].labels)
            assert actual[i][c].candidate_id == expected[i][c].candidate_id
            assert actual[i][c].metadata["bank_id"] == expected[i][c].metadata["bank_id"]
            assert actual[i][c].metadata["candidate_content_hash"] == expected[i][c].metadata["candidate_content_hash"]


@pytest.mark.parametrize("worker_count", [2, 4])
def test_spawn_pool_exact_numerical_equality(
    synthetic_batch: tuple[list[FittingView], list[torch.Tensor], list[int]],
    worker_count: int,
) -> None:
    """Multiprocessing spawn pool must produce exact byte-for-byte and hash equality."""
    fitting_views, features, seeds = synthetic_batch
    count = len(fitting_views)

    # Reference serial output
    reference = generate_candidate_banks(
        fitting_views, features=features, seeds=seeds, workers=0
    )

    # Persistent pool execution
    with CandidatePoolExecutor(workers=worker_count, worker_threads=1) as executor:
        actual = executor.generate_banks(fitting_views, features=features, seeds=seeds)

    assert len(actual) == count
    for i in range(count):
        assert len(actual[i]) == BANK_SIZE
        for c in range(BANK_SIZE):
            ref_c = reference[i][c]
            act_c = actual[i][c]
            assert act_c.candidate_id == ref_c.candidate_id
            assert act_c.source == ref_c.source
            assert torch.equal(act_c.labels, ref_c.labels)
            assert torch.equal(act_c.probabilities, ref_c.probabilities)
            assert torch.equal(act_c.validity, ref_c.validity)
            assert act_c.metadata["bank_id"] == ref_c.metadata["bank_id"]
            assert act_c.metadata["partition_hash"] == ref_c.metadata["partition_hash"]
            assert act_c.metadata["fit_input_hash"] == ref_c.metadata["fit_input_hash"]
            assert act_c.metadata["feature_hash"] == ref_c.metadata["feature_hash"]
            assert act_c.metadata["candidate_content_hash"] == ref_c.metadata["candidate_content_hash"]
            assert act_c.metadata["bank_index"] == ref_c.metadata["bank_index"]
            assert act_c.metadata["generation"]["source"] == ref_c.metadata["generation"]["source"]
            assert act_c.metadata["generation"]["seed"] == ref_c.metadata["generation"]["seed"]
            assert act_c.metadata["generation"]["hamming_vs_incumbent"] == ref_c.metadata["generation"]["hamming_vs_incumbent"]


def test_batch_without_features_and_default_seeds(
    synthetic_batch: tuple[list[FittingView], list[torch.Tensor], list[int]]
) -> None:
    """Execution with features=None and default seeds works in both serial and pool modes."""
    fitting_views, _, _ = synthetic_batch
    sub_views = fitting_views[:3]

    serial_banks = generate_candidate_banks(sub_views, features=None, seeds=None, workers=0)
    pool_banks = generate_candidate_banks(sub_views, features=None, seeds=None, workers=2, worker_threads=1)

    assert len(serial_banks) == len(sub_views)
    assert len(pool_banks) == len(sub_views)
    for i in range(len(sub_views)):
        for c in range(BANK_SIZE):
            assert torch.equal(serial_banks[i][c].labels, pool_banks[i][c].labels)
            assert serial_banks[i][c].metadata["bank_id"] == pool_banks[i][c].metadata["bank_id"]


def test_empty_batch_handling() -> None:
    """Empty batch inputs must return an empty list without error."""
    assert generate_candidate_banks([], workers=0) == []
    assert generate_candidate_banks([], workers=2) == []
    with CandidatePoolExecutor(workers=2) as executor:
        assert executor.generate_banks([]) == []


def test_executor_lifecycle_and_reuse(
    synthetic_batch: tuple[list[FittingView], list[torch.Tensor], list[int]]
) -> None:
    """Persistent pool can be reused across multiple calls and shuts down cleanly."""
    fitting_views, features, seeds = synthetic_batch
    sub_views = fitting_views[:2]
    sub_features = features[:2]
    sub_seeds = seeds[:2]

    executor = CandidatePoolExecutor(workers=2, worker_threads=1)
    try:
        # Call 1
        res1 = executor.generate_banks(sub_views, features=sub_features, seeds=sub_seeds)
        assert len(res1) == 2
        # Call 2 (reuses existing spawned workers)
        res2 = executor.generate_banks(sub_views, features=sub_features, seeds=sub_seeds)
        assert len(res2) == 2
        for i in range(2):
            for c in range(BANK_SIZE):
                assert torch.equal(res1[i][c].labels, res2[i][c].labels)
    finally:
        executor.close()

    # Closed executor raises on subsequent use
    with pytest.raises(CandidateExecutionError, match="closed"):
        executor.generate_banks(sub_views)


def test_invalid_worker_configuration() -> None:
    """Invalid worker count or threads must fail closed."""
    for invalid_workers in (-1, 1, 3, 5, "2", None):
        with pytest.raises(CandidateExecutionError):
            CandidatePoolExecutor(workers=invalid_workers)  # type: ignore[arg-type]

    for invalid_threads in (0, -1, "1", None):
        with pytest.raises(CandidateExecutionError):
            CandidatePoolExecutor(workers=2, worker_threads=invalid_threads)  # type: ignore[arg-type]


def test_input_validation_errors(
    synthetic_batch: tuple[list[FittingView], list[torch.Tensor], list[int]]
) -> None:
    """Mismatched sequence lengths or non-FittingView items raise CandidateExecutionError."""
    fitting_views, features, seeds = synthetic_batch

    # Mismatched seeds length
    with pytest.raises(CandidateExecutionError, match="Seeds count"):
        generate_candidate_banks(fitting_views, seeds=[1, 2])

    # Mismatched features length
    with pytest.raises(CandidateExecutionError, match="Features count"):
        generate_candidate_banks(fitting_views, features=features[:2], seeds=seeds)

    # Invalid item in views list
    with pytest.raises(CandidateExecutionError, match="not a FittingView"):
        generate_candidate_banks(["not_a_view"], seeds=[1])  # type: ignore[list-item]


def test_error_propagation_from_worker() -> None:
    """If a worker encounters an error, it is propagated cleanly to the parent process."""
    # Invalid FittingView fails validation cleanly
    bad_view = FittingView(
        image=torch.zeros((1, 10, 10)),
        support=torch.zeros((10, 10), dtype=torch.bool),
        context=torch.zeros((3, 10, 10)),
        study_id="bad",
        unit_id="bad_unit",
        protocol="spatial_predictive",
    )
    with pytest.raises(CandidateExecutionError, match="Invalid FittingView"):
        generate_candidate_banks([bad_view], seeds=[42], workers=2)


def test_payload_byte_bounds_exceeded(
    synthetic_batch: tuple[list[FittingView], list[torch.Tensor], list[int]]
) -> None:
    """Tensors exceeding MAX_PAYLOAD_BYTES_PER_UNIT must be rejected fail-closed."""
    fitting_views, _, seeds = synthetic_batch
    # Features exceeding 8 MiB (e.g. 10 MiB)
    huge_features = [torch.zeros((32, 512, 512), dtype=torch.float32)]
    with pytest.raises(CandidateExecutionError, match="exceeds maximum allowable payload"):
        generate_candidate_banks(fitting_views[:1], features=huge_features, seeds=seeds[:1])


def test_abrupt_worker_death_propagates_and_shutdown_completes():
    executor = CandidatePoolExecutor(workers=2)
    future = executor._get_pool().submit(os._exit, 17)
    with pytest.raises(BrokenProcessPool):
        future.result(timeout=15)
    executor.close()
    assert executor._pool is None
