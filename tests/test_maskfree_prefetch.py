"""Focused unit tests for bounded deterministic input prefetching (W2).

Tests verify:
1. ImageOnlyDataset configurable data_cache_bytes and cache_stats.
2. Synchronous (prefetch_batches=0) vs asynchronous prefetch (prefetch_batches=1, 2)
   yield bit-for-bit identical batches in identical FIFO order.
3. Prefetching does not touch torch or numpy RNG state (deterministic algorithms preserved).
4. Resume cursor (start_batch) slicing and yield_index alignment.
5. Strict validation on prefetch_batches (0, 1, 2 only; no silent clamping) and memory bounds.
6. Error at logical position: worker errors do NOT drop earlier valid batches from the queue.
7. Clean shutdown on normal termination, generator close/break, and error propagation.
8. Single Condition synchronization with zero deadlocks and verified thread stop on close.
9. dataset.iter_batches(...) equivalence with standalone iter_batches.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from self_audit_maskfree.data import (  # noqa: E402
    DEFAULT_PREFETCH_BATCHES,
    DEFAULT_PREFETCH_MAX_BYTES,
    MAX_ALLOWED_PREFETCH_BATCHES,
    ImageOnlyDataset,
    PrefetchBatchIterator,
    discover_dataset,
    iter_batches,
)
from self_audit_maskfree.data.cache import SourceDataCache  # noqa: E402
from self_audit_maskfree.data.prefetch import (  # noqa: E402
    canonical_unit_tensor_bound,
    estimate_batch_bytes,
    estimate_unit_bytes,
)

nib = pytest.importorskip("nibabel")

IMAGE_SIZE = 64
NATIVE_HW = (72, 68)
DEPTH = 5
FRAMES = 4


def _affine() -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = 1.25
    affine[1, 1] = 1.25
    affine[2, 2] = 8.0
    return affine


def _volume(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(120.0, 25.0, size=shape)).astype(np.float32)


def _write_nifti(path: Path, array: np.ndarray, *, frame_duration: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array, _affine())
    image.header.set_zooms(
        (1.25, 1.25, 8.0, frame_duration) if array.ndim == 4 else (1.25, 1.25, 8.0)
    )
    image.header.set_xyzt_units("mm", "sec" if array.ndim == 4 else None)
    nib.save(image, str(path))


def _make_synth_acdc(root: Path, n_patients: int = 2) -> Path:
    training = root / "training"
    for index in range(1, n_patients + 1):
        name = f"patient{index:03d}"
        folder = training / name
        raw = _volume(index, (*NATIVE_HW, DEPTH, FRAMES))
        _write_nifti(folder / f"{name}_4d.nii.gz", raw, frame_duration=0.035)
        for frame in (1, 2):
            _write_nifti(folder / f"{name}_frame{frame:02d}.nii.gz", raw[..., frame])
            _write_nifti(
                folder / f"{name}_frame{frame:02d}_gt.nii.gz",
                np.zeros((*NATIVE_HW, DEPTH), dtype=np.float32),
            )
        (folder / "Info.cfg").write_text("ED: 1\nES: 2\nGroup: NOR\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def shared_dataset(tmp_path_factory: pytest.TempPathFactory) -> ImageOnlyDataset:
    tmp_path = tmp_path_factory.mktemp("synth_dataset")
    root = _make_synth_acdc(tmp_path / "acdc", n_patients=2)
    manifest = discover_dataset(root, "acdc", seed=42)
    return ImageOnlyDataset(manifest, split="train", image_size=IMAGE_SIZE, seed=42)


def test_dataset_data_cache_bytes_constructor_and_stats(shared_dataset: ImageOnlyDataset) -> None:
    manifest = shared_dataset.manifest

    # Default constructor uses default cache (64MiB)
    ds_default = ImageOnlyDataset(manifest, split="train")
    assert ds_default.data_cache_bytes == 64 * 1024 * 1024
    stats = ds_default.cache_stats()
    assert isinstance(stats, dict)
    assert "hits" in stats
    assert "bytes" in stats

    # Explicit data_cache_bytes > 0
    ds_custom = ImageOnlyDataset(manifest, split="train", data_cache_bytes=32 * 1024 * 1024)
    assert ds_custom.data_cache_bytes == 32 * 1024 * 1024
    assert ds_custom.cache is not None
    assert ds_custom.cache.max_bytes == 32 * 1024 * 1024

    # Explicit data_cache_bytes == 0 disables cache
    ds_disabled = ImageOnlyDataset(manifest, split="train", data_cache_bytes=0)
    assert ds_disabled.data_cache_bytes == 0
    assert ds_disabled.cache is None
    assert ds_disabled.cache_stats() is None

    # Type/value validations
    with pytest.raises(TypeError):
        ImageOnlyDataset(manifest, split="train", data_cache_bytes="invalid")  # type: ignore
    with pytest.raises(TypeError):
        ImageOnlyDataset(manifest, split="train", data_cache_bytes=True)  # type: ignore
    with pytest.raises(ValueError):
        ImageOnlyDataset(manifest, split="train", data_cache_bytes=-1)


def test_prefetch_sync_vs_async_bit_identical(shared_dataset: ImageOnlyDataset) -> None:
    """Verify prefetch_batches=0, 1, 2 yield exactly identical batches and tensors."""
    n_units = len(shared_dataset)
    assert n_units >= 6

    # 3 batches of 2 units each
    batches = [[0, 1], [2, 3], [4, 5]]

    # Synchronous baseline
    sync_results = list(iter_batches(shared_dataset, batches, prefetch_batches=0))
    assert len(sync_results) == len(batches)

    # Async prefetch with 1 batch
    async_1_results = list(iter_batches(shared_dataset, batches, prefetch_batches=1))
    assert len(async_1_results) == len(batches)

    # Async prefetch with 2 batches
    async_2_results = list(iter_batches(shared_dataset, batches, prefetch_batches=2))
    assert len(async_2_results) == len(batches)

    for b_idx in range(len(batches)):
        sync_batch = sync_results[b_idx]
        async_1_batch = async_1_results[b_idx]
        async_2_batch = async_2_results[b_idx]

        assert len(sync_batch) == len(async_1_batch) == len(async_2_batch)

        for u_idx in range(len(sync_batch)):
            u_sync = sync_batch[u_idx]
            u_a1 = async_1_batch[u_idx]
            u_a2 = async_2_batch[u_idx]

            assert u_sync.record["unit_id"] == u_a1.record["unit_id"] == u_a2.record["unit_id"]

            # Exact tensor equality
            assert torch.equal(u_sync.fitting.image, u_a1.fitting.image)
            assert torch.equal(u_sync.fitting.image, u_a2.fitting.image)

            assert torch.equal(u_sync.fitting.support, u_a1.fitting.support)
            assert torch.equal(u_sync.fitting.support, u_a2.fitting.support)

            assert torch.equal(u_sync.fitting.context, u_a1.fitting.context)
            assert torch.equal(u_sync.fitting.context, u_a2.fitting.context)

            assert torch.equal(u_sync.selection.image, u_a1.selection.image)
            assert torch.equal(u_sync.selection.image, u_a2.selection.image)

            assert torch.equal(u_sync.selection.support, u_a1.selection.support)
            assert torch.equal(u_sync.selection.support, u_a2.selection.support)


def test_prefetch_reproducible_across_repeats(shared_dataset: ImageOnlyDataset) -> None:
    batches = [[0, 2], [1, 3], [4, 5]]
    run1 = list(iter_batches(shared_dataset, batches, prefetch_batches=2))
    run2 = list(iter_batches(shared_dataset, batches, prefetch_batches=2))

    for b1, b2 in zip(run1, run2):
        for u1, u2 in zip(b1, b2):
            assert u1.record["unit_id"] == u2.record["unit_id"]
            assert torch.equal(u1.fitting.context, u2.fitting.context)
            assert torch.equal(u1.selection.image, u2.selection.image)


def test_prefetch_does_not_touch_rng(shared_dataset: ImageOnlyDataset) -> None:
    torch.manual_seed(12345)
    np.random.seed(67890)

    torch_state_before = torch.get_rng_state()
    np_state_before = np.random.get_state()

    batches = [[0, 1], [2, 3]]
    _ = list(iter_batches(shared_dataset, batches, prefetch_batches=2))

    torch_state_after = torch.get_rng_state()
    np_state_after = np.random.get_state()

    assert torch.equal(torch_state_before, torch_state_after)
    assert np_state_before[0] == np_state_after[0]
    assert np.array_equal(np_state_before[1], np_state_after[1])


def test_prefetch_resume_cursor_and_yield_index(shared_dataset: ImageOnlyDataset) -> None:
    batches = [[0, 1], [2, 3], [4, 5], [1, 2]]
    start_batch = 2

    # With yield_index=True
    results = list(
        iter_batches(
            shared_dataset,
            batches,
            start_batch=start_batch,
            prefetch_batches=1,
            yield_index=True,
        )
    )
    assert len(results) == 2
    idx0, batch0 = results[0]
    idx1, batch1 = results[1]

    assert idx0 == 2
    assert idx1 == 3
    assert [u.record["unit_id"] for u in batch0] == [
        shared_dataset[i].record["unit_id"] for i in batches[2]
    ]
    assert [u.record["unit_id"] for u in batch1] == [
        shared_dataset[i].record["unit_id"] for i in batches[3]
    ]

    # Resume at end
    empty = list(
        iter_batches(
            shared_dataset,
            batches,
            start_batch=len(batches),
            prefetch_batches=1,
        )
    )
    assert len(empty) == 0


def test_prefetch_batches_validation_no_silent_clamping(shared_dataset: ImageOnlyDataset) -> None:
    batches = [[0, 1], [2, 3]]

    # prefetch_batches must be 0, 1, or 2; values > 2 must NOT be silently clamped
    with pytest.raises(ValueError, match="prefetch_batches must be 0, 1 or 2"):
        PrefetchBatchIterator(shared_dataset, batches, prefetch_batches=3)

    with pytest.raises(ValueError, match="prefetch_batches must be 0, 1 or 2"):
        PrefetchBatchIterator(shared_dataset, batches, prefetch_batches=10)

    with pytest.raises(ValueError, match="prefetch_batches must be 0, 1 or 2"):
        PrefetchBatchIterator(shared_dataset, batches, prefetch_batches=-1)

    with pytest.raises(TypeError):
        PrefetchBatchIterator(shared_dataset, batches, prefetch_batches="1")  # type: ignore


def test_prefetch_preserves_good_batches_before_worker_exception(
    shared_dataset: ImageOnlyDataset,
) -> None:
    """Worker error must not discard previously loaded valid batches from the queue."""
    batches = [[0, 1], [2, 3], [9999]]  # batch 0 and 1 valid, batch 2 invalid

    it = iter_batches(shared_dataset, batches, prefetch_batches=2)

    # Batch 0 must yield normally
    batch0 = next(it)
    assert len(batch0) == 2
    assert [u.record["unit_id"] for u in batch0] == [
        shared_dataset[i].record["unit_id"] for i in batches[0]
    ]

    # Batch 1 must yield normally
    batch1 = next(it)
    assert len(batch1) == 2
    assert [u.record["unit_id"] for u in batch1] == [
        shared_dataset[i].record["unit_id"] for i in batches[1]
    ]

    # Batch 2 must raise the logical error at its sequence position
    with pytest.raises((IndexError, KeyError)):
        next(it)


def test_prefetch_rejects_oversized_batch_bytes(shared_dataset: ImageOnlyDataset) -> None:
    batches = [[0, 1]]
    unit_bound = canonical_unit_tensor_bound(shared_dataset.image_size)
    batch_bound = 2 * unit_bound

    # prefetch_max_bytes smaller than batch_bound must raise ValueError
    too_small_bytes = batch_bound // 2
    it = iter_batches(shared_dataset, batches, prefetch_batches=1, prefetch_max_bytes=too_small_bytes)
    with pytest.raises(ValueError, match="exceeds prefetch_max_bytes"):
        next(it)


def test_prefetch_stats_and_peak_accounting(shared_dataset: ImageOnlyDataset) -> None:
    batches = [[0, 1], [2, 3], [4, 5]]
    it = iter_batches(shared_dataset, batches, prefetch_batches=2)

    results = list(it)
    assert len(results) == 3

    stats = it.stats()
    assert stats["max_prefetch_batches"] == 2
    assert stats["total_batches_prefetched"] == 3
    assert stats["total_units_prefetched"] == 6
    assert stats["peak_queued_bytes"] > 0
    assert stats["queued_bytes"] == 0  # Drained at completion


def test_oversized_later_batch_keeps_prior_batch_and_closes(shared_dataset: ImageOnlyDataset) -> None:
    unit_bound = canonical_unit_tensor_bound(shared_dataset.image_size)
    it = iter_batches(shared_dataset, [[0], [1, 2]], prefetch_batches=1,
                      prefetch_max_bytes=unit_bound)
    assert len(next(it)) == 1
    with pytest.raises(ValueError, match="exceeds prefetch_max_bytes"):
        next(it)
    assert it.stats()["queue_depth"] == 0
    assert not it._thread.is_alive()


def test_prefetch_clean_shutdown_on_early_break(shared_dataset: ImageOnlyDataset) -> None:
    batches = [[0, 1], [2, 3], [4, 5]]
    it = iter_batches(shared_dataset, batches, prefetch_batches=2)

    for i, batch in enumerate(it):
        assert len(batch) == 2
        break

    it.close()
    if it._thread is not None:
        assert not it._thread.is_alive()


def test_dataset_iter_batches_method(shared_dataset: ImageOnlyDataset) -> None:
    batches = [[0, 1], [2, 3]]
    method_results = list(shared_dataset.iter_batches(batches, prefetch_batches=1))
    standalone_results = list(iter_batches(shared_dataset, batches, prefetch_batches=1))

    assert len(method_results) == len(standalone_results) == 2
    for b_m, b_s in zip(method_results, standalone_results):
        for u_m, u_s in zip(b_m, b_s):
            assert u_m.record["unit_id"] == u_s.record["unit_id"]
            assert torch.equal(u_m.fitting.context, u_s.fitting.context)


def test_estimate_bytes_calculation(shared_dataset: ImageOnlyDataset) -> None:
    unit = shared_dataset[0]
    unit_bytes = estimate_unit_bytes(unit)
    assert unit_bytes > 0

    batch_bytes = estimate_batch_bytes([unit, shared_dataset[1]])
    assert batch_bytes >= unit_bytes * 2
