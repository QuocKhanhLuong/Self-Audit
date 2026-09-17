"""Bounded source-frame cache checks for the mask-free image-only loader.

These are deterministic CPU software checks.  The cache stores only decoded
native intensities; fit-only statistics, role masks, resizing, and all
verification data remain outside the cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from self_audit_maskfree.data import discover_dataset, load_full_input  # noqa: E402
from self_audit_maskfree.data import cache as cache_module  # noqa: E402
from self_audit_maskfree.data import geometry as geometry_module  # noqa: E402
from self_audit_maskfree.data.cache import (  # noqa: E402
    SourceDataCache,
    SourceMutationError,
    manifest_digest,
)
from self_audit_maskfree.data.geometry import read_slice_stack  # noqa: E402


def _write_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(array), np.eye(4, dtype=np.float64))
    image.header.set_zooms(tuple(1.0 for _ in array.shape))
    image.header.set_xyzt_units("mm", "sec" if array.ndim == 4 else None)
    nib.save(image, str(path))


def _record(path: Path, *, depth_axis: int = 2, frame_index: int = 0,
            frame_axis: int | None = None, slice_index: int = 0) -> dict[str, object]:
    return {
        "path": str(path),
        "depth_axis": depth_axis,
        "slice_index": slice_index,
        "frame_index": frame_index,
        "frame_axis": frame_axis,
        "source_hash": "source-v1",
        "frame_fingerprint": "frame-v1",
    }


def test_cold_warm_frame_equality_and_no_alias_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.npy"
    array = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    np.save(source, array)
    record = _record(source)
    calls = []

    def loader(_record):
        calls.append(True)
        return array

    cache = SourceDataCache(max_bytes=array.nbytes * 2)
    first = cache.get_frame(record, manifest_digest_value="manifest-a", manifest_id="a", loader=loader)
    first[0, 0, 0] = -999.0
    second = cache.get_frame(record, manifest_digest_value="manifest-a", manifest_id="a", loader=loader)

    assert len(calls) == 1
    assert np.array_equal(second, array)
    assert array.flags.writeable
    assert second.flags.writeable
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1
    assert cache.stats()["bytes"] == array.nbytes


def test_real_nifti_frame_and_stack_match_uncached_geometry(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    array = np.arange(6 * 7 * 5 * 2, dtype=np.float32).reshape(6, 7, 5, 2)
    _write_nifti(source, array)
    record = _record(source, frame_index=1, slice_index=3)
    cache = SourceDataCache(max_bytes=1 << 20)

    expected = read_slice_stack(
        source, depth_axis=2, slice_index=3, frame_index=1, frame_axis=None
    )
    actual = cache.get_slice_stack(
        record, manifest_digest_value="manifest-a", manifest_id="a"
    )
    again = cache.get_slice_stack(
        {**record, "slice_index": 0}, manifest_digest_value="manifest-a", manifest_id="a"
    )

    assert np.array_equal(actual, expected)
    assert np.array_equal(again, read_slice_stack(
        source, depth_axis=2, slice_index=0, frame_index=1, frame_axis=None
    ))
    assert cache.stats()["loads"] == 1
    assert cache.stats()["hits"] == 1


def test_same_volume_repeated_loads_are_eliminated_by_frame_cache(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "training" / "patient001" / "patient001_sa.nii.gz"
    array = np.arange(8 * 9 * 4, dtype=np.float32).reshape(8, 9, 4)
    _write_nifti(source, array)
    manifest = discover_dataset(tmp_path, "acdc", seed=42)
    unit = next(record for record in manifest["records"] if record["path"] == str(source))
    cache = SourceDataCache(max_bytes=1 << 20)
    native_load = nib.load
    calls = []

    def counted(path, *args, **kwargs):
        calls.append(str(path))
        return native_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", counted)
    first, first_record = load_full_input(
        manifest, unit["unit_id"], image_size=8, cache=cache
    )
    second, second_record = load_full_input(
        manifest, unit["unit_id"], image_size=8, cache=cache
    )
    uncached, _ = load_full_input(
        manifest, unit["unit_id"], image_size=8, cache=None
    )

    assert len(calls) == 2  # one cached decode plus the explicit uncached control
    assert torch.equal(first, second)
    assert torch.equal(first, uncached)
    assert first_record == second_record
    assert cache.stats()["loads"] == 1
    assert cache.stats()["hits"] == 1


def test_stat_generation_and_manifest_digest_invalidate_entries(tmp_path: Path) -> None:
    source = tmp_path / "source.npy"
    first_array = np.ones((4, 5, 3), dtype=np.float32)
    second_array = np.full((4, 5, 3), 7.0, dtype=np.float32)
    np.save(source, first_array)
    record = _record(source)
    cache = SourceDataCache(max_bytes=1 << 20)

    first = cache.get_frame(record, manifest_digest_value="manifest-a", manifest_id="a")
    np.save(source, second_array)
    second = cache.get_frame(record, manifest_digest_value="manifest-a", manifest_id="a")
    third = cache.get_frame(record, manifest_digest_value="manifest-b", manifest_id="b")

    assert np.array_equal(first, first_array)
    assert np.array_equal(second, second_array)
    assert np.array_equal(third, second_array)
    stats = cache.stats()
    assert stats["misses"] == 3
    assert stats["invalidations"] >= 1
    assert stats["hits"] == 0


def test_lru_ram_bound_reports_eviction_and_oversized_skip(tmp_path: Path) -> None:
    frame = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    records = []
    for index in range(3):
        source = tmp_path / f"source{index}.npy"
        np.save(source, frame + index)
        records.append(_record(source))
    cache = SourceDataCache(max_bytes=frame.nbytes * 2)

    for record in records:
        cache.get_frame(record, manifest_digest_value="manifest", manifest_id="id")
    stats = cache.stats()
    assert stats["bytes"] <= stats["max_bytes"]
    assert stats["entries"] <= 2
    assert stats["evictions"] >= 1

    oversized = SourceDataCache(max_bytes=frame.nbytes - 1)
    oversized_record = {**records[0], "shape": list(frame.shape)}
    oversized.get_slice_stack(oversized_record, manifest_digest_value="manifest", manifest_id="id")
    assert oversized.stats()["bytes"] == 0
    assert oversized.stats()["entries"] == 0
    assert oversized.stats()["uncacheable"] == 1


def test_rank3_legacy_frame_selectors_remain_no_ops(tmp_path: Path) -> None:
    source = tmp_path / "source.npy"
    array = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    np.save(source, array)
    record = {
        **_record(source, slice_index=1),
        "shape": list(array.shape),
        "frame_index": None,
        "frame_axis": 0,
    }
    cache = SourceDataCache(max_bytes=1 << 20)
    expected = read_slice_stack(source, depth_axis=2, slice_index=1, frame_index=None, frame_axis=0)
    actual = cache.get_slice_stack(record, manifest_digest_value="manifest", manifest_id="id")
    assert np.array_equal(actual, expected)


def test_rank4_nontrailing_frame_axis_matches_geometry(tmp_path: Path) -> None:
    source = tmp_path / "source.npy"
    # Axis 0 is frame, axis 2 is depth; the helper must shift depth after
    # removing the frame axis while retaining H/W order.
    array = np.arange(2 * 4 * 5 * 6, dtype=np.float32).reshape(2, 4, 5, 6)
    np.save(source, array)
    record = {
        **_record(source, depth_axis=2, frame_index=1, frame_axis=0, slice_index=3),
        "shape": list(array.shape),
    }
    cache = SourceDataCache(max_bytes=1 << 20)
    expected = read_slice_stack(source, depth_axis=2, slice_index=3, frame_index=1, frame_axis=0)
    actual = cache.get_slice_stack(record, manifest_digest_value="manifest", manifest_id="id")
    assert np.array_equal(actual, expected)


def test_decode_rejects_source_mutation_before_publishing(tmp_path: Path) -> None:
    source = tmp_path / "source.npy"
    np.save(source, np.zeros((4, 4, 2), dtype=np.float32))
    record = _record(source)
    cache = SourceDataCache(max_bytes=1 << 20)

    def mutating_loader(_record):
        source.write_bytes(source.read_bytes() + b"mutation")
        return np.zeros((4, 4, 2), dtype=np.float32)

    with pytest.raises(SourceMutationError):
        cache.get_frame(record, manifest_digest_value="manifest", manifest_id="id", loader=mutating_loader)
    assert cache.stats()["entries"] == 0
    assert cache.stats()["invalidations"] >= 1


def test_oversized_fallback_rejects_source_mutation(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.npy"
    array = np.zeros((4, 4, 2), dtype=np.float32)
    np.save(source, array)
    record = {**_record(source), "shape": list(array.shape)}
    cache = SourceDataCache(max_bytes=array.nbytes - 1)
    original = cache_module.read_slice_stack

    def mutating_stack(*args, **kwargs):
        value = original(*args, **kwargs)
        source.write_bytes(source.read_bytes() + b"mutation")
        return value

    monkeypatch.setattr(cache_module, "read_slice_stack", mutating_stack)
    with pytest.raises(SourceMutationError):
        cache.get_slice_stack(record, manifest_digest_value="manifest", manifest_id="id")
    assert cache.stats()["uncacheable"] == 1
    assert cache.stats()["invalidations"] >= 1


def test_oversized_fallback_reads_only_three_source_planes(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.npy"
    array = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    np.save(source, array)
    record = {**_record(source, slice_index=1), "shape": list(array.shape)}
    requests: list[object] = []

    class SpyProxy:
        shape = array.shape

        def __getitem__(self, key):
            requests.append(key)
            # A fallback is bounded only if it never materialises the whole
            # source frame.  Every legitimate request fixes the depth index.
            if isinstance(key, tuple) and all(isinstance(item, slice) for item in key):
                raise AssertionError("fallback attempted full-frame slicing")
            return array[key]

    proxy = SpyProxy()
    monkeypatch.setattr(geometry_module.np, "load", lambda *args, **kwargs: proxy)
    monkeypatch.setattr(cache_module, "read_slice_stack", geometry_module.read_slice_stack)
    cache = SourceDataCache(max_bytes=array.nbytes - 1)

    actual = cache.get_slice_stack(record, manifest_digest_value="manifest", manifest_id="id")

    expected_keys = [
        (slice(None), slice(None), 0),
        (slice(None), slice(None), 1),
        (slice(None), slice(None), 2),
    ]
    assert len(requests) == 3
    assert requests == expected_keys
    assert np.array_equal(actual, np.stack([array[:, :, 0], array[:, :, 1], array[:, :, 2]]))
    assert cache.stats()["uncacheable"] == 1


def test_cache_does_not_consume_rng_and_preserves_slice_order(tmp_path: Path) -> None:
    source = tmp_path / "source.npy"
    array = np.arange(6 * 7 * 5, dtype=np.float32).reshape(6, 7, 5)
    np.save(source, array)
    record = _record(source)
    cache = SourceDataCache(max_bytes=1 << 20)
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    first = cache.get_slice_stack({**record, "slice_index": 0}, manifest_digest_value="m", manifest_id="id")
    second = cache.get_slice_stack({**record, "slice_index": 4}, manifest_digest_value="m", manifest_id="id")
    after = torch.get_rng_state()

    assert torch.equal(before, after)
    assert np.array_equal(first, read_slice_stack(source, depth_axis=2, slice_index=0, frame_index=0))
    assert np.array_equal(second, read_slice_stack(source, depth_axis=2, slice_index=4, frame_index=0))


def test_manifest_digest_is_order_stable() -> None:
    assert manifest_digest({"b": 2, "a": [1, 3]}) == manifest_digest({"a": [1, 3], "b": 2})
