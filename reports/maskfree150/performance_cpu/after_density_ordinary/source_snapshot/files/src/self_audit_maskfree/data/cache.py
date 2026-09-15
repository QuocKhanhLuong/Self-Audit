"""Bounded process-local cache for immutable image source frames.

The mask-free data path repeatedly reads neighbouring slices from the same
source frame.  This module caches only the decoded native rank-3 frame, before
any role partitioning, fit-only statistics, normalization, or resizing.  The
cache is deliberately process-local and bounded; it is not a DataLoader or
checkpoint/resume mechanism.

Every entry is keyed by the complete source identity available to the data
layer: canonical path, manifest identity and digest, declared source/frame
fingerprints, frame/depth selectors, and the current filesystem stat
generation.  A source whose stat generation changes is a cache miss and is
decoded again.  The stat is checked both before and after a decode so a source
being replaced while it is read is never published as a valid cache entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .firewall import assert_image_only
from .geometry import read_slice_stack, read_source_frame


DEFAULT_MAX_BYTES = 64 * 1024 * 1024


class SourceCacheError(RuntimeError):
    """Raised when a decoded source cannot satisfy the cache contract."""


class SourceMutationError(SourceCacheError):
    """Raised when source bytes change during one decode transaction."""


@dataclass(frozen=True)
class FileStatIdentity:
    """The stat generation used to reject stale source entries."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_path(cls, path: str | Path) -> "FileStatIdentity":
        stat = os.stat(path)
        return cls(
            device=int(getattr(stat, "st_dev", 0)),
            inode=int(getattr(stat, "st_ino", 0)),
            size=int(stat.st_size),
            mtime_ns=int(getattr(stat, "st_mtime_ns", round(float(stat.st_mtime) * 1e9))),
            ctime_ns=int(getattr(stat, "st_ctime_ns", round(float(stat.st_ctime) * 1e9))),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class SourceCacheKey:
    """Immutable identity of one decoded source frame."""

    manifest_digest: str
    manifest_id: str | None
    source_hash: str | None
    frame_fingerprint: str | None
    path: str
    stat: FileStatIdentity
    frame_axis: int | None
    depth_axis: int
    frame_index: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_digest": self.manifest_digest,
            "manifest_id": self.manifest_id,
            "source_hash": self.source_hash,
            "frame_fingerprint": self.frame_fingerprint,
            "path": self.path,
            "stat": self.stat.as_dict(),
            "frame_axis": self.frame_axis,
            "depth_axis": self.depth_axis,
            "frame_index": self.frame_index,
        }


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Return a stable digest for the complete image manifest mapping."""

    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def decode_source_frame(record: Mapping[str, Any]) -> np.ndarray:
    """Decode one native frame through the canonical geometry reader."""

    return read_source_frame(
        record["path"],
        depth_axis=int(record["depth_axis"]),
        frame_index=record.get("frame_index"),
        frame_axis=record.get("frame_axis"),
    )


FrameLoader = Callable[[Mapping[str, Any]], np.ndarray]


class SourceDataCache:
    """A bounded LRU cache of immutable decoded native source frames."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("max_bytes must be an integer")
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict[SourceCacheKey, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._invalidations = 0
        self._loads = 0
        self._loaded_bytes = 0
        self._uncacheable = 0
        self._lock = threading.RLock()

    @staticmethod
    def _canonical_path(record: Mapping[str, Any]) -> Path:
        # Gate the caller's path before resolving symlinks.  The data firewall
        # is path-shaped by design, so a forbidden source name is never hidden
        # behind a canonicalisation step.
        return assert_image_only(record["path"]).expanduser().resolve()

    def _key(
        self,
        record: Mapping[str, Any],
        *,
        manifest_digest_value: str,
        manifest_id: str | None,
    ) -> SourceCacheKey:
        path = self._canonical_path(record)
        frame_axis_value = record.get("frame_axis")
        frame_axis = None if frame_axis_value is None else int(frame_axis_value)
        frame_index_value = record.get("frame_index")
        frame_index = None if frame_index_value is None else int(frame_index_value)
        source_hash_value = record.get("source_hash", record.get("source_fingerprint"))
        source_hash = None if source_hash_value is None else str(source_hash_value)
        fingerprint_value = record.get("frame_fingerprint")
        frame_fingerprint = None if fingerprint_value is None else str(fingerprint_value)
        return SourceCacheKey(
            manifest_digest=str(manifest_digest_value),
            manifest_id=None if manifest_id is None else str(manifest_id),
            source_hash=source_hash,
            frame_fingerprint=frame_fingerprint,
            path=str(path),
            stat=FileStatIdentity.from_path(path),
            frame_axis=frame_axis,
            depth_axis=int(record["depth_axis"]),
            frame_index=frame_index,
        )

    def _drop_stale(self, key: SourceCacheKey) -> None:
        stale = [
            candidate
            for candidate in self._entries
            if candidate.path == key.path
            and candidate.manifest_digest == key.manifest_digest
            and candidate.manifest_id == key.manifest_id
            and candidate.source_hash == key.source_hash
            and candidate.frame_fingerprint == key.frame_fingerprint
            and candidate.frame_axis == key.frame_axis
            and candidate.depth_axis == key.depth_axis
            and candidate.frame_index == key.frame_index
            and candidate.stat != key.stat
        ]
        for candidate in stale:
            value = self._entries.pop(candidate)
            self._bytes -= int(value.nbytes)
            self._invalidations += 1

    def _insert(self, key: SourceCacheKey, value: np.ndarray) -> None:
        size = int(value.nbytes)
        if self.max_bytes == 0 or size > self.max_bytes:
            return
        while self._entries and self._bytes + size > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= int(evicted.nbytes)
            self._evictions += 1
        self._entries[key] = value
        self._bytes += size

    def get_frame(
        self,
        record: Mapping[str, Any],
        *,
        manifest_digest_value: str,
        manifest_id: str | None = None,
        loader: FrameLoader | None = None,
        _copy: bool = True,
    ) -> np.ndarray:
        """Return a private clone of a decoded frame, loading it if needed.

        ``loader`` is intended for deterministic tests and must return the same
        native rank-3 float data as :func:`decode_source_frame`.  Production
        callers use the exact decoder above.
        """

        with self._lock:
            key = self._key(
                record,
                manifest_digest_value=manifest_digest_value,
                manifest_id=manifest_id,
            )
            self._drop_stale(key)
            cached = self._entries.get(key)
            if cached is not None:
                # Re-stat immediately before returning.  This closes the race
                # where a source is replaced between key construction and the
                # cache hit; a changed generation is never served as a hit.
                current = FileStatIdentity.from_path(key.path)
                if current == key.stat:
                    self._entries.move_to_end(key)
                    self._hits += 1
                    return np.array(cached, copy=True) if _copy else cached
                self._entries.pop(key, None)
                self._bytes -= int(cached.nbytes)
                self._invalidations += 1

            self._misses += 1
            before = FileStatIdentity.from_path(key.path)
            decoded = (loader or decode_source_frame)(record)
            after = FileStatIdentity.from_path(key.path)
            if after != before or before != key.stat:
                self._invalidations += 1
                raise SourceMutationError(
                    f"image source changed while decoding; refusing cache entry: {key.path}"
                )
            # Own the resident storage before marking it read-only.  A custom
            # deterministic loader may return its reusable input array; making
            # that object readonly in place would leak cache ownership back to
            # the caller and violate the no-alias contract.
            array = np.array(decoded, copy=True, order="C")
            if array.ndim != 3:
                raise SourceCacheError(
                    f"decoded source frame has shape {array.shape}, expected rank 3"
                )
            if not np.issubdtype(array.dtype, np.number):
                raise SourceCacheError(f"decoded source frame has non-numeric dtype {array.dtype}")
            array.setflags(write=False)
            self._loads += 1
            self._loaded_bytes += int(array.nbytes)
            self._insert(key, array)
            return np.array(array, copy=True) if _copy else array

    @staticmethod
    def _estimated_frame_bytes(record: Mapping[str, Any]) -> int | None:
        """Estimate the float32 resident size from frozen manifest metadata."""
        shape_value = record.get("shape")
        if not isinstance(shape_value, (list, tuple)):
            shape_value = record.get("native_shape")
        if not isinstance(shape_value, (list, tuple)):
            return None
        shape = [int(value) for value in shape_value]
        if len(shape) == 4:
            frame_axis_value = record.get("frame_axis")
            frame_axis = 3 if frame_axis_value is None else int(frame_axis_value)
            if frame_axis < 0 or frame_axis >= len(shape):
                return None
            shape.pop(frame_axis)
        if len(shape) != 3 or any(value < 0 for value in shape):
            return None
        return int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize

    def _frame_exceeds_budget(self, record: Mapping[str, Any]) -> bool:
        estimate = self._estimated_frame_bytes(record)
        return estimate is not None and estimate > self.max_bytes

    def get_slice_stack(
        self,
        record: Mapping[str, Any],
        *,
        manifest_digest_value: str,
        manifest_id: str | None = None,
        loader: FrameLoader | None = None,
    ) -> np.ndarray:
        """Return the exact three-plane stack used by the data layer."""

        # A full frame larger than the resident byte budget must not turn a
        # bounded cache into a repeated full-frame decode regression. Fall back
        # to the original three-plane reader in that case.
        if self._frame_exceeds_budget(record):
            with self._lock:
                self._misses += 1
                self._uncacheable += 1
                path = self._canonical_path(record)
                before = FileStatIdentity.from_path(path)
                stack = read_slice_stack(
                    record["path"],
                    depth_axis=int(record["depth_axis"]),
                    slice_index=int(record["slice_index"]),
                    frame_index=record.get("frame_index"),
                    frame_axis=record.get("frame_axis"),
                )
                after = FileStatIdentity.from_path(path)
                if after != before:
                    self._invalidations += 1
                    raise SourceMutationError(
                        f"image source changed while reading; refusing uncached result: {path}"
                    )
                return np.ascontiguousarray(stack)
        frame = self.get_frame(
            record,
            manifest_digest_value=manifest_digest_value,
            manifest_id=manifest_id,
            loader=loader,
            _copy=False,
        )
        depth_axis = int(record["depth_axis"])
        frame_axis_value = record.get("frame_axis")
        frame_axis = None if frame_axis_value is None else int(frame_axis_value)
        # Removing an axis before the depth axis shifts the depth index by one.
        effective_depth_axis = depth_axis
        source_shape = record.get("shape")
        source_is_rank4 = isinstance(source_shape, (list, tuple)) and len(source_shape) == 4
        if source_is_rank4 and frame_axis is not None and frame_axis < depth_axis:
            effective_depth_axis -= 1
        if effective_depth_axis < 0 or effective_depth_axis >= 3:
            raise SourceCacheError(
                f"depth_axis {depth_axis} invalid after frame selection (axis {effective_depth_axis})"
            )
        depth = int(frame.shape[effective_depth_axis])
        slice_index = int(record["slice_index"])
        if depth <= 0 or not 0 <= slice_index < depth:
            raise SourceCacheError(f"slice_index {slice_index} outside depth extent {depth}")
        planes = []
        for offset in (-1, 0, 1):
            z = int(np.clip(slice_index + offset, 0, depth - 1))
            planes.append(np.take(frame, z, axis=effective_depth_axis))
        stack = np.stack(planes, axis=0)
        if stack.ndim != 3:
            raise SourceCacheError(
                f"slice extraction produced shape {stack.shape}, expected [3,H,W]"
            )
        return np.ascontiguousarray(stack)

    def clear(self, *, reset_stats: bool = True) -> None:
        """Drop all resident arrays; optionally preserve cumulative counters."""

        with self._lock:
            self._entries.clear()
            self._bytes = 0
            if reset_stats:
                self._hits = 0
                self._misses = 0
                self._evictions = 0
                self._invalidations = 0
                self._loads = 0
                self._loaded_bytes = 0
                self._uncacheable = 0

    def stats(self) -> dict[str, int]:
        """Return counters and current resident bytes for reports/tests."""

        with self._lock:
            return {
                "hits": int(self._hits),
                "misses": int(self._misses),
                "bytes": int(self._bytes),
                "evictions": int(self._evictions),
                "invalidations": int(self._invalidations),
                "loads": int(self._loads),
                "loaded_bytes": int(self._loaded_bytes),
                "uncacheable": int(self._uncacheable),
                "entries": int(len(self._entries)),
                "max_bytes": int(self.max_bytes),
            }

    snapshot = stats

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_DEFAULT_SOURCE_CACHE = SourceDataCache()


def default_source_cache() -> SourceDataCache:
    """Return the bounded process-local cache shared by public data helpers."""

    return _DEFAULT_SOURCE_CACHE


def reset_default_source_cache() -> None:
    """Clear the default cache and all counters (primarily for test isolation)."""

    _DEFAULT_SOURCE_CACHE.clear(reset_stats=True)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "FileStatIdentity",
    "SourceCacheError",
    "SourceCacheKey",
    "SourceDataCache",
    "SourceMutationError",
    "decode_source_frame",
    "default_source_cache",
    "manifest_digest",
    "reset_default_source_cache",
]
