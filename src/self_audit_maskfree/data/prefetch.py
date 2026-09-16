"""Bounded deterministic single CPU thread input batch prefetcher.

Loads input batches ahead of time using one dedicated worker thread.
Guarantees:
- Deterministic batch ordering and unit ordering (strict FIFO).
- Single-thread CPU execution; no multiprocessing, no worker pools.
- Zero RNG consumption (no torch/numpy RNG calls or seed changes).
- Zero model execution or inference; input loading only.
- Bounded queued/inflight output tensors and queue depth (1-2 batches).
- Concurrency single-ownership: dataset items are loaded solely by the prefetch thread
  while active; foreground consumes prefetched batches.
- Single Condition synchronization; thread joins strictly outside lock.
- Pre-checks canonical batch tensor bounds before loading and preserves inflight byte accounting.
- Strict bound enforcement: actual loaded bytes must not exceed the reserved batch tensor bound.
- Logical-order error preservation: worker errors do not discard prior valid batches in queue.
- Clean shutdown on normal termination, generator close/break, or errors (clears queue/bytes and verifies thread stop).
"""
from __future__ import annotations

import collections
import threading
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import torch

if TYPE_CHECKING:
    from ..contracts import TrainingUnit
    from .dataset import ImageOnlyDataset

DEFAULT_PREFETCH_BATCHES: int = 0
DEFAULT_PREFETCH_MAX_BYTES: int = 32 * 1024 * 1024  # 33554432 (32 MiB)
MAX_ALLOWED_PREFETCH_BATCHES: int = 2
CANONICAL_BYTES_PER_PIXEL: int = 22  # (1*4 + 1 + 3*4 + 1*4 + 1) for fitting & selection views


def canonical_unit_tensor_bound(image_size: int) -> int:
    """Return canonical tensor byte bound for one TrainingUnit at image_size."""
    return CANONICAL_BYTES_PER_PIXEL * int(image_size) * int(image_size)


def estimate_unit_bytes(unit: TrainingUnit) -> int:
    """Return actual memory resident bytes for a TrainingUnit's tensors."""
    total = 0
    for tensor in (
        unit.fitting.image,
        unit.fitting.support,
        unit.fitting.context,
        unit.selection.image,
        unit.selection.support,
    ):
        if isinstance(tensor, torch.Tensor):
            total += int(tensor.element_size() * tensor.nelement())
    return total


def estimate_batch_bytes(units: Sequence[TrainingUnit]) -> int:
    """Return memory resident bytes for a batch of TrainingUnits."""
    return sum(estimate_unit_bytes(u) for u in units)


class PrefetchBatchIterator:
    """Bounded deterministic single CPU thread input batch prefetcher."""

    def __init__(
        self,
        dataset: ImageOnlyDataset,
        batches: Sequence[Sequence[int]],
        *,
        prefetch_batches: int = DEFAULT_PREFETCH_BATCHES,
        prefetch_max_bytes: int = DEFAULT_PREFETCH_MAX_BYTES,
        start_batch: int = 0,
        yield_index: bool = False,
    ) -> None:
        if isinstance(prefetch_batches, bool) or not isinstance(prefetch_batches, int):
            raise TypeError("prefetch_batches must be an integer")
        if prefetch_batches not in (0, 1, 2):
            raise ValueError(
                f"prefetch_batches must be 0, 1 or 2 (bounded input lookahead), got {prefetch_batches}"
            )
        if isinstance(prefetch_max_bytes, bool) or not isinstance(prefetch_max_bytes, int):
            raise TypeError("prefetch_max_bytes must be an integer")
        if prefetch_max_bytes <= 0:
            raise ValueError("prefetch_max_bytes must be positive")
        if isinstance(start_batch, bool) or not isinstance(start_batch, int):
            raise TypeError("start_batch must be an integer")
        if start_batch < 0:
            raise ValueError("start_batch must be non-negative")

        image_size = getattr(dataset, "image_size", None)
        if image_size is None or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("prefetch requires dataset with positive integer image_size attribute")

        self.dataset = dataset
        self.start_batch = start_batch
        self.prefetch_batches = prefetch_batches
        self.prefetch_max_bytes = prefetch_max_bytes
        self.yield_index = yield_index
        self._unit_tensor_bound = canonical_unit_tensor_bound(image_size)

        if start_batch > 0:
            self.batches: list[Sequence[int]] = list(batches[start_batch:])
        else:
            self.batches = list(batches)

        self._closed = False
        self._started = False
        self._sync_cursor = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Single Condition protecting all state
        self._cond = threading.Condition()
        self._queue: collections.deque[
            tuple[int, list[TrainingUnit] | None, int, BaseException | None]
        ] = collections.deque()
        self._queued_bytes = 0
        self._inflight_bytes = 0
        self._peak_queued_bytes = 0
        self._total_batches_prefetched = 0
        self._total_units_prefetched = 0
        self._completed = False

    def __len__(self) -> int:
        return len(self.batches)

    @property
    def peak_queued_bytes(self) -> int:
        with self._cond:
            return int(self._peak_queued_bytes)

    def stats(self) -> dict[str, int]:
        """Return live prefetch accounting and queue metrics."""
        with self._cond:
            return {
                "queued_bytes": int(self._queued_bytes),
                "peak_queued_bytes": int(self._peak_queued_bytes),
                "inflight_bytes": int(self._inflight_bytes),
                "queue_depth": int(len(self._queue)),
                "max_prefetch_batches": int(self.prefetch_batches),
                "prefetch_max_bytes": int(self.prefetch_max_bytes),
                "total_batches_prefetched": int(self._total_batches_prefetched),
                "total_units_prefetched": int(self._total_units_prefetched),
            }

    def _worker(self) -> None:
        for offset, batch_indices in enumerate(self.batches):
            batch_idx = self.start_batch + offset
            batch_tensor_bound = len(batch_indices) * self._unit_tensor_bound

            # If batch tensor bound alone exceeds budget, reject at its logical position
            if batch_tensor_bound > self.prefetch_max_bytes:
                err = ValueError(
                    f"batch tensor bound ({batch_tensor_bound} bytes) exceeds "
                    f"prefetch_max_bytes ({self.prefetch_max_bytes} bytes)"
                )
                with self._cond:
                    while len(self._queue) >= self.prefetch_batches and not self._stop_event.is_set():
                        self._cond.wait(timeout=0.05)
                    if not self._stop_event.is_set():
                        self._queue.append((batch_idx, None, 0, err))
                        self._completed = True
                        self._cond.notify_all()
                return

            with self._cond:
                while not self._stop_event.is_set():
                    # Can produce if:
                    # 1) queue depth < prefetch_batches limit AND
                    # 2) either nothing is queued/inflight OR adding this batch won't exceed prefetch_max_bytes
                    room_in_queue = len(self._queue) < self.prefetch_batches
                    room_in_bytes = (
                        (len(self._queue) == 0 and self._inflight_bytes == 0)
                        or (self._queued_bytes + self._inflight_bytes + batch_tensor_bound <= self.prefetch_max_bytes)
                    )
                    if room_in_queue and room_in_bytes:
                        break
                    self._cond.wait(timeout=0.05)

                if self._stop_event.is_set():
                    break

                # Reserve inflight bytes before loading
                self._inflight_bytes += batch_tensor_bound

            try:
                # Strictly deterministic input loading; single CPU thread, no RNG, no model
                units = [self.dataset[i] for i in batch_indices]
                actual_batch_bytes = estimate_batch_bytes(units)
                if actual_batch_bytes > batch_tensor_bound:
                    raise ValueError(
                        f"actual batch tensor bytes ({actual_batch_bytes}) exceeds "
                        f"reserved bound ({batch_tensor_bound})"
                    )
            except BaseException as exc:
                with self._cond:
                    self._inflight_bytes = max(0, self._inflight_bytes - batch_tensor_bound)
                    if not self._stop_event.is_set():
                        self._queue.append((batch_idx, None, 0, exc))
                        self._completed = True
                        self._cond.notify_all()
                return

            with self._cond:
                self._inflight_bytes = max(0, self._inflight_bytes - batch_tensor_bound)
                if self._stop_event.is_set():
                    break
                self._queue.append((batch_idx, units, actual_batch_bytes, None))
                self._queued_bytes += actual_batch_bytes
                if self._queued_bytes > self._peak_queued_bytes:
                    self._peak_queued_bytes = self._queued_bytes
                self._total_batches_prefetched += 1
                self._total_units_prefetched += len(units)
                self._cond.notify_all()

        with self._cond:
            self._completed = True
            self._cond.notify_all()

    def _ensure_worker_started(self) -> None:
        if not self._started:
            self._started = True
            self._thread = threading.Thread(
                target=self._worker,
                name="maskfree-input-prefetcher",
                daemon=True,
            )
            self._thread.start()

    def __iter__(self) -> "PrefetchBatchIterator":
        if self._closed:
            raise RuntimeError("cannot iterate over a closed PrefetchBatchIterator")
        if self.prefetch_batches > 0:
            self._ensure_worker_started()
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration

        # Synchronous path: 0 background threads
        if self.prefetch_batches <= 0:
            if self._sync_cursor >= len(self.batches):
                self.close()
                raise StopIteration
            batch_indices = self.batches[self._sync_cursor]
            batch_idx = self.start_batch + self._sync_cursor
            batch_tensor_bound = len(batch_indices) * self._unit_tensor_bound
            if batch_tensor_bound > self.prefetch_max_bytes:
                self.close()
                raise ValueError(
                    f"batch tensor bound ({batch_tensor_bound} bytes) exceeds "
                    f"prefetch_max_bytes ({self.prefetch_max_bytes} bytes)"
                )
            self._sync_cursor += 1
            units = [self.dataset[i] for i in batch_indices]
            actual_bytes = estimate_batch_bytes(units)
            if actual_bytes > batch_tensor_bound:
                self.close()
                raise ValueError(
                    f"actual batch tensor bytes ({actual_bytes}) exceeds "
                    f"reserved bound ({batch_tensor_bound})"
                )
            return (batch_idx, units) if self.yield_index else units

        # Asynchronous prefetch path
        self._ensure_worker_started()
        should_close = False
        batch_idx = 0
        units = None
        exc = None

        with self._cond:
            while (
                not self._queue
                and not (self._completed and not self._queue)
                and not self._stop_event.is_set()
            ):
                self._cond.wait(timeout=0.05)

            if not self._queue:
                if self._stop_event.is_set() or self._completed:
                    should_close = True
                else:
                    raise RuntimeError("prefetch queue unexpectedly empty")
            else:
                batch_idx, units, batch_bytes, exc = self._queue.popleft()
                self._queued_bytes -= batch_bytes
                self._cond.notify_all()

        # ALL close calls occur strictly OUTSIDE with self._cond
        if should_close:
            self.close()
            raise StopIteration

        if exc is not None:
            self.close()
            raise exc

        assert units is not None
        return (batch_idx, units) if self.yield_index else units

    def close(self) -> None:
        """Signal background worker to stop, clear queue/bytes, and wait for thread stop."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            self._queue.clear()
            self._queued_bytes = 0
            self._inflight_bytes = 0
            self._cond.notify_all()

        # Join strictly OUTSIDE condition lock to guarantee zero deadlocks
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("prefetch worker thread failed to stop within timeout")

    def __enter__(self) -> "PrefetchBatchIterator":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def iter_batches(
    dataset: ImageOnlyDataset,
    batches: Sequence[Sequence[int]],
    *,
    prefetch_batches: int = DEFAULT_PREFETCH_BATCHES,
    prefetch_max_bytes: int = DEFAULT_PREFETCH_MAX_BYTES,
    start_batch: int = 0,
    yield_index: bool = False,
) -> PrefetchBatchIterator:
    """Iterate over batches with optional bounded single-thread prefetch."""
    return PrefetchBatchIterator(
        dataset,
        batches,
        prefetch_batches=prefetch_batches,
        prefetch_max_bytes=prefetch_max_bytes,
        start_batch=start_batch,
        yield_index=yield_index,
    )
