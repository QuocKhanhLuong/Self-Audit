"""Bounded CPU candidate execution and optional spawn multiprocessing (W3).

Scientific role
---------------
Candidate bank generation (:func:`~self_audit_maskfree.hypotheses.generate_bank`)
is an image-only, CPU-bound workload. It creates the four frozen candidate
partitions (grouping k-means, anatomical initializer, boundary edit, split/merge)
and resolves their anatomical names via :mod:`self_audit_maskfree.ontology`.

During batch training (e.g. batch_size=8 at image_size=224), each unit's candidate
bank can be generated independently:
- There is zero cross-unit state or gradient flow.
- Each unit receives an explicit, deterministic seed derived from the epoch and unit ID.
- Candidate generation is strictly detached from producer and auditor graphs.

Execution modes
---------------
1. Canonical serial (``candidate_workers=0``, default):
   Executes in the caller's thread with zero process management or IPC overhead.
   This remains the authoritative reference.

2. Persistent spawn pool (``candidate_workers=2`` or ``4``, opt-in):
   Uses Python's ``spawn`` multiprocessing context with:
   - Dedicated per-worker thread count (``candidate_worker_threads=1`` default).
   - Zero CUDA initialization in child processes (CPU only).
   - Bounded IPC serialization: producer features are detached, moved to CPU, and
     sliced to :data:`~self_audit_maskfree.hypotheses.MAX_FEATURE_CHANNELS` before
     inter-process transfer.
   - Guaranteed preservation of original batch order.
   - Clean exception propagation and resource teardown.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from collections.abc import Sequence
from multiprocessing.context import BaseContext
from typing import Any, Self

import torch

from .contracts import FittingView, Hypothesis
from .hypotheses import MAX_FEATURE_CHANNELS, generate_bank

#: Whitelisted worker counts matching the project configuration schema.
VALID_WORKER_COUNTS = (0, 2, 4)


class CandidateExecutionError(ValueError):
    """Raised when candidate execution configuration or input contracts are violated."""


def _worker_init(worker_threads: int) -> None:
    """Initialize a spawned candidate worker.

    Ensures CPU-only execution with strictly bounded thread concurrency and no
    CUDA context creation.
    """
    threads = max(int(worker_threads), 1)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    # Ensure workers do not attempt to use or initialize CUDA.
    if hasattr(torch, "cuda") and torch.cuda.is_initialized():  # pragma: no cover
        raise CandidateExecutionError("Worker process detected unexpected initialized CUDA context")


def _worker_task(
    payload: tuple[int, FittingView, torch.Tensor | None, int]
) -> tuple[int, list[Hypothesis]]:
    """Execute candidate bank generation for one unit inside a worker process."""
    index, fitting_view, features, seed = payload
    if not isinstance(fitting_view, FittingView):
        raise CandidateExecutionError(
            f"Expected FittingView, got {type(fitting_view).__name__} at index {index}"
        )
    if fitting_view.image.device.type != "cpu":
        raise CandidateExecutionError(
            f"Worker received non-CPU fitting view on device {fitting_view.image.device}"
        )
    if features is not None and features.device.type != "cpu":
        raise CandidateExecutionError(
            f"Worker received non-CPU feature tensor on device {features.device}"
        )
    bank = generate_bank(fitting_view, features=features, seed=seed)
    return index, bank


#: Hard byte bound per unit payload across IPC boundary (8 MiB).
MAX_PAYLOAD_BYTES_PER_UNIT = 8 * 1024 * 1024


def _prepare_task_payloads(
    fitting_views: Sequence[FittingView],
    features: Sequence[torch.Tensor | None] | None,
    seeds: Sequence[int],
) -> list[tuple[int, FittingView, torch.Tensor | None, int]]:
    """Validate inputs and construct bounded serialization payloads."""
    count = len(fitting_views)
    if count > 8:
        raise CandidateExecutionError("candidate IPC is bounded to the canonical physical batch of 8")
    if count == 0:
        return []
    if len(seeds) != count:
        raise CandidateExecutionError(
            f"Seeds count ({len(seeds)}) must match fitting_views count ({count})"
        )
    if features is not None and len(features) != count:
        raise CandidateExecutionError(
            f"Features count ({len(features)}) must match fitting_views count ({count})"
        )

    payloads: list[tuple[int, FittingView, torch.Tensor | None, int]] = []
    for i in range(count):
        view = fitting_views[i]
        if not isinstance(view, FittingView):
            raise CandidateExecutionError(
                f"Item at index {i} is not a FittingView ({type(view).__name__})"
            )
        try:
            if any(t.device.type != "cpu" or t.requires_grad for t in (view.image, view.support, view.context)):
                raise CandidateExecutionError("candidate workers require detached CPU fitting tensors")
            view.validate()
        except Exception as exc:
            raise CandidateExecutionError(f"Invalid FittingView at index {i}: {exc}") from exc

        # Calculate unit tensor footprint
        unit_bytes = (
            view.image.numel() * view.image.element_size()
            + view.support.numel() * view.support.element_size()
            + view.context.numel() * view.context.element_size()
        )

        feat = features[i] if features is not None else None
        if feat is not None:
            # Bound IPC payload: candidate generation only consumes MAX_FEATURE_CHANNELS.
            feat = feat.detach().to(device="cpu", dtype=torch.float32)
            if feat.ndim == 4 and feat.shape[0] == 1:
                feat = feat[0]
            # A contiguous slice can still retain the entire physical-batch
            # storage. Clone the consumed channels so IPC shares only this unit.
            feat = feat[:MAX_FEATURE_CHANNELS].contiguous().clone()
            unit_bytes += feat.numel() * feat.element_size()

        if unit_bytes > MAX_PAYLOAD_BYTES_PER_UNIT:
            raise CandidateExecutionError(
                f"Unit at index {i} tensor footprint ({unit_bytes} bytes) exceeds "
                f"maximum allowable payload ({MAX_PAYLOAD_BYTES_PER_UNIT} bytes)"
            )

        compact_view = replace(view, image=view.image.clone(), support=view.support.clone(), context=view.context.clone())
        payloads.append((i, compact_view, feat, int(seeds[i])))
    return payloads


class CandidatePoolExecutor:
    """Persistent execution manager for candidate bank generation.

    Parameters
    ----------
    workers:
        Number of worker processes. 0 means serial in-process execution (default).
        2 or 4 enables a persistent spawn process pool.
    worker_threads:
        Number of CPU threads allocated to each worker's PyTorch runtime (default 1).
    """

    def __init__(self, workers: int = 0, worker_threads: int = 1) -> None:
        self._pool: Any | None = None
        self._closed = True
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
            raise CandidateExecutionError(f"workers must be a non-negative integer, got {workers!r}")
        if workers not in VALID_WORKER_COUNTS:
            raise CandidateExecutionError(
                f"workers must be one of {VALID_WORKER_COUNTS}, got {workers}"
            )
        if not isinstance(worker_threads, int) or isinstance(worker_threads, bool) or worker_threads < 1:
            raise CandidateExecutionError(
                f"worker_threads must be a positive integer, got {worker_threads!r}"
            )

        self.workers = int(workers)
        self.worker_threads = int(worker_threads)
        self._ctx: BaseContext = mp.get_context("spawn")
        self._closed = False

    def _get_pool(self) -> Any:
        """Lazily create or return the persistent spawn pool."""
        if self.workers <= 0:
            return None
        if self._closed:
            raise CandidateExecutionError("CandidatePoolExecutor has been closed")
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=self._ctx,
                initializer=_worker_init,
                initargs=(self.worker_threads,),
            )
        return self._pool

    def generate_banks(
        self,
        fitting_views: Sequence[FittingView],
        features: Sequence[torch.Tensor | None] | None = None,
        seeds: Sequence[int] | None = None,
    ) -> list[list[Hypothesis]]:
        """Generate candidate banks for a batch of fitting views.

        Parameters
        ----------
        fitting_views:
            Sequence of FittingView instances, one per batch unit.
        features:
            Optional sequence of detached producer feature tensors.
        seeds:
            Explicit per-unit deterministic seeds. If None, default unit seeds are assigned.

        Returns
        -------
        list[list[Hypothesis]]
            Candidate banks strictly matching the original input order.
        """
        if self._closed:
            raise CandidateExecutionError("CandidatePoolExecutor has been closed")

        count = len(fitting_views)
        if count == 0:
            return []

        resolved_seeds: list[int]
        if seeds is None:
            resolved_seeds = [42 + i for i in range(count)]
        else:
            resolved_seeds = [int(s) for s in seeds]

        payloads = _prepare_task_payloads(fitting_views, features, resolved_seeds)

        # 1. Canonical serial execution (workers=0)
        if self.workers == 0:
            results: list[list[Hypothesis]] = []
            for _, view, feat, seed in payloads:
                results.append(generate_bank(view, features=feat, seed=seed))
            return results

        # 2. Persistent spawn pool execution (workers > 0)
        pool = self._get_pool()
        try:
            # Materialize under the exception boundary. BrokenProcessPool is
            # propagated if a worker dies; multiprocessing.Pool.map can hang.
            raw_results = list(pool.map(_worker_task, payloads, chunksize=1))
        except Exception as exc:
            self.terminate()
            raise CandidateExecutionError(f"Candidate generation worker failed: {exc}") from exc

        # Guarantee preservation of original input order
        ordered_banks: list[list[Hypothesis]] = [None] * count  # type: ignore[list-item]
        for index, bank in raw_results:
            ordered_banks[index] = bank

        for i, b in enumerate(ordered_banks):
            if b is None:  # pragma: no cover
                raise CandidateExecutionError(f"Missing result for unit index {i}")

        return ordered_banks

    def close(self) -> None:
        """Gracefully close the worker pool."""
        if not self._closed:
            self._closed = True
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
                self._pool = None

    def terminate(self) -> None:
        """Stop only this executor's children on IPC/worker failure.

        CPython before 3.14 has no public terminate_workers(). Capture its
        owned Process handles before shutdown; never search/kill by name.
        """
        pool, self._pool = self._pool, None
        self._closed = True
        if pool is None:
            return
        processes = list((getattr(pool, "_processes", None) or {}).values())
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True) and getattr(self, "_pool", None) is not None:
            with contextlib.suppress(BaseException):
                self.close()


def generate_candidate_banks(
    fitting_views: Sequence[FittingView],
    features: Sequence[torch.Tensor | None] | None = None,
    seeds: Sequence[int] | None = None,
    *,
    workers: int = 0,
    worker_threads: int = 1,
    pool: CandidatePoolExecutor | None = None,
) -> list[list[Hypothesis]]:
    """Batch candidate bank generation with optional multiprocessing.

    Parameters
    ----------
    fitting_views:
        Batch of FittingView objects.
    features:
        Optional batch of feature tensors.
    seeds:
        Explicit per-unit seeds.
    workers:
        Worker count (0, 2, 4). Default 0 (canonical serial).
    worker_threads:
        Per-worker PyTorch thread limit (default 1).
    pool:
        Optional pre-existing CandidatePoolExecutor instance. If provided,
        reuses the persistent pool across multiple calls.
    """
    if pool is not None:
        return pool.generate_banks(fitting_views, features=features, seeds=seeds)

    if workers == 0:
        # Zero-allocation direct path
        count = len(fitting_views)
        if count == 0:
            return []
        resolved_seeds = [42 + i for i in range(count)] if seeds is None else list(seeds)
        payloads = _prepare_task_payloads(fitting_views, features, resolved_seeds)
        return [generate_bank(view, features=feat, seed=seed) for _, view, feat, seed in payloads]

    with CandidatePoolExecutor(workers=workers, worker_threads=worker_threads) as temp_pool:
        return temp_pool.generate_banks(fitting_views, features=features, seeds=seeds)
