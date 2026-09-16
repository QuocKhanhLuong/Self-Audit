"""Ordered CPU candidate execution with a bounded spawn-pool IPC window.

Physical neural batches may exceed the IPC window: a B32 call with chunk_size=8
submits and drains four groups of eight without changing any per-unit seed,
feature, candidate, or audit equation. The bounds below cover tensor payloads,
NOT total process RSS. Final banks are necessarily retained for all B units by
the caller; interpreter/metadata/allocator/native workspaces cost extra memory.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

import torch

from .contracts import FittingView, Hypothesis
from .hypotheses import BANK_SIZE, MAX_FEATURE_CHANNELS, generate_bank

VALID_WORKER_COUNTS = (0, 2, 4, 8)
MAX_INFLIGHT_UNITS = 8
MAX_PAYLOAD_BYTES_PER_UNIT = 8 * 1024 * 1024
# Labels, four-class probabilities and validity for the bounded candidate bank.
# This is a transport limit, not an approximation, truncation, or RSS guarantee.
MAX_RESULT_TENSOR_BYTES_PER_UNIT = 32 * 1024 * 1024


class CandidateExecutionError(ValueError):
    """Invalid execution configuration, input, result, or worker failure."""


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CandidateExecutionError(f"{name} must be a positive integer, got {value!r}")
    if maximum is not None and value > maximum:
        raise CandidateExecutionError(f"{name} must be <= {maximum}, got {value}")
    return value


def _worker_init(worker_threads: int) -> None:
    threads = _positive_int(worker_threads, "worker_threads")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_initialized():
        raise CandidateExecutionError("Worker has an unexpected initialized CUDA context")


def _tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _validate_bank(bank: list[Hypothesis]) -> int:
    if not isinstance(bank, list) or len(bank) != BANK_SIZE:
        raise CandidateExecutionError(f"Worker must return exactly {BANK_SIZE} candidates")
    total = 0
    for candidate in bank:
        if not isinstance(candidate, Hypothesis):
            raise CandidateExecutionError("Worker returned a non-Hypothesis candidate")
        tensors = (candidate.labels, candidate.probabilities, candidate.validity)
        if any(t.device.type != "cpu" or t.requires_grad for t in tensors):
            raise CandidateExecutionError("Worker results must be detached CPU tensors")
        total += _tensor_bytes(tensors)
    if total > MAX_RESULT_TENSOR_BYTES_PER_UNIT:
        raise CandidateExecutionError("Worker result exceeds the per-unit tensor transport bound")
    return total


def _worker_task(payload):
    index, view, features, seed = payload
    if not isinstance(view, FittingView):
        raise CandidateExecutionError(f"Expected FittingView at index {index}")
    if any(t.device.type != "cpu" or t.requires_grad for t in (view.image, view.support, view.context)):
        raise CandidateExecutionError("Worker received non-CPU or attached fitting tensors")
    if features is not None and (features.device.type != "cpu" or features.requires_grad):
        raise CandidateExecutionError("Worker received non-CPU or attached features")
    bank = generate_bank(view, features=features, seed=seed)
    _validate_bank(bank)
    return index, bank


def _prepare_task_payloads(fitting_views, features, seeds):
    """Prepare ONE window, never the entire large physical batch."""
    count = len(fitting_views)
    if count > MAX_INFLIGHT_UNITS:
        raise CandidateExecutionError(f"candidate IPC window exceeds {MAX_INFLIGHT_UNITS} units")
    if len(seeds) != count:
        raise CandidateExecutionError(f"Seeds count ({len(seeds)}) must match fitting_views count ({count})")
    if features is not None and len(features) != count:
        raise CandidateExecutionError(f"Features count ({len(features)}) must match fitting_views count ({count})")
    payloads = []
    for i, view in enumerate(fitting_views):
        if not isinstance(view, FittingView):
            raise CandidateExecutionError(f"Item at index {i} is not a FittingView ({type(view).__name__})")
        try:
            if any(t.device.type != "cpu" or t.requires_grad for t in (view.image, view.support, view.context)):
                raise CandidateExecutionError("candidate workers require detached CPU fitting tensors")
            view.validate()
        except Exception as exc:
            raise CandidateExecutionError(f"Invalid FittingView at index {i}: {exc}") from exc
        unit_bytes = _tensor_bytes((view.image, view.support, view.context))
        feat = features[i] if features is not None else None
        if feat is not None:
            feat = feat.detach().to(device="cpu", dtype=torch.float32)
            if feat.ndim == 4 and feat.shape[0] == 1:
                feat = feat[0]
            # Clone: a view can otherwise retain the full physical-batch storage.
            feat = feat[:MAX_FEATURE_CHANNELS].contiguous().clone()
            unit_bytes += _tensor_bytes((feat,))
        if unit_bytes > MAX_PAYLOAD_BYTES_PER_UNIT:
            raise CandidateExecutionError(
                f"Unit at index {i} tensor footprint ({unit_bytes} bytes) exceeds maximum allowable payload "
                f"({MAX_PAYLOAD_BYTES_PER_UNIT} bytes)"
            )
        compact = replace(view, image=view.image.clone(), support=view.support.clone(), context=view.context.clone())
        payloads.append((i, compact, feat, int(seeds[i])))
    return payloads


class CandidatePoolExecutor:
    """Persistent CPU-only spawn executor; at most chunk_size tasks in flight.

    Workers may be 0/2/4/8. chunk_size is 1..8 regardless of physical batch.
    Caller-supplied seeds are sliced, never reset at chunk boundaries.
    This object is owned by one trainer and is not concurrently reentrant.
    """

    def __init__(self, workers: int = 0, worker_threads: int = 1, chunk_size: int = 8) -> None:
        self._pool = None
        self._closed = True
        if isinstance(workers, bool) or not isinstance(workers, int) or workers not in VALID_WORKER_COUNTS:
            raise CandidateExecutionError(f"workers must be one of {VALID_WORKER_COUNTS}, got {workers!r}")
        self.workers = workers
        self.worker_threads = _positive_int(worker_threads, "worker_threads")
        self.chunk_size = _positive_int(chunk_size, "chunk_size", maximum=MAX_INFLIGHT_UNITS)
        self._ctx = mp.get_context("spawn")
        self._closed = False
        self._stats = {"calls": 0, "completed_calls": 0, "chunks": 0, "units": 0,
                       "max_inflight_units": 0, "max_input_tensor_bytes": 0,
                       "max_result_window_tensor_bytes": 0, "max_retained_result_tensor_bytes": 0}

    def stats(self) -> dict[str, Any]:
        return {**self._stats, "workers": self.workers, "chunk_size": self.chunk_size,
                "input_window_tensor_limit_bytes": self.chunk_size * MAX_PAYLOAD_BYTES_PER_UNIT,
                "result_window_tensor_limit_bytes": self.chunk_size * MAX_RESULT_TENSOR_BYTES_PER_UNIT,
                "scope": "tensor payloads only; final B banks, Python metadata and process RSS are additional"}

    def _get_pool(self):
        if self._closed:
            raise CandidateExecutionError("CandidatePoolExecutor has been closed")
        if self.workers == 0:
            return None
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=self._ctx,
                                             initializer=_worker_init, initargs=(self.worker_threads,))
        return self._pool

    def generate_banks(self, fitting_views, features=None, seeds=None):
        if self._closed:
            raise CandidateExecutionError("CandidatePoolExecutor has been closed")
        count = len(fitting_views)
        resolved = [42 + i for i in range(count)] if seeds is None else [int(s) for s in seeds]
        if len(resolved) != count:
            raise CandidateExecutionError(f"Seeds count ({len(resolved)}) must match fitting_views count ({count})")
        if features is not None and len(features) != count:
            raise CandidateExecutionError(f"Features count ({len(features)}) must match fitting_views count ({count})")
        self._stats["calls"] += 1
        results = []
        retained_bytes = 0
        try:
            for start in range(0, count, self.chunk_size):
                end = min(count, start + self.chunk_size)
                payloads = _prepare_task_payloads(fitting_views[start:end],
                    None if features is None else features[start:end], resolved[start:end])
                input_bytes = sum(_tensor_bytes((v.image, v.support, v.context)) +
                                  (0 if f is None else _tensor_bytes((f,))) for _, v, f, _ in payloads)
                self._stats["max_inflight_units"] = max(self._stats["max_inflight_units"], len(payloads))
                self._stats["max_input_tensor_bytes"] = max(self._stats["max_input_tensor_bytes"], input_bytes)
                # map sees ONLY the bounded window. chunksize=1 is not itself a
                # queue bound on Python 3.10-3.13; never pass the full batch here.
                iterator = (map(_worker_task, payloads) if self.workers == 0 else
                            self._get_pool().map(_worker_task, payloads, chunksize=1))
                ordered = [None] * len(payloads)
                window_bytes = 0
                for index, bank in iterator:
                    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(ordered):
                        raise CandidateExecutionError(f"Invalid worker result index: {index!r}")
                    if ordered[index] is not None:
                        raise CandidateExecutionError(f"Duplicate worker result index: {index}")
                    window_bytes += _validate_bank(bank)
                    ordered[index] = bank
                if any(bank is None for bank in ordered):
                    raise CandidateExecutionError("Missing worker result in candidate window")
                self._stats["chunks"] += 1
                self._stats["units"] += len(ordered)
                self._stats["max_result_window_tensor_bytes"] = max(self._stats["max_result_window_tensor_bytes"], window_bytes)
                retained_bytes += window_bytes
                self._stats["max_retained_result_tensor_bytes"] = max(self._stats["max_retained_result_tensor_bytes"], retained_bytes)
                results.extend(ordered)
                del payloads, ordered, iterator
        except BaseException as exc:
            self.terminate()
            if not isinstance(exc, Exception) or isinstance(exc, CandidateExecutionError):
                raise
            raise CandidateExecutionError(f"Candidate generation worker failed: {exc}") from exc
        self._stats["completed_calls"] += 1
        return results

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
                self._pool = None

    def terminate(self) -> None:
        """Stop only owned children; compatible with Python before 3.14."""
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() if exc_type is None else self.terminate()

    def __del__(self):
        if not getattr(self, "_closed", True) and getattr(self, "_pool", None) is not None:
            with contextlib.suppress(BaseException):
                self.close()


def generate_candidate_banks(fitting_views, features=None, seeds=None, *, workers=0,
                             worker_threads=1, chunk_size=8, pool=None):
    """Use a caller-owned executor, or a temporary bounded executor."""
    if pool is not None:
        return pool.generate_banks(fitting_views, features=features, seeds=seeds)
    with CandidatePoolExecutor(workers=workers, worker_threads=worker_threads, chunk_size=chunk_size) as executor:
        return executor.generate_banks(fitting_views, features=features, seeds=seeds)
