"""Shared serialization helper module for weights_only safe checkpointing and strict artifact IO.

Provides recursive CPU-normalization to weights_only-safe types, strict path-error
reporting for unsupported objects and dtypes, rejection of unsafe sets/frozensets,
rejection of bytes/bytearray, builtin primitive casting for string subclasses,
mapping key normalization and collision detection, explicit nonfinite metadata
sentinel preservation without zeroing, and durable atomic save with unswallowed fsync.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from self_audit.artifact_io import (
    ArtifactSerializationError,
    atomic_write_json,
    clean_wandb_payload,
    json_safe_artifact,
    read_json_artifact,
)


class CheckpointSerializationError(TypeError, ValueError):
    """Raised when an object or dtype cannot be safely serialized in a checkpoint."""
    pass


SAFE_TENSOR_DTYPES = frozenset(
    {
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
        torch.complex64,
        torch.complex128,
    }
)


def is_finite_tensor(tensor: torch.Tensor) -> bool:
    """Check if all elements of a tensor are finite."""
    return bool(torch.isfinite(tensor.detach()).all().item())


def _normalize_training_tensor(val: torch.Tensor, path: str) -> torch.Tensor:
    """Normalize a model/optimizer/scheduler/scaler tensor to CPU and verify safety & finiteness."""
    if not torch.is_tensor(val):
        raise CheckpointSerializationError(
            f"Unsupported object of type {type(val).__name__} at '{path}': expected Tensor"
        )
    if type(val) is not torch.Tensor and type(val) is not torch.nn.Parameter:
        raise CheckpointSerializationError(
            f"Unsupported tensor subclass {type(val).__name__} at '{path}': custom tensor subclasses cannot be serialized safely under weights_only=True"
        )
    # Reject unsupported dtypes (such as uint32 on torch <= 2.4.1)
    if val.dtype == getattr(torch, "uint32", None):
        raise CheckpointSerializationError(
            f"Unsupported tensor dtype {val.dtype} at '{path}': not supported for safe weights_only serialization"
        )
    if val.dtype not in SAFE_TENSOR_DTYPES:
        raise CheckpointSerializationError(
            f"Unsupported tensor dtype {val.dtype} at '{path}': not supported for safe weights_only serialization"
        )
    if not is_finite_tensor(val):
        raise FloatingPointError(f"Non-finite tensor in {path}")
    return val.detach().cpu().clone()


def _normalize_mapping_key(k: Any, path: str) -> str | int:
    """Validate and normalize mapping keys preserving optimizer integer keys."""
    if isinstance(k, (bool, np.bool_)):
        raise CheckpointSerializationError(
            f"Unsupported mapping key type bool ({k!r}) at '{path}': boolean keys are not weights_only-safe"
        )
    if isinstance(k, (int, np.integer)):
        return int(k)
    if isinstance(k, (str, Path, os.PathLike)):
        return str(k)
    raise CheckpointSerializationError(
        f"Unsupported mapping key type {type(k).__name__} ({k!r}) at '{path}': keys must be string or integer"
    )


def _normalize_training_tree(val: Any, path: str) -> Any:
    """Recursively normalize training objects (model, optimizer, scheduler, scaler, rng_state).

    Strictly enforces finiteness on all tensors and floats, moves tensors to CPU,
    rejects sets/frozensets, rejects bytes/bytearray, normalizes string subclasses to builtin str,
    and validates mapping keys.
    """
    if torch.is_tensor(val):
        return _normalize_training_tensor(val, path)
    if isinstance(val, (set, frozenset)):
        raise CheckpointSerializationError(
            f"Unsupported container type {type(val).__name__} at '{path}': sets and frozensets are not supported under weights_only=True"
        )
    if isinstance(val, (bytes, bytearray)):
        raise CheckpointSerializationError(
            f"Unsupported object of type {type(val).__name__} at '{path}': bytes and bytearray are not supported under weights_only=True"
        )
    if isinstance(val, (float, np.floating)):
        f_val = float(val)
        if not math.isfinite(f_val):
            raise FloatingPointError(f"Non-finite value in {path}: {val}")
        return f_val
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, str):
        return str(val)
    if val is None:
        return None
    if isinstance(val, Mapping):
        normalized_map: dict[str | int, Any] = {}
        seen_keys: set[str | int] = set()
        for k, v in val.items():
            norm_k = _normalize_mapping_key(k, path)
            if norm_k in seen_keys:
                raise CheckpointSerializationError(
                    f"Mapping key collision at '{path}': duplicate normalized key {norm_k!r}"
                )
            seen_keys.add(norm_k)
            child_path = f"{path}.{norm_k}" if isinstance(norm_k, str) else f"{path}[{norm_k!r}]"
            normalized_map[norm_k] = _normalize_training_tree(v, child_path)
        return normalized_map
    if isinstance(val, list):
        return [_normalize_training_tree(v, f"{path}[{idx}]") for idx, v in enumerate(val)]
    if isinstance(val, tuple):
        return tuple(_normalize_training_tree(v, f"{path}[{idx}]") for idx, v in enumerate(val))
    raise CheckpointSerializationError(
        f"Unsupported object of type {type(val).__name__} at '{path}': not weights_only-safe"
    )


def _normalize_metadata_tree(val: Any, path: str) -> Any:
    """Recursively normalize metadata objects (config, provenance, extra).

    - Converts Path objects to string.
    - Normalizes string and byte subclasses (such as TorchVersion) to builtin primitives.
    - Rejects sets and frozensets.
    - Preserves nonfinite floats (NaN, Inf, -Inf) and None sentinels without zeroing.
    - Normalizes NumPy types to Python primitives.
    - Recursively validates mapping keys and detects collisions.
    - Rejects unsupported custom objects and dtypes before writing.
    """
    if isinstance(val, (set, frozenset)):
        raise CheckpointSerializationError(
            f"Unsupported container type {type(val).__name__} at '{path}': sets and frozensets are not supported under weights_only=True"
        )
    if isinstance(val, (Path, os.PathLike)):
        return str(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        # Explicit nonfinite metadata sentinel policy without zeroing:
        # float('nan'), float('inf'), float('-inf') are preserved as standard Python floats.
        return float(val)
    if isinstance(val, str):
        return str(val)
    if isinstance(val, (bytes, bytearray)):
        raise CheckpointSerializationError(
            f"Unsupported object of type {type(val).__name__} at '{path}': bytes and bytearray are not supported under weights_only=True"
        )
    if val is None:
        return None
    if torch.is_tensor(val):
        if type(val) is not torch.Tensor and type(val) is not torch.nn.Parameter:
            raise CheckpointSerializationError(
                f"Unsupported tensor subclass {type(val).__name__} at '{path}': custom tensor subclasses cannot be serialized safely under weights_only=True"
            )
        if val.dtype == getattr(torch, "uint32", None):
            raise CheckpointSerializationError(
                f"Unsupported tensor dtype {val.dtype} at '{path}': not supported for safe weights_only serialization"
            )
        if val.dtype not in SAFE_TENSOR_DTYPES:
            raise CheckpointSerializationError(
                f"Unsupported tensor dtype {val.dtype} at '{path}': not supported for safe weights_only serialization"
            )
        return val.detach().cpu().clone()
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return _normalize_metadata_tree(val.item(), path)
        return _normalize_metadata_tree(val.tolist(), path)
    if isinstance(val, Mapping):
        normalized_map: dict[str | int, Any] = {}
        seen_keys: set[str | int] = set()
        for k, v in val.items():
            norm_k = _normalize_mapping_key(k, path)
            if norm_k in seen_keys:
                raise CheckpointSerializationError(
                    f"Mapping key collision at '{path}': duplicate normalized key {norm_k!r}"
                )
            seen_keys.add(norm_k)
            child_path = f"{path}.{norm_k}" if isinstance(norm_k, str) else f"{path}[{norm_k!r}]"
            normalized_map[norm_k] = _normalize_metadata_tree(v, child_path)
        return normalized_map
    if isinstance(val, list):
        return [_normalize_metadata_tree(v, f"{path}[{idx}]") for idx, v in enumerate(val)]
    if isinstance(val, tuple):
        return tuple(_normalize_metadata_tree(v, f"{path}[{idx}]") for idx, v in enumerate(val))
    raise CheckpointSerializationError(
        f"Unsupported object of type {type(val).__name__} at '{path}': cannot be serialized safely under weights_only=True"
    )


normalize_metadata_tree = _normalize_metadata_tree


def normalize_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str | int, Any]:
    """Recursively normalize and validate all payload sections to CPU weights_only-safe types.

    Enforces strict finiteness on model, optimizer, scheduler, scaler.
    Validates rng_state recursively without bypassing.
    Normalizes metadata (config, provenance, extra) to weights_only-safe types with
    explicit nonfinite sentinel policy without zeroing.
    Normalizes string subclasses (like TorchVersion) to builtin primitives.
    Rejects sets/frozensets, bytes/bytearray, custom tensor subclasses, and unsupported objects/dtypes with exact path errors.
    Recursively validates mapping keys at root level and nested levels, preserving optimizer integer keys and detecting collisions.
    """
    if not isinstance(payload, Mapping):
        raise CheckpointSerializationError(
            f"Checkpoint payload must be a mapping, got {type(payload).__name__}"
        )
    normalized: dict[str | int, Any] = {}
    seen_keys: set[str | int] = set()
    for key, value in payload.items():
        norm_key = _normalize_mapping_key(key, "root")
        if norm_key in seen_keys:
            raise CheckpointSerializationError(
                f"Mapping key collision at 'root': duplicate normalized key {norm_key!r}"
            )
        seen_keys.add(norm_key)
        path_str = str(norm_key)
        if norm_key in ("model", "optimizer", "scheduler", "scaler", "rng_state"):
            normalized[norm_key] = _normalize_training_tree(value, path_str)
        else:
            normalized[norm_key] = _normalize_metadata_tree(value, path_str)
    return normalized


def _atomic_save_torch_raw(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Internal atomic save primitive that writes an already-normalized payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)

        torch.save(payload, temporary)

        with open(temporary, "r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, path)

        dir_fd = None
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            os.fsync(dir_fd)
        except (OSError, AttributeError):
            pass
        finally:
            if dir_fd is not None:
                os.close(dir_fd)

        temporary = None
        return path
    except BaseException:
        if temporary is not None:
            try:
                if temporary.exists():
                    temporary.unlink()
            except Exception:
                pass
        raise


def atomic_save_torch(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Public safe atomic save API. Normalizes payload to weights_only safe types."""
    normalized = normalize_checkpoint_payload(payload)
    return _atomic_save_torch_raw(normalized, path)
