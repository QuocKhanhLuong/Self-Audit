"""Shared dependency-light artifact IO helper module.

Provides:
- Strict recursive JSON conversion (json_safe_artifact) supporting:
  * torch.Tensor.detach().cpu() (scalars, 1D/ND to lists, nonfinite floats to null)
  * NumPy scalars and arrays to Python primitives and lists
  * Path and os.PathLike to strings
  * Tuples and lists to lists
  * None to null
  * Finite primitives (bool, int, float, str)
  * Mappings with normalized string keys and collision detection
  * Nonfinite floats (NaN, Inf, -Inf) serialized to null (never zero)
  * Rejection of custom objects, custom tensor subclasses, and key collisions with useful path
  * NO default=str fallback for arbitrary objects
- Atomic same-directory temporary write (atomic_write_json) with:
  * flush + fsync + os.replace
  * best-effort directory fsync
  * non-masking cleanup on failure
  * preservation of pre-existing valid artifact on serialization/fsync/replace errors
  * unswallowed ENOSPC / OSError
- Strict JSON artifact reader (read_json_artifact) rejecting non-standard constants (NaN, Infinity)
- W&B telemetry payload converter (clean_wandb_payload) preserving arrays and nulls without silent NaN emission
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch


class ArtifactSerializationError(TypeError, ValueError):
    """Raised when an object or mapping key cannot be safely serialized to strict JSON."""
    pass


def json_safe_artifact(val: Any, path: str = "root") -> Any:
    """Recursively convert values into strictly JSON-compliant structures.

    Nonfinite floats (NaN, Infinity, -Infinity) become None (JSON null), never zero.
    Empty and undefined metrics stay distinguishable from numeric zero.
    Custom objects, custom tensor subclasses, and mapping key collisions are strictly
    rejected with exact tree paths.
    """
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        f_val = float(val)
        if not math.isfinite(f_val):
            return None
        return f_val
    if isinstance(val, str):
        return str(val)
    if isinstance(val, (Path, os.PathLike)):
        return str(val)
    if val is None:
        return None
    if torch.is_tensor(val):
        if type(val) is not torch.Tensor and type(val) is not torch.nn.Parameter:
            raise ArtifactSerializationError(
                f"Unsupported tensor subclass {type(val).__name__} at '{path}': "
                "custom tensor subclasses cannot be serialized to JSON"
            )
        t = val.detach().cpu()
        if t.ndim == 0:
            return json_safe_artifact(t.item(), path)
        return [json_safe_artifact(item, f"{path}[{idx}]") for idx, item in enumerate(t.tolist())]
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return json_safe_artifact(val.item(), path)
        return [json_safe_artifact(item, f"{path}[{idx}]") for idx, item in enumerate(val.tolist())]
    if isinstance(val, Mapping):
        normalized: dict[str, Any] = {}
        seen_keys: set[str] = set()
        for k, v in val.items():
            if isinstance(k, (bool, np.bool_)):
                raise ArtifactSerializationError(
                    f"Unsupported mapping key type bool ({k!r}) at '{path}': "
                    "boolean keys cannot be serialized to JSON"
                )
            if isinstance(k, (int, np.integer)):
                norm_k = str(int(k))
            elif isinstance(k, (str, Path, os.PathLike)):
                norm_k = str(k)
            else:
                raise ArtifactSerializationError(
                    f"Unsupported mapping key type {type(k).__name__} ({k!r}) at '{path}': "
                    "keys must be str, int, or Path"
                )
            if norm_k in seen_keys:
                raise ArtifactSerializationError(
                    f"Mapping key collision at '{path}': duplicate normalized key {norm_k!r}"
                )
            seen_keys.add(norm_k)
            subpath = f"{path}.{norm_k}" if path != "root" else norm_k
            normalized[norm_k] = json_safe_artifact(v, subpath)
        return normalized
    if isinstance(val, (list, tuple)):
        return [json_safe_artifact(item, f"{path}[{idx}]") for idx, item in enumerate(val)]

    raise ArtifactSerializationError(
        f"Unsupported type {type(val).__name__} ({val!r}) at '{path}': "
        "only standard primitives, paths, tensors, arrays, and standard collections are supported"
    )


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> Path:
    """Safely and atomically write a JSON artifact to disk.

    1. Recursively converts payload using `json_safe_artifact`.
    2. Encodes to UTF-8 JSON with `allow_nan=False`.
    3. Writes to a hidden temporary file in the same directory.
    4. Flushes and fsyncs the file handle.
    5. Fsyncs the parent directory where supported.
    6. Atomically replaces destination path with `os.replace`.
    7. Cleans up temporary file on failure without masking the underlying error.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_data = json_safe_artifact(payload)
    encoded = json.dumps(safe_data, indent=indent, sort_keys=sort_keys, allow_nan=False)

    prefix = f".{target.name}."
    suffix = ".tmp"
    tmp_fd, tmp_path_str = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=target.parent)
    tmp_path = Path(tmp_path_str)

    try:
        with open(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        # Atomic replacement into destination
        os.replace(tmp_path, target)

        # Sync parent directory to persist directory entry change
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Not all filesystems/OSes support fsync on directory descriptors
            pass

        return target
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def read_json_artifact(path: str | Path) -> dict[str, Any]:
    """Read a JSON artifact file with strict rejection of NaN and Infinity constants."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Artifact file not found: {target}")

    def _reject_constant(constant: str) -> None:
        raise ValueError(
            f"Non-standard JSON constant {constant!r} encountered in {target}; "
            "valid artifacts must use null for non-finite values"
        )

    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def clean_wandb_payload(val: Any, path: str = "root") -> Any:
    """Convert structures for W&B telemetry reusing the same strict JSON semantics.

    Non-finite floats become None (JSON null), distinguishing undefined metrics from zero.
    Mapping keys are normalized with collision detection.
    Arbitrary custom objects and custom tensor subclasses are strictly rejected.
    """
    return json_safe_artifact(val, path=path)
