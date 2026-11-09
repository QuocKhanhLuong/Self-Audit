"""Fail-closed native full-volume geometry for the Candidate-A 3-D track.

The runtime contract is intentionally small and exact::

    {"shape": (D, H, W),
     "spacing_zyx": (dz, dy, dx),
     "affine_zyx": 4x4 affine in millimetres,
     "context": "full_volume"}

No crop origin, resampling transform, or hidden patch coordinate is accepted.
The existing cache stores ``[T, D, H, W]`` and its preparation geometry uses
the same native affine; this module only makes the full-volume scope explicit
at the A-3D model boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..data import validate_geometry as validate_cache_geometry


_CONTEXT = "full_volume"
_RUNTIME_KEYS = {"shape", "spacing_zyx", "affine_zyx", "context"}


def _shape(value: Sequence[int], *, name: str = "shape") -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain positive integer D,H,W")
    result = tuple(value)
    if any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in result):
        raise ValueError(f"{name} must contain positive integer D,H,W")
    return tuple(int(v) for v in result)


def _spacing(value: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("spacing_zyx must contain positive dz,dy,dx in millimetres")
    result = tuple(float(v) for v in value)
    if any(not np.isfinite(v) or v <= 0.0 for v in result):
        raise ValueError("spacing_zyx must contain positive dz,dy,dx in millimetres")
    return result


def _affine(value: Any, spacing: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("affine_zyx must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-6):
        raise ValueError("affine_zyx must be homogeneous")
    linear = matrix[:3, :3]
    lengths = np.linalg.norm(linear, axis=0)
    if not np.allclose(lengths, spacing, rtol=1e-5, atol=1e-6):
        raise ValueError("affine_zyx axis lengths disagree with spacing_zyx")
    if (lengths <= 0).any():
        raise ValueError("affine_zyx has a degenerate spatial axis")
    directions = linear / lengths
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError("sheared affine_zyx is unsupported for native orthogonal voxel-face geometry")
    return tuple(tuple(float(v) for v in row) for row in matrix)


@dataclass(frozen=True)
class VolumeGeometry3D:
    """Immutable native geometry with an explicit full-volume scope."""

    shape: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]
    affine_zyx: tuple[tuple[float, ...], ...]
    context: str = _CONTEXT

    def __post_init__(self) -> None:
        shape = _shape(self.shape)
        spacing = _spacing(self.spacing_zyx)
        affine = _affine(self.affine_zyx, spacing)
        if self.context != _CONTEXT:
            raise ValueError("A-3D geometry context must be declared full_volume")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "spacing_zyx", spacing)
        object.__setattr__(self, "affine_zyx", affine)

    @property
    def affine_array(self) -> np.ndarray:
        return np.asarray(self.affine_zyx, dtype=np.float64).copy()

    @property
    def voxel_volume_mm3(self) -> float:
        return float(abs(np.linalg.det(self.affine_array[:3, :3])))

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "spacing_zyx": self.spacing_zyx,
            # Keep the runtime geometry directly JSON-serializable.  Model
            # code can still obtain an ndarray through ``affine_array``;
            # persisted sample/update metadata must not contain NumPy scalars.
            "affine_zyx": self.affine_array.tolist(),
            "context": self.context,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VolumeGeometry3D":
        if not isinstance(value, Mapping) or set(value) != _RUNTIME_KEYS:
            raise ValueError(
                "runtime geometry must contain exactly shape, spacing_zyx, affine_zyx and context"
            )
        return cls(
            shape=_shape(value["shape"]),
            spacing_zyx=_spacing(value["spacing_zyx"]),
            affine_zyx=_affine(value["affine_zyx"], _spacing(value["spacing_zyx"])),
            context=value["context"],
        )

    @classmethod
    def from_cache_geometry(
        cls, value: Mapping[str, Any], *, expected_shape: Sequence[int] | None = None
    ) -> "VolumeGeometry3D":
        """Convert unchanged cache metadata into the explicit runtime scope."""
        cache = validate_cache_geometry(value, shape_zyx=expected_shape)
        # Cache validation preserves unknown metadata for provenance.  Do not
        # let a caller that bypasses CandidateA3DData reinterpret a cropped or
        # resampled cache as a native full volume.
        if cache.get("context") not in (None, _CONTEXT):
            raise ValueError("A-3D cache geometry must declare full_volume or omit context")
        if any(key in cache for key in ("crop", "crop_origin", "crop_origin_zyx", "resample")):
            raise ValueError("A-3D cache geometry contains crop/resample metadata")
        return cls(
            shape=tuple(cache["shape_zyx"]),
            spacing_zyx=tuple(cache["spacing_zyx_mm"]),
            affine_zyx=cache["affine_zyx"],
            context=_CONTEXT,
        )


def canonical_geometry(value: Mapping[str, Any] | VolumeGeometry3D) -> dict[str, Any]:
    """Return a validated copy of the exact runtime geometry dictionary."""
    if isinstance(value, VolumeGeometry3D):
        return value.as_dict()
    return VolumeGeometry3D.from_mapping(value).as_dict()


def validate_geometry(value: Mapping[str, Any] | VolumeGeometry3D) -> dict[str, Any]:
    """Alias used by callers that need a fail-closed geometry validator."""
    return canonical_geometry(value)


def require_full_volume_shape(geometry: VolumeGeometry3D, shape: Sequence[int]) -> None:
    """Reject cropped/patch tensors whose shape differs from native geometry."""
    if _shape(shape, name="volume shape") != geometry.shape:
        raise ValueError(
            "A-3D requires native full-volume context; crop/resample shape is unsupported"
        )


def canonical_float32_geometry(geometry: Mapping[str, Any] | VolumeGeometry3D) -> dict[str, Any]:
    """Canonicalize packet geometry exactly as its float32 header does."""
    current = VolumeGeometry3D.from_mapping(geometry) if not isinstance(geometry, VolumeGeometry3D) else geometry
    spacing = tuple(float(v) for v in np.asarray(current.spacing_zyx, dtype=np.float32))
    affine = np.asarray(current.affine_zyx, dtype=np.float32)
    return VolumeGeometry3D(
        shape=current.shape, spacing_zyx=spacing, affine_zyx=affine, context=_CONTEXT
    ).as_dict()


__all__ = [
    "VolumeGeometry3D", "canonical_geometry", "canonical_float32_geometry",
    "require_full_volume_shape", "validate_geometry",
]
