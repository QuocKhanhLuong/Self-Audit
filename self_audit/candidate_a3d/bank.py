"""Finite native-volume measurement bank for the Candidate-A 3-D track.

The bank is encoder-independent.  Query locations are generated once from a
configuration and mapped to native ``(D, H, W)`` voxel geometry at runtime;
they are never inferred from a label.  REGION queries expose a fixed cuboid
footprint and BOUNDARY queries expose one nearest native voxel.  The latter is
the declared sampling rule used by the student value head, teacher target and
decoder tests alike.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from scipy import ndimage


MeasurementType = Literal["REGION", "BOUNDARY"]
_MEASUREMENT_TYPES = ("REGION", "BOUNDARY")
_IGNORE_INDEX = 255


def _grid(value: Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain positive integers (nz, ny, nx)")
    result = tuple(value)
    if any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in result):
        raise ValueError(f"{name} must contain positive integers (nz, ny, nx)")
    return tuple(int(v) for v in result)


def _shape(value: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("shape must contain positive integer D,H,W")
    result = tuple(value)
    if any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in result):
        raise ValueError("shape must contain positive integer D,H,W")
    return tuple(int(v) for v in result)


def _spacing(value: Sequence[float], *, batch: int | None = None, device=None) -> torch.Tensor:
    out = torch.as_tensor(value, dtype=torch.float32, device=device)
    if batch is not None:
        if out.ndim == 1:
            if out.numel() != 3:
                raise ValueError("spacing_zyx must have shape [B,3] or [3]")
            out = out[None].expand(batch, -1)
        if out.shape != (batch, 3):
            raise ValueError("spacing_zyx must have shape [B,3] or [3]")
    elif out.ndim != 1 or out.numel() != 3:
        raise ValueError("spacing_zyx must contain (dz,dy,dx)")
    if not torch.isfinite(out).all() or bool((out <= 0).any()):
        raise ValueError("spacing_zyx must be finite and positive")
    return out


@dataclass(frozen=True)
class BankConfig3D:
    """Reproducible finite 3-D query bank.

    Grid coordinates are ordered ``(z, y, x)`` and normalized to voxel
    centres.  ``region_scales`` are isotropic fractions of each volume axis;
    a value of one means the full native volume.
    """

    classes: tuple[int, ...] = (1, 2, 3)
    region_grid: tuple[int, int, int] = (2, 2, 2)
    boundary_grid: tuple[int, int, int] = (2, 2, 2)
    region_scales: tuple[float, ...] = (0.5, 1.0)
    radius_mm: float = 5.0
    types: tuple[str, ...] = ("REGION", "BOUNDARY")

    def __post_init__(self) -> None:
        classes = tuple(int(c) for c in self.classes)
        if not classes or len(set(classes)) != len(classes) or any(c <= 0 for c in classes):
            raise ValueError("classes must be unique positive foreground class IDs")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "region_grid", _grid(self.region_grid, "region_grid"))
        object.__setattr__(self, "boundary_grid", _grid(self.boundary_grid, "boundary_grid"))
        scales = tuple(float(s) for s in self.region_scales)
        if not scales or any(not math.isfinite(s) or s <= 0.0 or s > 1.0 for s in scales):
            raise ValueError("region_scales must be finite values in (0,1]")
        object.__setattr__(self, "region_scales", scales)
        radius = float(self.radius_mm)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius_mm must be finite and positive")
        object.__setattr__(self, "radius_mm", radius)
        types = tuple(str(t).upper() for t in self.types)
        if not types or len(set(types)) != len(types) or any(t not in _MEASUREMENT_TYPES for t in types):
            raise ValueError("types must be a unique non-empty subset of REGION and BOUNDARY")
        object.__setattr__(self, "types", types)

    def canonical(self) -> dict:
        return {
            "classes": list(self.classes),
            "region_grid": list(self.region_grid),
            "boundary_grid": list(self.boundary_grid),
            "region_scales": list(self.region_scales),
            "radius_mm": self.radius_mm,
            "types": list(self.types),
        }


@dataclass(frozen=True)
class Query3D:
    index: int
    kind: MeasurementType
    class_id: int
    z: float
    y: float
    x: float
    scale: float
    query_id: str

    # The aliases make the semantic coordinates explicit without making the
    # model depend on a particular field spelling.
    @property
    def location_zyx(self) -> tuple[float, float, float]:
        return self.z, self.y, self.x

    def canonical(self) -> dict:
        return {
            "index": self.index,
            "type": self.kind,
            "class": self.class_id,
            "z": self.z,
            "y": self.y,
            "x": self.x,
            "scale": self.scale,
            "id": self.query_id,
        }


def _centres(grid: tuple[int, int, int]) -> Iterable[tuple[float, float, float]]:
    nz, ny, nx = grid
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                yield ((iz + 0.5) / nz, (iy + 0.5) / ny, (ix + 0.5) / nx)


def _query_id(kind: str, class_id: int, z: float, y: float, x: float, scale: float) -> str:
    return f"{kind}:c{class_id}:z{z:.8f}:y{y:.8f}:x{x:.8f}:s{scale:.8f}"


class QueryBank3D:
    """Deterministic finite bank of typed 3-D measurements."""

    def __init__(self, config: BankConfig3D | None = None):
        self.config = config or BankConfig3D()
        queries: list[Query3D] = []
        for kind in self.config.types:
            centres = tuple(_centres(self.config.region_grid if kind == "REGION" else self.config.boundary_grid))
            scales = self.config.region_scales if kind == "REGION" else (1.0,)
            for class_id in self.config.classes:
                for z, y, x in centres:
                    for scale in scales:
                        index = len(queries)
                        queries.append(Query3D(
                            index=index, kind=kind, class_id=class_id,
                            z=z, y=y, x=x, scale=scale,
                            query_id=_query_id(kind, class_id, z, y, x, scale),
                        ))
        self.queries = tuple(queries)
        self._by_id = {q.query_id: q for q in self.queries}
        if len(self._by_id) != len(self.queries):
            raise ValueError("3-D bank query IDs are not unique")
        canonical = {"config": self.config.canonical(), "queries": [q.canonical() for q in self.queries]}
        self.fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def query(self, value: int | str | Query3D) -> Query3D:
        if isinstance(value, Query3D):
            if value.index < 0 or value.index >= len(self) or self.queries[value.index] != value:
                raise ValueError("query does not belong to this bank")
            return value
        if isinstance(value, str):
            try:
                return self._by_id[value]
            except KeyError as exc:
                raise ValueError(f"unknown query ID: {value}") from exc
        if isinstance(value, bool) or int(value) != value or not 0 <= int(value) < len(self):
            raise ValueError("query index out of range")
        return self.queries[int(value)]

    def descriptors(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """Return ``[Q, 2 + n_classes + 4]`` semantic descriptors."""
        device = torch.device(device) if device is not None else torch.device("cpu")
        class_index = {c: i for i, c in enumerate(self.config.classes)}
        type_index = {kind: i for i, kind in enumerate(_MEASUREMENT_TYPES)}
        out = torch.zeros((len(self), 2 + len(self.config.classes) + 4), dtype=dtype, device=device)
        for q in self.queries:
            out[q.index, type_index[q.kind]] = 1
            out[q.index, 2 + class_index[q.class_id]] = 1
            out[q.index, 2 + len(self.config.classes):] = torch.tensor(
                (q.z, q.y, q.x, q.scale), dtype=dtype, device=device)
        return out

    @staticmethod
    def _nearest(q: Query3D, shape: Sequence[int]) -> tuple[int, int, int]:
        d, h, w = _shape(shape)
        return (
            min(d - 1, max(0, int(math.floor(q.z * (d - 1) + 0.5)))),
            min(h - 1, max(0, int(math.floor(q.y * (h - 1) + 0.5)))),
            min(w - 1, max(0, int(math.floor(q.x * (w - 1) + 0.5)))),
        )

    def point_index(self, query: int | str | Query3D, shape: Sequence[int]) -> tuple[int, int, int]:
        q = self.query(query)
        if q.kind != "BOUNDARY":
            raise ValueError("point_index is defined only for BOUNDARY queries")
        return self._nearest(q, shape)

    def region_slices(self, query: int | str | Query3D, shape: Sequence[int]) -> tuple[slice, slice, slice]:
        q = self.query(query)
        if q.kind != "REGION":
            raise ValueError("region_slices is defined only for REGION queries")
        d, h, w = _shape(shape)
        if q.scale >= 1.0:
            return slice(0, d), slice(0, h), slice(0, w)
        starts, stops = [], []
        for centre, size in zip((q.z, q.y, q.x), (d, h, w)):
            lo = int(math.floor((centre - q.scale / 2.0) * size))
            hi = int(math.ceil((centre + q.scale / 2.0) * size))
            lo, hi = max(0, lo), min(size, hi)
            if hi <= lo:
                idx = min(size - 1, max(0, int(math.floor(centre * (size - 1) + 0.5))))
                lo, hi = idx, idx + 1
            starts.append(lo); stops.append(hi)
        return tuple(slice(lo, hi) for lo, hi in zip(starts, stops))

    def _validate_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        if labels.ndim != 4:
            raise ValueError("3-D labels must have shape [B,D,H,W]")
        if labels.dtype.is_floating_point or labels.dtype.is_complex:
            raise ValueError("3-D labels must be integer tensors")
        valid = {0, *self.config.classes}
        observed = set(int(v) for v in torch.unique(labels.detach().cpu()).tolist())
        if _IGNORE_INDEX in observed or not observed.issubset(valid):
            raise ValueError("labels contain invalid/ignore class IDs")
        return labels

    @staticmethod
    def _boundary_faces(mask: np.ndarray, spacing: tuple[float, float, float]) -> list[tuple[int, float, float, float, float, float]]:
        """Build axis-aligned class-interface faces in physical coordinates.

        A face is ``(axis, coordinate, lo0, hi0, lo1, hi1)``; the two in-plane
        ranges are in the remaining axis order.  This makes point distances
        exact for voxel faces and preserves anisotropic spacing.  Only
        interfaces between two voxels inside the declared full volume are
        anatomical interfaces: the outer FOV is not treated as background.
        This is the A-3D continuous voxel-face convention, not a voxel-centre
        EDT convention.
        """
        d, h, w = mask.shape
        shape = (d, h, w)
        faces: list[tuple[int, float, float, float, float, float]] = []
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            length = shape[axis]
            # Faces at 0 and ``length`` are FOV boundaries, not measured
            # anatomy.  Vectorising the internal transition search avoids a
            # Python visit to every voxel for each query on a clinical volume.
            pad_width = [(0, 0)] * 3
            pad_width[axis] = (1, 1)
            padded = np.pad(mask, pad_width, mode="constant", constant_values=False)
            left = np.take(padded, np.arange(1, length), axis=axis)
            right = np.take(padded, np.arange(2, length + 1), axis=axis)
            transitions = np.argwhere(left != right)
            for coordinates in transitions:
                face = int(coordinates[axis]) + 1
                ranges = []
                for item in other:
                    coordinate = int(coordinates[item])
                    ranges.extend((coordinate * spacing[item], (coordinate + 1) * spacing[item]))
                faces.append((axis, face * spacing[axis], ranges[0], ranges[1], ranges[2], ranges[3]))
        return faces

    @staticmethod
    def _signed_distance(mask: torch.Tensor, point: tuple[int, int, int], spacing: tuple[float, float, float], radius: float) -> float:
        mask = mask.detach().bool()
        if mask.ndim != 3:
            raise ValueError("boundary mask must have shape [D,H,W]")
        if not bool(mask.any()):
            return 1.0
        # A class occupying the entire full volume has no interface.  The
        # signed-distance sentinel is explicit and stable rather than an
        # ``inf`` that would later be clipped by accident.
        if bool(mask.all()):
            return -1.0
        distance = QueryBank3D._distance_field(mask, spacing)
        best = float(distance[point])
        signed = -best if bool(mask[point]) else best
        return float(max(-1.0, min(1.0, signed / radius)))

    @staticmethod
    def _face_nodes(mask: np.ndarray) -> np.ndarray:
        """Mark doubled-grid nodes belonging to internal voxel-face rectangles.

        A voxel centre is at an odd doubled-grid coordinate and an internal
        face is at an even coordinate on its normal axis.  Marking the 3x3
        in-plane nodes for every face is sufficient: the closest point from an
        odd voxel centre to a rectangular face is always one of those nodes.
        This gives exact continuous voxel-face distances at voxel centres while
        using O(volume) memory instead of an O(voxels * faces) matrix.
        """
        if mask.ndim != 3:
            raise ValueError("boundary mask must have shape [D,H,W]")
        shape = tuple(int(value) for value in mask.shape)
        nodes = np.zeros(tuple(2 * value + 1 for value in shape), dtype=bool)
        for axis, length in enumerate(shape):
            if length < 2:
                continue
            pad_width = [(0, 0)] * 3
            pad_width[axis] = (1, 1)
            padded = np.pad(mask, pad_width, mode="constant", constant_values=False)
            left = np.take(padded, np.arange(1, length), axis=axis)
            right = np.take(padded, np.arange(2, length + 1), axis=axis)
            transitions = np.argwhere(left != right)
            if not len(transitions):
                continue
            normal = 2 * (transitions[:, axis] + 1)
            other = [index for index in range(3) if index != axis]
            for offset0 in range(3):
                for offset1 in range(3):
                    indices = [None, None, None]
                    indices[axis] = normal
                    indices[other[0]] = 2 * transitions[:, other[0]] + offset0
                    indices[other[1]] = 2 * transitions[:, other[1]] + offset1
                    nodes[tuple(indices)] = True
        return nodes

    @classmethod
    def _distance_field(cls, mask: torch.Tensor, spacing: tuple[float, float, float]) -> torch.Tensor:
        """Return physical distance to internal voxel-face rectangles.

        ``scipy.ndimage.distance_transform_edt`` is applied to the doubled
        grid with half-spacing.  Sampling odd indices returns distances at
        native voxel centres, exactly matching the continuous-face formula.
        """
        mask_cpu = mask.detach().bool().cpu().numpy()
        nodes = cls._face_nodes(mask_cpu)
        if not bool(nodes.any()):
            raise ValueError("distance field requires at least one internal interface")
        half_spacing = tuple(float(value) / 2.0 for value in spacing)
        distances = ndimage.distance_transform_edt(~nodes, sampling=half_spacing)
        sampled = distances[1::2, 1::2, 1::2]
        return torch.as_tensor(sampled, dtype=torch.float32, device=mask.device)

    def _sdf_single(self, class_mask: torch.Tensor, spacing: tuple[float, float, float]) -> torch.Tensor:
        """Build one finite normalized signed-distance volume.

        The target uses the same continuous voxel-face convention as the
        boundary queries.  An absent class maps to ``+1``; a class occupying
        the complete native volume has no internal interface and maps to the
        explicit ``-1`` sentinel.  No infinity is ever clipped implicitly.
        """

        mask = class_mask.detach().bool()
        if mask.ndim != 3:
            raise ValueError("class_mask must have shape [D,H,W]")
        if not bool(mask.any()):
            return torch.ones(mask.shape, dtype=torch.float32, device=mask.device)
        if bool(mask.all()):
            return -torch.ones(mask.shape, dtype=torch.float32, device=mask.device)
        distance = self._distance_field(mask, spacing)
        signed = torch.where(mask, -distance, distance)
        result = (signed / float(self.config.radius_mm)).clamp(-1.0, 1.0).to(torch.float32)
        if not torch.isfinite(result).all():
            raise RuntimeError("non-finite signed-distance target")
        return result

    @staticmethod
    def boundary_status(mask: torch.Tensor | np.ndarray) -> str:
        """Describe the finite-volume boundary case used by BOUNDARY targets.

        ``absent`` maps to ``+1``; ``full_occupied_no_boundary`` maps to
        ``-1``.  Otherwise the class has at least one internal voxel-face
        interface and uses the physical continuous-face distance.
        """
        array = mask.detach().cpu().bool().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask, dtype=bool)
        if array.ndim != 3:
            raise ValueError("boundary mask must have shape [D,H,W]")
        if not array.any():
            return "absent"
        if array.all():
            return "full_occupied_no_boundary"
        return "interface"

    def sdf_targets(
        self,
        labels: torch.Tensor,
        spacing_zyx: Sequence[float] | torch.Tensor | Mapping[str, Any],
    ) -> torch.Tensor:
        """Return ``[B,C,D,H,W]`` normalized physical signed distances.

        Channel zero is background and the remaining channels cover every
        configured foreground class.  The implementation is intentionally
        independent of the query bank locations so the teacher's dense SDF
        target and boundary-query target share one geometry convention.
        """

        labels = self._validate_labels(labels)
        if isinstance(spacing_zyx, Mapping):
            if tuple(spacing_zyx.get("shape", ())) != tuple(labels.shape[-3:]):
                raise ValueError("3-D SDF geometry shape does not match labels")
            if spacing_zyx.get("context") != "full_volume" or "spacing_zyx" not in spacing_zyx:
                raise ValueError("3-D SDF geometry must declare full_volume spacing_zyx")
            spacing_zyx = spacing_zyx["spacing_zyx"]
        spacing = _spacing(spacing_zyx, batch=labels.shape[0], device=labels.device)
        channels = max(self.config.classes) + 1
        output = torch.empty(
            (labels.shape[0], channels, *labels.shape[-3:]),
            dtype=torch.float32, device=labels.device,
        )
        for batch_index in range(labels.shape[0]):
            sample_spacing = tuple(float(value) for value in spacing[batch_index].detach().cpu())
            for class_id in range(channels):
                output[batch_index, class_id] = self._sdf_single(
                    labels[batch_index] == class_id, sample_spacing,
                )
        if not torch.isfinite(output).all():
            raise RuntimeError("non-finite signed-distance target")
        return output

    def targets(self, labels: torch.Tensor, spacing_zyx: Sequence[float] | torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
        """Return ``[B,Q]`` REGION fractions and clipped BOUNDARY SDF values."""
        labels = self._validate_labels(labels)
        if isinstance(spacing_zyx, Mapping):
            if tuple(spacing_zyx.get("shape", ())) != tuple(labels.shape[-3:]):
                raise ValueError("3-D target geometry shape does not match labels")
            if spacing_zyx.get("context") != "full_volume" or "spacing_zyx" not in spacing_zyx:
                raise ValueError("3-D target geometry must declare full_volume spacing_zyx")
            spacing_zyx = spacing_zyx["spacing_zyx"]
        spacing = _spacing(spacing_zyx, batch=labels.shape[0], device=labels.device)
        out = torch.empty((labels.shape[0], len(self)), dtype=torch.float32, device=labels.device)
        for b in range(labels.shape[0]):
            sample = labels[b]
            sample_spacing = tuple(float(v) for v in spacing[b].detach().cpu())
            # Build each class field once and sample boundary queries from the
            # same dense target used for teacher SDF supervision.  This avoids
            # rebuilding interface faces for every query and keeps the two
            # boundary protocols exactly aligned.
            sdf = self.sdf_targets(sample.unsqueeze(0), sample_spacing)[0]
            for q in self.queries:
                if q.kind == "REGION":
                    region = sample[self.region_slices(q, sample.shape)]
                    out[b, q.index] = (region == q.class_id).float().mean()
                else:
                    out[b, q.index] = sdf[q.class_id][self.point_index(q, sample.shape)]
        return out

    def teacher_values(self, region_probabilities: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        """Read branch outputs at the same fixed bank geometry."""
        if region_probabilities.ndim != 5 or sdf.shape != region_probabilities.shape:
            raise ValueError("teacher probabilities and sdf must be [B,C,D,H,W] with matching shapes")
        if region_probabilities.shape[1] <= max(self.config.classes):
            raise ValueError("teacher outputs do not contain all bank classes")
        values = torch.empty((region_probabilities.shape[0], len(self)), dtype=region_probabilities.dtype,
                             device=region_probabilities.device)
        shape = region_probabilities.shape[-3:]
        for q in self.queries:
            if q.kind == "REGION":
                region = region_probabilities[:, q.class_id][(...,) + self.region_slices(q, shape)]
                values[:, q.index] = region.mean(dim=(-3, -2, -1))
            else:
                z, y, x = self.point_index(q, shape)
                values[:, q.index] = sdf[:, q.class_id, z, y, x]
        return values


__all__ = ["BankConfig3D", "Query3D", "QueryBank3D", "MeasurementType"]
