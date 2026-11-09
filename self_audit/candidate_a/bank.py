"""Finite, geometry-aware anatomical measurement bank for candidate A.

The bank deliberately contains no learned state and no reference to an encoder.
It is a deterministic 2-D in-plane protocol: each query is identified by its
measurement type, foreground class, normalized location, and normalized scale.
The normalized coordinates are mapped to the centres of the native pixels of an
image.  In particular, this module does not infer query locations from a mask.

``REGION`` measurements are foreground proportions in a fixed rectangular
fractional field of view.  ``BOUNDARY`` measurements are clipped signed
distances (negative inside the requested class and positive outside) at a fixed
pixel-centre location.  Distances are measured to the interface between
voxel-cell faces, using the supplied in-plane ``(dy, dx)`` spacing, rather than
to a collection of voxel centres.

The implementation intentionally validates labels and geometry fail-closed.
The active repository's native data loader is responsible for supplying a
complete, in-plane labelled centre slice; ignore values are not silently
converted to background.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Literal, Sequence

import torch


MeasurementType = Literal["REGION", "BOUNDARY"]
_MEASUREMENT_TYPES = ("REGION", "BOUNDARY")
_IGNORE_INDEX = 255


def _grid_pair(value: Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if len(value) != 2 or any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in value):
        raise ValueError(f"{name} must be a pair of positive integers")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class BankConfig:
    """Reproducible finite-bank configuration.

    ``region_grid`` and ``boundary_grid`` are ``(ny, nx)`` grids.  Their query
    centres are ``((ix + .5) / nx, (iy + .5) / ny)`` where the first normalized
    coordinate is x and the second is y.  ``region_scales`` are isotropic
    fractions of the full image; a scale of one is explicitly the full image
    (independent of grid-centre rounding).  ``types`` controls the stable outer
    query ordering and can be used for REGION-only or BOUNDARY-only controls.

    All geometry is in-plane.  No z spacing or implicit 3-D distance is used.
    """

    classes: tuple[int, ...] = (1, 2, 3)
    region_grid: tuple[int, int] = (2, 2)
    boundary_grid: tuple[int, int] = (2, 2)
    region_scales: tuple[float, ...] = (0.5, 1.0)
    radius_mm: float = 5.0
    types: tuple[str, ...] = ("REGION", "BOUNDARY")

    def __post_init__(self) -> None:
        classes = tuple(int(c) for c in self.classes)
        if not classes or len(set(classes)) != len(classes) or any(c <= 0 for c in classes):
            raise ValueError("classes must be unique positive foreground class IDs")
        object.__setattr__(self, "classes", classes)

        object.__setattr__(self, "region_grid", _grid_pair(self.region_grid, "region_grid"))
        object.__setattr__(self, "boundary_grid", _grid_pair(self.boundary_grid, "boundary_grid"))

        scales = tuple(float(s) for s in self.region_scales)
        if not scales or any(not math.isfinite(s) or s <= 0.0 or s > 1.0 for s in scales):
            raise ValueError("region_scales must be finite values in (0, 1]")
        object.__setattr__(self, "region_scales", scales)

        radius = float(self.radius_mm)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_mm must be finite and positive")
        object.__setattr__(self, "radius_mm", radius)

        types = tuple(str(t).upper() for t in self.types)
        if not types or len(set(types)) != len(types) or any(t not in _MEASUREMENT_TYPES for t in types):
            raise ValueError("types must be a non-empty unique subset of REGION and BOUNDARY")
        object.__setattr__(self, "types", types)

    @property
    def num_channels(self) -> int:
        """Number of label/SDF channels including background channel zero."""

        return max(self.classes) + 1

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
class Query:
    """One encoder-independent bank query."""

    index: int
    kind: MeasurementType
    class_id: int
    u: float
    v: float
    scale: float
    query_id: str

    def canonical(self) -> dict:
        return {
            "index": self.index,
            "type": self.kind,
            "class": self.class_id,
            "u": self.u,
            "v": self.v,
            "scale": self.scale,
            "id": self.query_id,
        }


def _format_query_id(kind: str, class_id: int, u: float, v: float, scale: float) -> str:
    # Decimal formatting is part of the protocol, not presentation.  It makes
    # IDs stable across Python versions and independent of binary float repr.
    return f"{kind}:c{class_id}:u{u:.8f}:v{v:.8f}:s{scale:.8f}"


def _validate_spacing(spacing: torch.Tensor | Sequence[float], batch: int, *, device: torch.device) -> torch.Tensor:
    out = torch.as_tensor(spacing, device=device, dtype=torch.float32)
    if out.ndim == 1:
        if out.numel() != 2:
            raise ValueError("spacing must have shape [B, 2] or [2]")
        out = out[None].expand(batch, -1)
    if out.ndim != 2 or out.shape != (batch, 2) or not torch.isfinite(out).all() or (out <= 0).any():
        raise ValueError("spacing must be finite, positive in-plane (dy, dx) values")
    return out


def _validate_labels(labels: torch.Tensor, classes: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B, H, W] for the 2-D protocol")
    if labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise ValueError("labels must be an integer tensor; fractional/NaN labels are invalid")
    valid = {0, *classes}
    observed = set(int(v) for v in torch.unique(labels.detach().cpu()).tolist())
    if _IGNORE_INDEX in observed or not observed.issubset(valid):
        bad = sorted(observed - valid)
        raise ValueError(f"labels contain invalid/ignore values; observed invalid IDs: {bad}")
    return labels


def _grid_centres(grid: tuple[int, int]) -> Iterable[tuple[float, float]]:
    ny, nx = grid
    for iy in range(ny):
        for ix in range(nx):
            yield ((ix + 0.5) / nx, (iy + 0.5) / ny)


def _interface_segments(mask: torch.Tensor, spacing_yx: tuple[float, float]) -> list[tuple[float, float, float, float]]:
    """Return axis-aligned cell-interface segments as ``(x0, x1, y)`` or ``(x, y0, y1)``.

    The final tuple stores a four-value tagged representation: horizontal
    segments have ``x0 <= x1`` and ``y0 == y1``; vertical segments have
    ``x0 == x1`` and ``y0 <= y1``.  Cell edges are at half-voxel physical
    boundaries, hence a pixel centre is at ``((x + .5)dx, (y + .5)dy)``.
    """

    mask_np = mask.detach().cpu().bool().numpy()
    h, w = mask_np.shape
    dy, dx = spacing_yx
    segments: list[tuple[float, float, float, float]] = []

    # Horizontal faces: compare the cell immediately above and below each
    # y-edge.  Outside-image cells are background, making full foreground masks
    # finite and giving them the image boundary as their interface.
    for y_edge in range(h + 1):
        above = mask_np[y_edge - 1] if y_edge > 0 else None
        below = mask_np[y_edge] if y_edge < h else None
        if above is None:
            transition = below
        elif below is None:
            transition = above
        else:
            transition = above != below
        for x in (int(v) for v in (transition.nonzero()[0] if transition is not None else [])):
            segments.append((x * dx, (x + 1) * dx, y_edge * dy, y_edge * dy))

    # Vertical faces: compare cells left/right of each x-edge.
    for x_edge in range(w + 1):
        left = mask_np[:, x_edge - 1] if x_edge > 0 else None
        right = mask_np[:, x_edge] if x_edge < w else None
        if left is None:
            transition = right
        elif right is None:
            transition = left
        else:
            transition = left != right
        for y in (int(v) for v in (transition.nonzero()[0] if transition is not None else [])):
            segments.append((x_edge * dx, x_edge * dx, y * dy, (y + 1) * dy))
    return segments


def _distance_to_segments(
    points_xy: torch.Tensor,
    segments: list[tuple[float, float, float, float]],
    *,
    point_chunk: int = 2048,
    segment_chunk: int = 4096,
) -> torch.Tensor:
    """Exact point-to-axis-segment distance with bounded temporary memory."""

    if not segments:
        return torch.full((points_xy.shape[0],), float("inf"), dtype=torch.float64, device=points_xy.device)
    seg = torch.tensor(segments, dtype=torch.float64, device=points_xy.device)
    result = torch.full((points_xy.shape[0],), float("inf"), dtype=torch.float64, device=points_xy.device)
    # Nested chunks keep a checkerboard mask from allocating N_points x
    # N_segments at once.  Typical anatomical masks have a small boundary;
    # this remains exact for dense masks as well, only taking longer.
    for p0 in range(0, points_xy.shape[0], point_chunk):
        p = points_xy[p0 : p0 + point_chunk].to(torch.float64)
        best = torch.full((p.shape[0],), float("inf"), dtype=torch.float64, device=p.device)
        for s0 in range(0, seg.shape[0], segment_chunk):
            s = seg[s0 : s0 + segment_chunk]
            x0, x1, y0, y1 = s.unbind(1)
            horizontal = y0 == y1
            # For a horizontal segment, clamp x and retain y; for a vertical
            # segment, clamp y and retain x.  Since the bank's interface is
            # axis-aligned this is the exact Euclidean distance.
            px = p[:, 0, None]
            py = p[:, 1, None]
            hx = torch.maximum(x0[None], torch.minimum(px, x1[None]))
            hy = y0[None].expand_as(hx)
            vx = x0[None].expand_as(hx)
            vy = torch.maximum(y0[None], torch.minimum(py, y1[None]))
            d2_h = (px - hx).square() + (py - hy).square()
            d2_v = (px - vx).square() + (py - vy).square()
            d2 = torch.where(horizontal[None], d2_h, d2_v)
            best = torch.minimum(best, d2.min(dim=1).values.sqrt())
        result[p0 : p0 + p.shape[0]] = best
    return result


class QueryBank:
    """Deterministic finite set of anatomical measurements."""

    def __init__(self, config: BankConfig | None = None):
        self.config = config or BankConfig()
        queries: list[Query] = []
        for kind in self.config.types:
            grid = self.config.region_grid if kind == "REGION" else self.config.boundary_grid
            centres = tuple(_grid_centres(grid))
            for class_id in self.config.classes:
                for u, v in centres:
                    scales = self.config.region_scales if kind == "REGION" else (1.0,)
                    for scale in scales:
                        idx = len(queries)
                        query_id = _format_query_id(kind, class_id, u, v, scale)
                        queries.append(Query(idx, kind, class_id, u, v, scale, query_id))
        self.queries = tuple(queries)
        self._by_id = {q.query_id: q for q in self.queries}
        if len(self._by_id) != len(self.queries):
            raise ValueError("bank query IDs are not unique")
        canonical = {"config": self.config.canonical(), "queries": [q.canonical() for q in self.queries]}
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.fingerprint = hashlib.sha256(encoded).hexdigest()

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def query(self, j: int | str | Query) -> Query:
        if isinstance(j, Query):
            if j.index < 0 or j.index >= len(self) or self.queries[j.index] != j:
                raise ValueError("query does not belong to this bank")
            return j
        if isinstance(j, str):
            try:
                return self._by_id[j]
            except KeyError as exc:
                raise ValueError(f"unknown query ID: {j}") from exc
        if isinstance(j, bool) or int(j) != j or int(j) < 0 or int(j) >= len(self):
            raise ValueError(f"query index out of range: {j}")
        return self.queries[int(j)]

    def descriptors(self, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return encoder-independent ``[Q, 2 + C + 3]`` descriptors.

        The first two entries are type one-hot (REGION, BOUNDARY), followed by
        class one-hot, ``(u, v, scale)``.  The descriptor is metadata only; a
        decoder must not receive the source image feature through this path.
        """

        device = torch.device(device) if device is not None else torch.device("cpu")
        kind_to_i = {kind: i for i, kind in enumerate(_MEASUREMENT_TYPES)}
        class_to_i = {c: i for i, c in enumerate(self.config.classes)}
        out = torch.zeros((len(self), 2 + len(self.config.classes) + 3), dtype=dtype, device=device)
        for q in self.queries:
            out[q.index, kind_to_i[q.kind]] = 1
            out[q.index, 2 + class_to_i[q.class_id]] = 1
            out[q.index, 2 + len(self.config.classes) :] = torch.tensor((q.u, q.v, q.scale), dtype=dtype, device=device)
        return out

    def region_mask(self, j: int | str | Query, height: int, width: int, *, device=None) -> torch.Tensor:
        """Return the fixed native-pixel footprint of a REGION query."""

        q = self.query(j)
        if q.kind != "REGION":
            raise ValueError("region_mask is only defined for REGION queries")
        if isinstance(height, bool) or isinstance(width, bool) or int(height) <= 0 or int(width) <= 0:
            raise ValueError("height and width must be positive integers")
        height, width = int(height), int(width)
        yy = (torch.arange(height, device=device, dtype=torch.float64) + 0.5) / height
        xx = (torch.arange(width, device=device, dtype=torch.float64) + 0.5) / width
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        if q.scale >= 1.0:
            out = torch.ones((height, width), dtype=torch.bool, device=device)
        else:
            out = (gx - q.u).abs() <= q.scale / 2.0
            out &= (gy - q.v).abs() <= q.scale / 2.0
            # A tiny native image can otherwise have no pixel centre in a
            # fractional cell.  Selecting its nearest centre keeps every bank
            # query well-defined without consulting the label.
            if not bool(out.any()):
                ix = min(width - 1, max(0, int(math.floor(q.u * (width - 1) + 0.5))))
                iy = min(height - 1, max(0, int(math.floor(q.v * (height - 1) + 0.5))))
                out[iy, ix] = True
        return out

    @staticmethod
    def _boundary_index(q: Query, height: int, width: int) -> tuple[int, int]:
        # Normalized coordinates are mapped to the nearest native pixel centre
        # ((i + .5) / N); this is fixed by geometry and never mask-dependent.
        x = min(width - 1, max(0, int(math.floor(q.u * (width - 1) + 0.5))))
        y = min(height - 1, max(0, int(math.floor(q.v * (height - 1) + 0.5))))
        return y, x

    def _sdf_single(self, class_mask: torch.Tensor, spacing_yx: tuple[float, float]) -> torch.Tensor:
        # Empty classes have the explicit protocol target +1.  This convention
        # avoids an undefined nearest interface and is shared by target and
        # teacher code.
        if not bool(class_mask.any()):
            return torch.ones(class_mask.shape, dtype=torch.float32, device=class_mask.device)
        h, w = class_mask.shape
        dy, dx = spacing_yx
        yy = (torch.arange(h, device=class_mask.device, dtype=torch.float64) + 0.5) * dy
        xx = (torch.arange(w, device=class_mask.device, dtype=torch.float64) + 0.5) * dx
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        points = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
        distance = _distance_to_segments(points, _interface_segments(class_mask, spacing_yx)).reshape(h, w)
        signed = torch.where(class_mask, -distance, distance)
        return (signed / self.config.radius_mm).clamp(-1.0, 1.0).to(torch.float32)

    def sdf_targets(self, labels: torch.Tensor, spacing: torch.Tensor | Sequence[float]) -> torch.Tensor:
        """Build normalized signed-distance targets with shape ``[B, C, H, W]``."""

        labels = _validate_labels(labels, self.config.classes)
        sp = _validate_spacing(spacing, labels.shape[0], device=labels.device)
        batch, height, width = labels.shape
        out = torch.empty((batch, self.config.num_channels, height, width), dtype=torch.float32, device=labels.device)
        for b in range(batch):
            spacing_yx = (float(sp[b, 0]), float(sp[b, 1]))
            for class_id in range(self.config.num_channels):
                out[b, class_id] = self._sdf_single(labels[b] == class_id, spacing_yx)
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite signed-distance target")
        return out

    def targets(self, labels: torch.Tensor, spacing: torch.Tensor | Sequence[float]) -> torch.Tensor:
        """Return ``[B, Q]`` REGION/BOUNDARY values from ground-truth labels."""

        labels = _validate_labels(labels, self.config.classes)
        sp = _validate_spacing(spacing, labels.shape[0], device=labels.device)
        batch, height, width = labels.shape
        sdf = None
        out = torch.empty((batch, len(self)), dtype=torch.float32, device=labels.device)
        masks = {q.index: self.region_mask(q, height, width, device=labels.device) for q in self.queries if q.kind == "REGION"}
        for q in self.queries:
            if q.kind == "REGION":
                region = masks[q.index]
                denom = float(region.sum())
                out[:, q.index] = ((labels == q.class_id) & region[None]).sum(dim=(1, 2)).float() / denom
            else:
                if sdf is None:
                    sdf = self.sdf_targets(labels, sp)
                y, x = self._boundary_index(q, height, width)
                out[:, q.index] = sdf[:, q.class_id, y, x]
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite query target")
        return out

    def teacher_values(self, region_probs: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        """Read query values from teacher probabilities and normalized SDF outputs."""

        if not isinstance(region_probs, torch.Tensor) or region_probs.ndim != 4:
            raise ValueError("region_probs must have shape [B, C, H, W]")
        if not isinstance(sdf, torch.Tensor) or sdf.ndim != 4 or sdf.shape != region_probs.shape:
            raise ValueError("sdf must have the same shape [B, C, H, W] as region_probs")
        if region_probs.shape[1] < self.config.num_channels:
            raise ValueError("teacher tensors do not contain all configured class channels")
        if not torch.isfinite(region_probs).all() or not torch.isfinite(sdf).all():
            raise ValueError("teacher tensors contain non-finite values")
        if (region_probs < -1e-5).any() or (region_probs > 1.00001).any():
            raise ValueError("region_probs must be probabilities in [0, 1]")
        if (sdf < -1.00001).any() or (sdf > 1.00001).any():
            raise ValueError("teacher sdf must be normalized to [-1, 1]")
        batch, _, height, width = region_probs.shape
        out = torch.empty((batch, len(self)), dtype=region_probs.dtype, device=region_probs.device)
        masks = {q.index: self.region_mask(q, height, width, device=region_probs.device) for q in self.queries if q.kind == "REGION"}
        for q in self.queries:
            if q.kind == "REGION":
                region = masks[q.index]
                out[:, q.index] = region_probs[:, q.class_id][..., region].mean(dim=-1)
            else:
                y, x = self._boundary_index(q, height, width)
                out[:, q.index] = sdf[:, q.class_id, y, x]
        return out
