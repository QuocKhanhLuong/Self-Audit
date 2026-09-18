"""Deterministic, raw-ID-independent region graph for cardiac_adapter_v1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .provenance import sha256_json


CONNECTIVITY = 4


def _neighbors(row: int, col: int, height: int, width: int) -> Iterable[tuple[int, int]]:
    if row > 0:
        yield row - 1, col
    if row + 1 < height:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < width:
        yield row, col + 1


def _component_key(pixels: tuple[tuple[int, int], ...]) -> str:
    rows = [pixel[0] for pixel in pixels]
    cols = [pixel[1] for pixel in pixels]
    return f"r{min(rows)}c{min(cols)}R{max(rows)}C{max(cols)}A{len(pixels)}"


def _component_geometry_digest(pixels: tuple[tuple[int, int], ...]) -> str:
    """Disambiguate equal frozen bbox/area keys without using raw IDs."""
    return sha256_json([[int(row), int(col)] for row, col in pixels])[:16]


@dataclass(frozen=True)
class RegionComponent:
    """One maximal 4-connected component of one opaque raw cluster."""

    index: int
    key: str
    source_id: int
    pixels: tuple[tuple[int, int], ...]
    area: int
    area_fraction: float
    border_contact: bool
    border_pixel_count: int
    centroid_rc: tuple[float, float]
    centroid_normalized_rc: tuple[float, float]
    bbox_rc: tuple[int, int, int, int]
    adjacency: tuple[tuple[int, int], ...]
    hole_pixels: tuple[frozenset[tuple[int, int]], ...]

    @property
    def is_border(self) -> bool:
        return self.border_contact


@dataclass(frozen=True)
class RegionGraph:
    """Canonical graph whose node order and digest do not use raw ID values."""

    shape: tuple[int, int]
    components: tuple[RegionComponent, ...]
    pixel_component: np.ndarray
    digest: str

    def component_encloses(self, outer: RegionComponent, inner: RegionComponent) -> bool:
        if outer.index == inner.index or outer.border_contact or inner.border_contact:
            return False
        if inner.index not in dict(outer.adjacency):
            return False
        inner_pixels = set(inner.pixels)
        hole_union = set().union(*(set(hole) for hole in outer.hole_pixels))
        return inner_pixels.issubset(hole_union)

    def enclosure_pairs(self, *, excluded: set[int] | None = None) -> tuple[tuple[RegionComponent, RegionComponent], ...]:
        excluded = set() if excluded is None else set(excluded)
        pairs: list[tuple[RegionComponent, RegionComponent]] = []
        by_index = {component.index: component for component in self.components}
        for outer in self.components:
            if outer.index in excluded:
                continue
            for inner_index, _ in outer.adjacency:
                if inner_index in excluded or inner_index == outer.index:
                    continue
                inner = by_index[inner_index]
                if self.component_encloses(outer, inner):
                    pairs.append((outer, inner))
        return tuple(pairs)

    def component_fully_inside_hole(self, component: RegionComponent, outer: RegionComponent) -> bool:
        pixels = set(component.pixels)
        hole_union = set().union(*(set(hole) for hole in outer.hole_pixels))
        return pixels.issubset(hole_union)


def _holes(mask: np.ndarray) -> tuple[frozenset[tuple[int, int]], ...]:
    """Return complement components not connected to the global image border."""
    height, width = mask.shape
    occupied = np.asarray(mask, dtype=bool)
    occupied_rows, occupied_cols = np.nonzero(occupied)
    if occupied_rows.size == 0:
        return tuple()

    # A non-border component's holes are confined to its bbox.  A one-pixel
    # complement collar preserves exact connectivity to the full-FOV exterior
    # while avoiding a full-image flood fill for every small component.  Border
    # components use the full image because a border-touching component can
    # still contain a separate enclosed complement pocket.
    touches_border = bool(
        occupied_rows.min() == 0
        or occupied_cols.min() == 0
        or occupied_rows.max() == height - 1
        or occupied_cols.max() == width - 1
    )
    if touches_border:
        top, left, bottom, right = 0, 0, height, width
        work = occupied
    else:
        top = int(occupied_rows.min()) - 1
        left = int(occupied_cols.min()) - 1
        bottom = int(occupied_rows.max()) + 2
        right = int(occupied_cols.max()) + 2
        work = occupied[top:bottom, left:right]

    work_height, work_width = work.shape
    visited = np.zeros(work.shape, dtype=bool)

    def _visit_from_boundary(row: int, col: int) -> None:
        if work[row, col] or visited[row, col]:
            return
        stack = [(row, col)]
        visited[row, col] = True
        while stack:
            current_row, current_col = stack.pop()
            for next_row, next_col in _neighbors(current_row, current_col, work_height, work_width):
                if not work[next_row, next_col] and not visited[next_row, next_col]:
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

    for col in range(work_width):
        _visit_from_boundary(0, col)
        _visit_from_boundary(work_height - 1, col)
    for row in range(work_height):
        _visit_from_boundary(row, 0)
        _visit_from_boundary(row, work_width - 1)

    holes: list[frozenset[tuple[int, int]]] = []
    for row in range(work_height):
        for col in range(work_width):
            if work[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                pixels.append((current_row + top, current_col + left))
                for next_row, next_col in _neighbors(current_row, current_col, work_height, work_width):
                    if not work[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
            holes.append(frozenset(pixels))
    return tuple(sorted(holes, key=lambda hole: (min(hole), len(hole))))


def _canonical_graph_payload(graph_shape: tuple[int, int], components: tuple[RegionComponent, ...]) -> dict[str, Any]:
    groups: dict[int, list[str]] = {}
    nodes: list[dict[str, Any]] = []
    edges: set[tuple[str, str, int]] = set()
    containments: set[tuple[str, str]] = set()
    key_by_index = {component.index: component.key for component in components}
    component_by_index = {component.index: component for component in components}
    for component in components:
        groups.setdefault(component.source_id, []).append(component.key)
        nodes.append({
            "key": component.key,
            "area": component.area,
            "area_fraction_numerator": component.area,
            "area_fraction_denominator": graph_shape[0] * graph_shape[1],
            "border_contact": component.border_contact,
            "border_pixel_count": component.border_pixel_count,
            "centroid_rc": list(component.centroid_rc),
            "centroid_normalized_rc": list(component.centroid_normalized_rc),
            "bbox_rc": list(component.bbox_rc),
            "hole_areas": sorted(len(hole) for hole in component.hole_pixels),
        })
        for other_index, boundary_length in component.adjacency:
            left, right = sorted((component.key, key_by_index[other_index]))
            edges.add((left, right, boundary_length))
        hole_union = set().union(*(set(hole) for hole in component.hole_pixels))
        for inner_index, _ in component.adjacency:
            inner = component_by_index[inner_index]
            if (
                not component.border_contact
                and not inner.border_contact
                and set(inner.pixels).issubset(hole_union)
            ):
                containments.add((component.key, inner.key))
    # Raw values are intentionally omitted.  Equality classes are represented
    # only by canonical component keys, so an ID permutation has no effect.
    equivalence_groups = sorted(sorted(keys) for keys in groups.values())
    return {
        "shape": list(graph_shape),
        "nodes": sorted(nodes, key=lambda node: node["key"]),
        "edges": sorted(edges),
        "containments": sorted(containments),
        "source_equivalence_groups": equivalence_groups,
    }


def build_region_graph(partition: np.ndarray) -> RegionGraph:
    """Extract components/holes/adjacency using only the frozen topology."""
    values = np.asarray(partition)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("anonymous partition must be a 2-D integer array")
    height, width = (int(values.shape[0]), int(values.shape[1]))
    if height < 1 or width < 1:
        raise ValueError("anonymous partition must be non-empty")
    visited = np.zeros(values.shape, dtype=bool)
    discovered: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for row in range(height):
        for col in range(width):
            if visited[row, col]:
                continue
            raw_id = int(values[row, col])
            stack = [(row, col)]
            visited[row, col] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                pixels.append((current_row, current_col))
                for next_row, next_col in _neighbors(current_row, current_col, height, width):
                    if not visited[next_row, next_col] and int(values[next_row, next_col]) == raw_id:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
            discovered.append((raw_id, tuple(sorted(pixels))))
    discovered.sort(key=lambda item: (_component_key(item[1]), item[1]))
    base_key_counts: dict[str, int] = {}
    for _, pixels in discovered:
        base_key = _component_key(pixels)
        base_key_counts[base_key] = base_key_counts.get(base_key, 0) + 1
    pixel_component = np.full(values.shape, -1, dtype=np.int32)
    components_mutable: list[dict[str, Any]] = []
    for index, (raw_id, pixels) in enumerate(discovered):
        base_key = _component_key(pixels)
        key = base_key
        if base_key_counts[base_key] > 1:
            key = f"{base_key}G{_component_geometry_digest(pixels)}"
        mask = np.zeros(values.shape, dtype=bool)
        rows = np.fromiter((pixel[0] for pixel in pixels), dtype=np.intp)
        cols = np.fromiter((pixel[1] for pixel in pixels), dtype=np.intp)
        mask[rows, cols] = True
        pixel_component[rows, cols] = index
        area = len(pixels)
        border_flags = [row in (0, height - 1) or col in (0, width - 1) for row, col in pixels]
        centroid = (sum(row for row, _ in pixels) / area, sum(col for _, col in pixels) / area)
        normalized = (
            centroid[0] / (height - 1) if height > 1 else 0.0,
            centroid[1] / (width - 1) if width > 1 else 0.0,
        )
        components_mutable.append({
            "index": index,
            "key": key,
            "source_id": raw_id,
            "pixels": pixels,
            "area": area,
            "area_fraction": area / float(height * width),
            "border_contact": any(border_flags),
            "border_pixel_count": sum(border_flags),
            "centroid_rc": (float(centroid[0]), float(centroid[1])),
            "centroid_normalized_rc": (float(normalized[0]), float(normalized[1])),
            "bbox_rc": tuple(int(value) for value in (min(rows), min(cols), max(rows), max(cols))),
            "hole_pixels": _holes(mask),
            "adjacency": {},
        })
    for row in range(height):
        for col in range(width):
            current = int(pixel_component[row, col])
            for next_row, next_col in ((row + 1, col), (row, col + 1)):
                if next_row >= height or next_col >= width:
                    continue
                other = int(pixel_component[next_row, next_col])
                if current != other:
                    components_mutable[current]["adjacency"][other] = components_mutable[current]["adjacency"].get(other, 0) + 1
                    components_mutable[other]["adjacency"][current] = components_mutable[other]["adjacency"].get(current, 0) + 1
    components = tuple(
        RegionComponent(
            index=item["index"], key=item["key"], source_id=item["source_id"], pixels=item["pixels"],
            area=item["area"], area_fraction=item["area_fraction"], border_contact=item["border_contact"],
            border_pixel_count=item["border_pixel_count"], centroid_rc=item["centroid_rc"],
            centroid_normalized_rc=item["centroid_normalized_rc"], bbox_rc=item["bbox_rc"],
            adjacency=tuple(sorted(item["adjacency"].items())), hole_pixels=item["hole_pixels"],
        )
        for item in components_mutable
    )
    digest = sha256_json(_canonical_graph_payload((height, width), components))
    return RegionGraph(shape=(height, width), components=components, pixel_component=pixel_component, digest=digest)
