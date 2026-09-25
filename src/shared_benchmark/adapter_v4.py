"""Joint, topology-only resolver for ``cardiac_adapter_v4``.

The resolver never changes a source component's pixels.  It ranks complete
MYO/LV hypotheses before assigning any foreground labels, then keeps only
labels that remain stable among near-optimal hypotheses.  This avoids the
v2/v3 failure mode where an early BG or enclosure decision removes evidence
needed by later rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Any, Mapping

import numpy as np

from .provenance import sha256_json
from .region_graph import RegionComponent, RegionGraph, build_region_graph
from .semantic_contract import (
    BG, LV, MYO, RV, VOID, AdapterResult, array_hash, adapter_metadata_payload,
    canonical_metadata_hash, semantic_artifact_stage,
)
from .spatial import grid_hash


@dataclass(frozen=True)
class _LVMYOCandidate:
    outer: int
    inner: tuple[int, ...]
    enclosure_support: float
    score: float
    source: str


def _by_index(graph: RegionGraph) -> dict[int, RegionComponent]:
    return {component.index: component for component in graph.components}


def _perimeter(component: RegionComponent, shape: tuple[int, int]) -> int:
    pixels = set(component.pixels)
    height, width = shape
    count = 0
    for row, col in pixels:
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if nr < 0 or nc < 0 or nr >= height or nc >= width or (nr, nc) not in pixels:
                count += 1
    return count


def _group_perimeter(components: tuple[RegionComponent, ...], shape: tuple[int, int]) -> int:
    pixels = set().union(*(set(component.pixels) for component in components))
    height, width = shape
    count = 0
    for row, col in pixels:
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if nr < 0 or nc < 0 or nr >= height or nc >= width or (nr, nc) not in pixels:
                count += 1
    return count


def _connected_group(members: tuple[int, ...], by_index: Mapping[int, RegionComponent]) -> bool:
    """Reject a semantic group made of disconnected, unrelated fragments."""
    if len(members) < 2:
        return True
    member_set = set(members)
    visited = {members[0]}
    pending = [members[0]]
    while pending:
        current = pending.pop()
        for neighbor, _ in by_index[current].adjacency:
            if neighbor in member_set and neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return visited == member_set


def _candidate_score(outer: RegionComponent, inner: tuple[RegionComponent, ...], support: float, shape: tuple[int, int]) -> float:
    area_fraction = (outer.area + sum(component.area for component in inner)) / float(shape[0] * shape[1])
    return 0.70 * support + 0.30 * sqrt(area_fraction)


def _lv_myo_candidates(graph: RegionGraph, solver: Mapping[str, Any]) -> list[_LVMYOCandidate]:
    """Return exact-hole and high-support open-ring candidates.

    A group of components in the same MYO hole is one LV candidate.  This is
    the v4 many-to-one case: over-clustered LV pixels can remain separate in
    the raw partition yet obtain the same semantic label.
    """
    by_index = _by_index(graph)
    max_group = int(solver["max_lv_group_components"])
    minimum_support = float(solver["min_soft_enclosure_support"])
    candidates: dict[tuple[int, tuple[int, ...]], _LVMYOCandidate] = {}

    for outer in graph.components:
        if outer.border_contact:
            continue
        outer_adjacency = dict(outer.adjacency)
        for hole in outer.hole_pixels:
            members = tuple(sorted(
                component.index for component in graph.components
                if not component.border_contact and component.index != outer.index
                and set(component.pixels).issubset(hole)
            ))
            if not members or len(members) > max_group:
                continue
            inner = tuple(by_index[index] for index in members)
            if not any(index in outer_adjacency for index in members):
                continue
            key = (outer.index, members)
            candidates[key] = _LVMYOCandidate(
                outer=outer.index, inner=members, enclosure_support=1.0,
                score=_candidate_score(outer, inner, 1.0, graph.shape), source="exact_hole_group",
            )

        # A soft enclosure may contain several adjacent over-clustered LV
        # pieces.  Score their union so internal split boundaries do not count
        # as an anatomical gap.
        neighbors = tuple(index for index, _ in sorted(
            (
                (index, adjacency) for index, adjacency in outer_adjacency.items()
                if not by_index[index].border_contact and index != outer.index
            ),
            key=lambda row: (-row[1], row[0]),
        )[:int(solver["max_neighbor_components"])])
        for group_size in range(1, min(max_group, len(neighbors)) + 1):
            for members in combinations(neighbors, group_size):
                if not _connected_group(members, by_index):
                    continue
                inner = tuple(by_index[index] for index in members)
                boundary = _group_perimeter(inner, graph.shape)
                support = sum(outer_adjacency[index] for index in members) / float(boundary) if boundary else 0.0
                if support < minimum_support:
                    continue
                key = (outer.index, members)
                candidate = _LVMYOCandidate(
                    outer=outer.index, inner=members, enclosure_support=support,
                    score=_candidate_score(outer, inner, support, graph.shape), source="soft_enclosure",
                )
                prior = candidates.get(key)
                if prior is None or candidate.score > prior.score:
                    candidates[key] = candidate
    return sorted(candidates.values(), key=lambda item: (-item.score, item.outer, item.inner, item.source))[:int(solver["max_hypotheses"])]


def _rv_choice(graph: RegionGraph, myo_index: int, excluded: set[int], solver: Mapping[str, Any]) -> tuple[int | None, dict[str, Any]]:
    by_index = _by_index(graph)
    myo = by_index[myo_index]
    options: list[tuple[float, int, int]] = []
    for component in graph.components:
        if component.index in excluded or component.border_contact:
            continue
        adjacency = dict(myo.adjacency).get(component.index, 0)
        if not adjacency:
            continue
        perimeter = _perimeter(component, graph.shape)
        score = adjacency / float(perimeter) + 0.10 * sqrt(component.area_fraction) if perimeter else 0.0
        options.append((score, component.index, adjacency))
    options.sort(key=lambda row: (-row[0], row[1]))
    if not options or options[0][0] < float(solver["min_rv_score"]):
        return None, {"status": "unsupported_rv", "candidates": []}
    best = options[0]
    if len(options) > 1 and best[0] - options[1][0] < float(solver["rv_margin"]):
        return None, {
            "status": "ambiguous_rv", "candidates": [
                {"component_index": index, "score": score, "adjacency": adjacency}
                for score, index, adjacency in options
            ],
        }
    return best[1], {
        "status": "resolved", "score": best[0], "adjacency": best[2],
        "candidates": [{"component_index": index, "score": score, "adjacency": adjacency} for score, index, adjacency in options],
    }


def _component_labels(graph: RegionGraph, candidate: _LVMYOCandidate, solver: Mapping[str, Any]) -> tuple[dict[int, int], dict[str, Any]]:
    labels = {component.index: BG for component in graph.components if component.border_contact}
    labels[candidate.outer] = MYO
    labels.update({index: LV for index in candidate.inner})
    rv, rv_trace = _rv_choice(graph, candidate.outer, set(labels), solver)
    if rv is not None:
        labels[rv] = RV
    return labels, rv_trace


def adapt_partition_v4(
    record: Mapping[str, Any], partition: np.ndarray, central_image: np.ndarray, *,
    spec: Mapping[str, Any], spec_hash: str, clean_record: Mapping[str, Any],
    partition_value: np.ndarray, image_value: np.ndarray, manifest_hash: str,
) -> AdapterResult:
    """Resolve one partition with joint hypotheses under the frozen v4 spec."""
    del record, partition, central_image
    graph = build_region_graph(partition_value)
    solver = spec["solver"]
    candidates = _lv_myo_candidates(graph, solver)
    eligible = [candidate for candidate in candidates if candidate.score >= float(solver["min_lv_myo_score"])]
    best = eligible[0] if eligible else None
    near_best = [] if best is None else [
        candidate for candidate in eligible
        if best.score - candidate.score < float(solver["hypothesis_margin"])
    ]

    selected_labels: dict[int, int] = {}
    rv_trace: dict[str, Any] = {"status": "unsupported_rv", "candidates": []}
    if best is not None:
        selected_labels, rv_trace = _component_labels(graph, best, solver)
        # A label is emitted only when every near-optimal hypothesis gives the
        # same component the same semantic role.  BG is provisional in the
        # objective but stable because v4 foreground candidates are non-border.
        alternatives = [_component_labels(graph, candidate, solver)[0] for candidate in near_best]
        selected_labels = {
            index: label for index, label in selected_labels.items()
            if all(alternative.get(index) == label for alternative in alternatives)
        }

    semantic = np.full(partition_value.shape, VOID, dtype=np.uint8)
    by_index = _by_index(graph)
    for index, label in selected_labels.items():
        component = by_index[index]
        for row, col in component.pixels:
            semantic[row, col] = label
    validity = np.asarray(semantic != VOID, dtype=bool)

    assignment_reasons: dict[str, str] = {}
    void_reasons: dict[str, str] = {}
    for component in graph.components:
        label = selected_labels.get(component.index)
        if label == BG:
            assignment_reasons[component.key] = "border_background_consensus"
        elif label == MYO:
            assignment_reasons[component.key] = best.source if best else "unassigned_component"
        elif label == LV:
            assignment_reasons[component.key] = "lv_group_" + (best.source if best else "unassigned")
        elif label == RV:
            assignment_reasons[component.key] = "highest_margin_adjacent_rv"
        else:
            if best is None:
                void_reasons[component.key] = "no_confident_lv_myo_hypothesis"
            elif component.index in {best.outer, *best.inner}:
                void_reasons[component.key] = "hypothesis_conflict"
            elif rv_trace["status"] == "ambiguous_rv" and any(row["component_index"] == component.index for row in rv_trace["candidates"]):
                void_reasons[component.key] = "ambiguous_rv"
            else:
                void_reasons[component.key] = "unassigned_component"

    role_reasons = {
        "BG": "resolved" if any(label == BG for label in selected_labels.values()) else "unsupported_background",
        "MYO": "resolved" if any(label == MYO for label in selected_labels.values()) else "no_confident_lv_myo_hypothesis",
        "LV": "resolved" if any(label == LV for label in selected_labels.values()) else "no_confident_lv_myo_hypothesis",
        "RV": rv_trace["status"],
    }
    assignments = [
        {
            "component_key": component.key,
            "source_cluster_id": component.source_id,
            "semantic": {BG: "BG", RV: "RV", MYO: "MYO", LV: "LV"}.get(selected_labels.get(component.index), "VOID"),
            "semantic_id": selected_labels.get(component.index, VOID),
            "reason": assignment_reasons.get(component.key, void_reasons.get(component.key, "unassigned_component")),
        }
        for component in graph.components
    ]
    unresolved = {role: reason for role, reason in role_reasons.items() if reason != "resolved"}
    unresolved.update({f"component:{key}": reason for key, reason in void_reasons.items()})
    hypothesis_trace = [
        {
            "myo_component": graph_component.key,
            "lv_components": [by_index[index].key for index in candidate.inner],
            "enclosure_support": candidate.enclosure_support,
            "score": candidate.score,
            "source": candidate.source,
            "near_optimal": candidate in near_best,
        }
        for candidate in candidates
        for graph_component in (by_index[candidate.outer],)
    ]
    semantic_digest = array_hash(semantic)
    validity_digest = array_hash(validity)
    grid_digest = grid_hash(clean_record["shared_grid"])
    scientific_result_hash = sha256_json({
        "adapter_version": str(spec["adapter_version"]), "adapter_config_sha256": spec_hash,
        "sample_id": clean_record["sample_id"], "shared_manifest_sha256": manifest_hash,
        "shared_grid_sha256": grid_digest, "component_graph_digest": graph.digest,
        "semantic_map_sha256": semantic_digest, "validity_map_sha256": validity_digest,
    })
    metadata: dict[str, Any] = {
        "adapter_version": str(spec["adapter_version"]),
        "input_schema_version": str(spec["input_schema_version"]),
        "output_schema_version": str(spec["output_schema_version"]),
        "sample_id": clean_record["sample_id"], "shared_manifest_sha256": manifest_hash,
        "partition_sha256": array_hash(partition_value), "central_image_sha256": array_hash(image_value),
        "adapter_config_sha256": spec_hash, "shared_grid_sha256": grid_digest,
        "component_graph_digest": graph.digest, "assignments": assignments,
        "assignment_reasons": dict(sorted(assignment_reasons.items())),
        "void_reasons": dict(sorted(void_reasons.items())), "role_reasons": dict(sorted(role_reasons.items())),
        "unresolved_reasons": dict(sorted(unresolved.items())), "coverage": float(validity.mean()),
        "semantic_map_sha256": semantic_digest, "validity_map_sha256": validity_digest,
        "scientific_result_sha256": scientific_result_hash, "intensity_resolution": "disabled",
        "orientation_resolution": "disabled", "region_splitting": False,
        "semantic_artifact_stage": semantic_artifact_stage(str(spec["adapter_version"])),
        "solver": {
            "candidate_count": len(candidates), "eligible_count": len(eligible),
            "near_optimal_count": len(near_best), "search_truncated": len(candidates) >= int(solver["max_hypotheses"]),
            "best_score": None if best is None else best.score, "hypotheses": hypothesis_trace,
            "rv": rv_trace,
        },
    }
    metadata["metadata_sha256"] = canonical_metadata_hash(adapter_metadata_payload(metadata))
    return AdapterResult(semantic_map=semantic, validity_map=validity, metadata=metadata)
