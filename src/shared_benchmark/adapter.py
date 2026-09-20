"""Frozen, topology-only implementation of ``cardiac_adapter_v2``.

The adapter consumes an anonymous partition and a shared image-only record. It
copies existing component pixels into semantic labels; it never creates a
boundary and never reads GT or method-specific evidence.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .provenance import sha256_json
from .region_graph import RegionComponent, RegionGraph, build_region_graph
from .semantic_contract import (
    ADAPTER_INPUT_SCHEMA_VERSION,
    ADAPTER_OUTPUT_SCHEMA_VERSION,
    ADAPTER_VERSION,
    BG,
    LV,
    MYO,
    RV,
    VOID,
    AdapterContractError,
    AdapterResult,
    adapter_metadata_payload,
    array_hash,
    canonical_metadata_hash,
    load_and_validate_spec,
    validate_adapter_record,
)
from .spatial import SPATIAL_CONTRACT_VERSION, grid_hash


def _assign(
    semantic: np.ndarray,
    component: RegionComponent,
    label: int,
    assignments: dict[str, int],
    assignment_reasons: dict[str, str],
    unresolved_reasons: dict[str, str],
    reason: str,
) -> None:
    if assignments.get(component.key, label) != label:
        raise AdapterContractError(f"conflicting semantic assignment for {component.key}")
    assignments[component.key] = label
    assignment_reasons[component.key] = reason
    unresolved_reasons.pop(component.key, None)
    for row, col in component.pixels:
        semantic[row, col] = label


def _void_component(component: RegionComponent, reason: str, reasons: dict[str, str]) -> None:
    reasons.setdefault(component.key, reason)


def _role_name(label: int) -> str:
    return {BG: "BG", RV: "RV", MYO: "MYO", LV: "LV"}[label]


def _background(
    graph: RegionGraph, semantic: np.ndarray, assignments: dict[str, int],
    assignment_reasons: dict[str, str], unresolved_reasons: dict[str, str],
) -> dict[str, str]:
    candidates = tuple(component for component in graph.components if component.border_contact)
    if not candidates:
        return {"BG": "unsupported_cardinality"}
    largest_area = max(component.area for component in candidates)
    seeds = tuple(component for component in candidates if component.area == largest_area)
    if len(seeds) != 1:
        for component in candidates:
            _void_component(component, "ambiguous_bg", unresolved_reasons)
        return {"BG": "ambiguous_bg"}
    seed = seeds[0]
    for component in candidates:
        # Equality is used only to preserve border-touching fragments of the
        # same opaque anonymous cluster.  A disconnected non-border fragment
        # is deliberately not promoted to BG.
        if component.border_contact and component.source_id == seed.source_id:
            _assign(
                semantic, component, BG, assignments, assignment_reasons,
                unresolved_reasons, "unique_border_background",
            )
    return {"BG": "resolved"}


def _resolve_enclosure(
    graph: RegionGraph,
    semantic: np.ndarray,
    assignments: dict[str, int],
    assignment_reasons: dict[str, str],
    unresolved_reasons: dict[str, str],
) -> dict[str, str]:
    excluded = {component.index for component in graph.components if component.key in assignments}
    pairs = graph.enclosure_pairs(excluded=excluded)
    if len(pairs) == 1:
        outer, inner = pairs[0]
        _assign(
            semantic, outer, MYO, assignments, assignment_reasons,
            unresolved_reasons, "unique_enclosure_outer",
        )
        _assign(
            semantic, inner, LV, assignments, assignment_reasons,
            unresolved_reasons, "unique_enclosure_inner",
        )
        return {"MYO": "resolved", "LV": "resolved"}
    if len(pairs) > 1:
        for outer, inner in pairs:
            _void_component(outer, "ambiguous_enclosure", unresolved_reasons)
            _void_component(inner, "ambiguous_enclosure", unresolved_reasons)
        return {"MYO": "ambiguous_enclosure", "LV": "ambiguous_enclosure"}
    for component in graph.components:
        if component.key not in assignments and not component.border_contact:
            _void_component(component, "missing_enclosed_cavity", unresolved_reasons)
    return {"MYO": "missing_enclosed_cavity", "LV": "missing_enclosed_cavity"}


def _resolve_rv(
    graph: RegionGraph,
    semantic: np.ndarray,
    assignments: dict[str, int],
    assignment_reasons: dict[str, str],
    unresolved_reasons: dict[str, str],
) -> dict[str, str]:
    myo = tuple(component for component in graph.components if assignments.get(component.key) == MYO)
    if len(myo) != 1:
        return {"RV": "unsupported_rv"}
    myo_component = myo[0]
    candidates: list[RegionComponent] = []
    for component in graph.components:
        if component.key in assignments:
            continue
        if graph.component_fully_inside_hole(component, myo_component):
            continue
        if dict(myo_component.adjacency).get(component.index, 0) > 0:
            candidates.append(component)
    if len(candidates) == 1:
        _assign(
            semantic, candidates[0], RV, assignments, assignment_reasons,
            unresolved_reasons, "unique_adjacent_rv",
        )
        return {"RV": "resolved"}
    if len(candidates) > 1:
        for component in candidates:
            _void_component(component, "ambiguous_rv", unresolved_reasons)
        return {"RV": "ambiguous_rv"}
    return {"RV": "unsupported_rv"}


def _assignment_trace(
    graph: RegionGraph, assignments: Mapping[str, int], assignment_reasons: Mapping[str, str],
    unresolved_reasons: Mapping[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "component_key": component.key,
            "source_cluster_id": component.source_id,
            "semantic": _role_name(assignments[component.key]) if component.key in assignments else "VOID",
            "semantic_id": assignments.get(component.key, VOID),
            "reason": assignment_reasons.get(
                component.key, unresolved_reasons.get(component.key, "unassigned_component")
            ),
        }
        for component in graph.components
    ]


def _unresolved_reasons(
    role_reasons: Mapping[str, str], component_reasons: Mapping[str, str]
) -> dict[str, str]:
    unresolved: dict[str, str] = {
        role: reason for role, reason in role_reasons.items() if reason != "resolved"
    }
    unresolved.update({f"component:{key}": reason for key, reason in component_reasons.items()})
    return dict(sorted(unresolved.items()))


def adapt_partition(
    record: Mapping[str, Any],
    partition: np.ndarray,
    central_image: np.ndarray,
    *,
    adapter_spec: Mapping[str, Any],
) -> AdapterResult:
    """Resolve one anonymous partition under the immutable v2 contract."""
    spec, spec_hash = load_and_validate_spec(adapter_spec)
    clean_record, partition_value, image_value, manifest_hash = validate_adapter_record(record, partition, central_image)
    graph = build_region_graph(partition_value)
    semantic = np.full(partition_value.shape, VOID, dtype=np.uint8)
    assignments: dict[str, int] = {}
    assignment_reasons: dict[str, str] = {}
    component_reasons: dict[str, str] = {}
    role_reasons: dict[str, str] = {}
    role_reasons.update(_background(graph, semantic, assignments, assignment_reasons, component_reasons))
    role_reasons.update(_resolve_enclosure(graph, semantic, assignments, assignment_reasons, component_reasons))
    role_reasons.update(_resolve_rv(graph, semantic, assignments, assignment_reasons, component_reasons))
    for component in graph.components:
        if component.key not in assignments and component.key not in component_reasons:
            component_reasons[component.key] = "unassigned_component"
    validity = np.asarray(semantic != VOID, dtype=bool)
    partition_digest = array_hash(partition_value)
    image_digest = array_hash(image_value)
    semantic_digest = array_hash(semantic)
    validity_digest = array_hash(validity)
    grid_digest = grid_hash(clean_record["shared_grid"])
    trace = _assignment_trace(graph, assignments, assignment_reasons, component_reasons)
    unresolved = _unresolved_reasons(role_reasons, component_reasons)
    scientific_result_hash = sha256_json({
        "adapter_version": ADAPTER_VERSION,
        "adapter_config_sha256": spec_hash,
        "sample_id": clean_record["sample_id"],
        "shared_manifest_sha256": manifest_hash,
        "shared_grid_sha256": grid_digest,
        "component_graph_digest": graph.digest,
        "semantic_map_sha256": semantic_digest,
        "validity_map_sha256": validity_digest,
    })
    metadata = {
        "adapter_version": ADAPTER_VERSION,
        "input_schema_version": ADAPTER_INPUT_SCHEMA_VERSION,
        "output_schema_version": ADAPTER_OUTPUT_SCHEMA_VERSION,
        "sample_id": clean_record["sample_id"],
        "shared_manifest_sha256": manifest_hash,
        "partition_sha256": partition_digest,
        "central_image_sha256": image_digest,
        "adapter_config_sha256": spec_hash,
        "shared_grid_sha256": grid_digest,
        "component_graph_digest": graph.digest,
        "assignments": trace,
        "assignment_reasons": dict(sorted(assignment_reasons.items())),
        "unresolved_reasons": unresolved,
        "role_reasons": dict(sorted(role_reasons.items())),
        "void_reasons": dict(sorted(component_reasons.items())),
        "coverage": float(validity.mean()),
        "semantic_map_sha256": semantic_digest,
        "validity_map_sha256": validity_digest,
        "scientific_result_sha256": scientific_result_hash,
        "intensity_resolution": "disabled",
        "orientation_resolution": "disabled",
        "region_splitting": False,
    }
    # Force a final JSON-serializability check before returning a frozen result.
    metadata["metadata_sha256"] = canonical_metadata_hash(adapter_metadata_payload(metadata))
    return AdapterResult(semantic_map=semantic, validity_map=validity, metadata=metadata)
