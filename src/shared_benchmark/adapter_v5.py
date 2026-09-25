"""Regional-conflict resolver for ``cardiac_adapter_v5``.

v5 preserves v4's component-only boundary but fixes two audited failure modes:
absence from another near-best anatomy hypothesis no longer erases a local
foreground label, and RV may be a connected group of MYO-adjacent fragments.
"""
from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import Any, Mapping

import numpy as np

from .adapter_v4 import _LVMYOCandidate, _by_index, _connected_group, _group_perimeter, _lv_myo_candidates
from .provenance import sha256_json
from .region_graph import RegionGraph, build_region_graph
from .semantic_contract import BG, LV, MYO, RV, VOID, AdapterResult, adapter_metadata_payload, array_hash, canonical_metadata_hash, semantic_artifact_stage
from .spatial import grid_hash


def _rv_group_choice(graph: RegionGraph, myo_index: int, excluded: set[int], solver: Mapping[str, Any]) -> tuple[tuple[int, ...], dict[str, Any]]:
    by_index = _by_index(graph)
    myo_adjacency = dict(by_index[myo_index].adjacency)
    neighbors = tuple(index for index, _ in sorted(
        ((index, length) for index, length in myo_adjacency.items() if index not in excluded and not by_index[index].border_contact),
        key=lambda row: (-row[1], row[0]),
    )[:int(solver["max_neighbor_components"])])
    options: list[tuple[float, tuple[int, ...], int]] = []
    for size in range(1, min(int(solver["max_rv_group_components"]), len(neighbors)) + 1):
        for members in combinations(neighbors, size):
            if not _connected_group(members, by_index):
                continue
            group = tuple(by_index[index] for index in members)
            perimeter = _group_perimeter(group, graph.shape)
            contact = sum(myo_adjacency[index] for index in members)
            area_fraction = sum(component.area for component in group) / float(graph.shape[0] * graph.shape[1])
            score = contact / float(perimeter) + 0.10 * sqrt(area_fraction) if perimeter else 0.0
            options.append((score, members, contact))
    options.sort(key=lambda row: (-row[0], row[1]))
    trace = [{"component_indices": list(members), "score": score, "adjacency": contact} for score, members, contact in options]
    if not options or options[0][0] < float(solver["min_rv_score"]):
        return tuple(), {"status": "unsupported_rv", "candidates": trace}
    if len(options) > 1 and options[0][0] - options[1][0] < float(solver["rv_margin"]):
        return tuple(), {"status": "ambiguous_rv", "candidates": trace}
    return options[0][1], {"status": "resolved", "score": options[0][0], "adjacency": options[0][2], "candidates": trace}


def _labels(graph: RegionGraph, candidate: _LVMYOCandidate, solver: Mapping[str, Any]) -> tuple[dict[int, int], dict[str, Any]]:
    labels = {component.index: BG for component in graph.components if component.border_contact}
    labels[candidate.outer] = MYO
    labels.update({index: LV for index in candidate.inner})
    rv, trace = _rv_group_choice(graph, candidate.outer, set(labels), solver)
    labels.update({index: RV for index in rv})
    return labels, trace


def _retain_nonconflicting_roles(selected: Mapping[int, int], alternatives: list[Mapping[int, int]]) -> dict[int, int]:
    """Keep a chosen local role unless a close hypothesis contradicts it.

    A component absent from another hypothesis carries no label evidence.  It
    must not be treated as a conflicting role, which was the v4 consensus bug.
    """
    return {
        index: label for index, label in selected.items()
        if label == BG or not any(other.get(index) not in (None, label) for other in alternatives)
    }


def adapt_partition_v5(record: Mapping[str, Any], partition: np.ndarray, central_image: np.ndarray, *, spec: Mapping[str, Any], spec_hash: str, clean_record: Mapping[str, Any], partition_value: np.ndarray, image_value: np.ndarray, manifest_hash: str) -> AdapterResult:
    del record, partition, central_image
    graph = build_region_graph(partition_value)
    solver = spec["solver"]
    candidates = _lv_myo_candidates(graph, solver)
    eligible = [item for item in candidates if item.score >= float(solver["min_lv_myo_score"])]
    best = eligible[0] if eligible else None
    near_best = [] if best is None else [item for item in eligible if best.score - item.score < float(solver["hypothesis_margin"])]
    labels: dict[int, int] = {}
    conflicting_indices: set[int] = set()
    rv_trace: dict[str, Any] = {"status": "unsupported_rv", "candidates": []}
    if best is not None:
        labels, rv_trace = _labels(graph, best, solver)
        alternatives = [_labels(graph, item, solver)[0] for item in near_best]
        # Only contradictory role assignments void a component.  Another
        # hypothesis omitting a distant component is not evidence that the
        # selected local MYO/LV group is wrong.
        conflicting_indices = {
            index for index, label in labels.items()
            if label != BG and any(other.get(index) not in (None, label) for other in alternatives)
        }
        labels = _retain_nonconflicting_roles(labels, alternatives)
    by_index = _by_index(graph)
    semantic = np.full(partition_value.shape, VOID, dtype=np.uint8)
    for index, label in labels.items():
        for row, col in by_index[index].pixels:
            semantic[row, col] = label
    validity = np.asarray(semantic != VOID, dtype=bool)
    reasons: dict[str, str] = {}
    void_reasons: dict[str, str] = {}
    for component in graph.components:
        label = labels.get(component.index)
        if label == BG:
            reasons[component.key] = "border_background_consensus"
        elif label == MYO:
            reasons[component.key] = best.source if best else "unassigned_component"
        elif label == LV:
            reasons[component.key] = "lv_group_" + (best.source if best else "unassigned")
        elif label == RV:
            reasons[component.key] = "highest_margin_adjacent_rv_group"
        elif best is None:
            void_reasons[component.key] = "no_confident_lv_myo_hypothesis"
        elif component.index in conflicting_indices:
            void_reasons[component.key] = "hypothesis_role_conflict"
        elif rv_trace["status"] == "ambiguous_rv" and any(component.index in row["component_indices"] for row in rv_trace["candidates"]):
            void_reasons[component.key] = "ambiguous_rv"
        else:
            void_reasons[component.key] = "unassigned_component"
    role_reasons = {
        "BG": "resolved" if any(label == BG for label in labels.values()) else "unsupported_background",
        "MYO": "resolved" if any(label == MYO for label in labels.values()) else "no_confident_lv_myo_hypothesis",
        "LV": "resolved" if any(label == LV for label in labels.values()) else "no_confident_lv_myo_hypothesis",
        "RV": rv_trace["status"],
    }
    assignments = [{
        "component_key": component.key, "source_cluster_id": component.source_id,
        "semantic": {BG: "BG", RV: "RV", MYO: "MYO", LV: "LV"}.get(labels.get(component.index), "VOID"),
        "semantic_id": labels.get(component.index, VOID),
        "reason": reasons.get(component.key, void_reasons.get(component.key, "unassigned_component")),
    } for component in graph.components]
    unresolved = {role: reason for role, reason in role_reasons.items() if reason != "resolved"}
    unresolved.update({f"component:{key}": value for key, value in void_reasons.items()})
    semantic_digest, validity_digest = array_hash(semantic), array_hash(validity)
    grid_digest = grid_hash(clean_record["shared_grid"])
    scientific_result_hash = sha256_json({"adapter_version": str(spec["adapter_version"]), "adapter_config_sha256": spec_hash, "sample_id": clean_record["sample_id"], "shared_manifest_sha256": manifest_hash, "shared_grid_sha256": grid_digest, "component_graph_digest": graph.digest, "semantic_map_sha256": semantic_digest, "validity_map_sha256": validity_digest})
    metadata: dict[str, Any] = {
        "adapter_version": str(spec["adapter_version"]), "input_schema_version": str(spec["input_schema_version"]), "output_schema_version": str(spec["output_schema_version"]), "sample_id": clean_record["sample_id"], "shared_manifest_sha256": manifest_hash, "partition_sha256": array_hash(partition_value), "central_image_sha256": array_hash(image_value), "adapter_config_sha256": spec_hash, "shared_grid_sha256": grid_digest, "component_graph_digest": graph.digest, "assignments": assignments, "assignment_reasons": dict(sorted(reasons.items())), "void_reasons": dict(sorted(void_reasons.items())), "role_reasons": dict(sorted(role_reasons.items())), "unresolved_reasons": dict(sorted(unresolved.items())), "coverage": float(validity.mean()), "semantic_map_sha256": semantic_digest, "validity_map_sha256": validity_digest, "scientific_result_sha256": scientific_result_hash, "intensity_resolution": "disabled", "orientation_resolution": "disabled", "region_splitting": False, "semantic_artifact_stage": semantic_artifact_stage(str(spec["adapter_version"])),
        "solver": {
            "candidate_count": len(candidates), "eligible_count": len(eligible),
            "near_optimal_count": len(near_best),
            "search_truncated": len(candidates) >= int(solver["max_hypotheses"]),
            "best_score": None if best is None else best.score,
            "conflict_policy": "regional_role_only", "rv": rv_trace,
            "hypotheses": [{
                "myo_component": by_index[item.outer].key,
                "lv_components": [by_index[index].key for index in item.inner],
                "enclosure_support": item.enclosure_support, "score": item.score,
                "source": item.source, "near_optimal": item in near_best,
            } for item in candidates],
        },
    }
    metadata["metadata_sha256"] = canonical_metadata_hash(adapter_metadata_payload(metadata))
    return AdapterResult(semantic_map=semantic, validity_map=validity, metadata=metadata)
