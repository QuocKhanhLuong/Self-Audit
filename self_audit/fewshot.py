"""Metadata-only construction of nested dense TRAIN few-shot manifests.

The helper deliberately does not instantiate :class:`TrainingData`: doing so
would open every mask in the source labeled pool.  A derived manifest keeps
only the selected patient records with masks and moves the remaining patients
to image-only U records.  The regular loader then validates and opens masks
only for the selected budget.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .data import CLASS_MAPPING, digest, geometry_map, label_subset


_TRAIN_SCHEMA = "self_audit.data.v2"
_SELECTION_DESCRIPTION = "sha256(seed:patient_id) over source labeled IDs"


def _absolute_payload_path(source_path: Path, value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    return str((source_path.parent / value).resolve()) if not Path(value).is_absolute() else str(Path(value).resolve())


def _image_record(record: Any, source_path: Path, *, labeled: bool) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("TRAIN records must be objects")
    required = {"id", "image", "image_sha256"}
    expected = required | ({"masks"} if labeled else set())
    if set(record) != expected:
        kind = "labeled" if labeled else "unlabeled"
        raise ValueError(f"invalid {kind} TRAIN record keys")
    patient_id = record["id"]
    if not isinstance(patient_id, str) or not patient_id:
        raise ValueError("TRAIN patient IDs must be non-empty strings")
    if not isinstance(record["image_sha256"], str) or not record["image_sha256"]:
        raise ValueError("TRAIN image_sha256 must be a non-empty string")
    result = deepcopy(record)
    result["image"] = _absolute_payload_path(source_path, record["image"], field="image")
    if not labeled:
        return result
    masks = record["masks"]
    if not isinstance(masks, dict) or not masks:
        raise ValueError("labeled TRAIN records must contain a non-empty masks map")
    for frame, mask in masks.items():
        if not isinstance(frame, str) or not frame or not isinstance(mask, dict):
            raise ValueError("invalid labeled mask metadata")
        if set(mask) != {"path", "sha256"}:
            raise ValueError("labeled mask metadata must contain only path and sha256")
        if not isinstance(mask["sha256"], str) or not mask["sha256"]:
            raise ValueError("labeled mask sha256 must be a non-empty string")
        result["masks"][frame]["path"] = _absolute_payload_path(
            source_path, mask["path"], field="mask path"
        )
    return result


def _validate_source(manifest_path: Path, manifest: dict[str, Any], *, expected_sha256: str,
                     dataset: str, synthetic: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if digest(manifest_path) != expected_sha256:
        raise ValueError("few-shot source manifest hash mismatch")
    if manifest.get("schema") != _TRAIN_SCHEMA or manifest.get("role") != "training":
        raise ValueError("few-shot source must be a v2 training manifest")
    if bool(manifest.get("synthetic", False)) != synthetic:
        raise ValueError("few-shot source synthetic/patient identity mismatch")
    if manifest.get("dataset") != dataset:
        raise ValueError("few-shot source dataset identity mismatch")
    if manifest.get("class_mapping") != CLASS_MAPPING:
        raise ValueError("few-shot source class_mapping must be the explicit native ontology")
    labeled = manifest.get("labeled")
    unlabeled = manifest.get("unlabeled")
    if not isinstance(labeled, list) or not isinstance(unlabeled, list):
        raise ValueError("few-shot source must contain labeled and unlabeled lists")
    labeled_ids = [record.get("id") if isinstance(record, dict) else None for record in labeled]
    unlabeled_ids = [record.get("id") if isinstance(record, dict) else None for record in unlabeled]
    all_ids = labeled_ids + unlabeled_ids
    if any(not isinstance(patient_id, str) or not patient_id for patient_id in all_ids):
        raise ValueError("TRAIN patient IDs must be non-empty strings")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("TRAIN labeled and unlabeled patient IDs must be unique")
    evaluation_ids = manifest.get("evaluation_ids", [])
    if not isinstance(evaluation_ids, list) or set(all_ids) & set(evaluation_ids):
        raise ValueError("TRAIN and evaluation patient IDs overlap")
    if manifest.get("label_budget") != len(labeled):
        raise ValueError("source label_budget must equal the labeled TRAIN pool")
    if len(labeled) < 2:
        raise ValueError("few-shot source needs at least two labeled TRAIN patients")
    # Validate geometry and record shape only. geometry_map validates the JSON
    # contract but never opens an image or mask payload.
    geometry_map(manifest)
    source_labeled = [_image_record(record, manifest_path, labeled=True) for record in labeled]
    source_unlabeled = [_image_record(record, manifest_path, labeled=False) for record in unlabeled]
    return source_labeled, source_unlabeled


def build_subset_manifest(source_manifest: str | Path, expected_sha256: str, *, k: int,
                          seed: int = 42, dataset: str, synthetic: bool = False) -> dict[str, Any]:
    """Build a nested ``k``-patient labeled TRAIN manifest without reading masks.

    ``k`` is a dense patient count, and all annotated frames for selected
    patients remain labeled.  The source labeled pool is ranked once by
    ``sha256(f"{seed}:{patient_id}")``; therefore calls for 2, 5, 8 and 10
    produce nested cohorts when they share the source manifest and seed.
    """

    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("few-shot k must be an integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("few-shot seed must be a non-negative integer")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("few-shot dataset must be a non-empty string")
    manifest_path = Path(source_manifest).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"few-shot source manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("few-shot source manifest must contain one JSON object")
    source_labeled, source_unlabeled = _validate_source(
        manifest_path, manifest, expected_sha256=expected_sha256,
        dataset=dataset, synthetic=synthetic,
    )
    if k < 2 or k > len(source_labeled):
        raise ValueError(f"few-shot k must satisfy 2 <= k <= {len(source_labeled)}")
    eligible_ids = sorted(record["id"] for record in source_labeled)
    selected_ids = label_subset(eligible_ids, k, seed)
    selected_set = set(selected_ids)
    selected = [record for record in source_labeled if record["id"] in selected_set]
    downgraded = [
        {"id": record["id"], "image": record["image"], "image_sha256": record["image_sha256"]}
        for record in source_labeled if record["id"] not in selected_set
    ]
    # Preserve source list order for deterministic manifests; the selected IDs
    # are nested because selection itself is based on one fixed ranking.
    derived = deepcopy(manifest)
    derived["labeled"] = selected
    derived["unlabeled"] = downgraded + source_unlabeled
    derived["label_budget"] = k
    derived["label_seed"] = seed
    derived["fewshot"] = {
        "source_manifest": str(manifest_path),
        "source_sha256": expected_sha256,
        "selection": _SELECTION_DESCRIPTION,
        "eligible_labeled_ids": eligible_ids,
        "selected_ids": sorted(selected_ids),
        "unit": "dense_labeled_train_patients",
        "seed": seed,
        "k": k,
    }
    # Keep only source-compatible top-level data; no masks are ever introduced
    # for a previously image-only U patient.
    return derived


__all__ = ["build_subset_manifest"]
