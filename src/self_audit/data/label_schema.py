"""Single source of truth for the four-class cardiac segmentation labels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


UNIFIED_LABEL_SCHEMA: dict[int, str] = {
    0: "Background",
    1: "RV",
    2: "MYO",
    3: "LV",
}
UNIFIED_LABEL_IDS = frozenset(UNIFIED_LABEL_SCHEMA)
CLASS_NAMES = tuple(UNIFIED_LABEL_SCHEMA.values())
NUM_CLASSES = len(UNIFIED_LABEL_SCHEMA)


def validate_label_mapping(mapping: Mapping[Any, Any]) -> dict[int, int]:
    """Validate and normalize a dataset-specific source-to-unified mapping.

    A mapping may contain only the source IDs observed by a dataset, but every
    target must be one of the four unified IDs and targets must be unique. The
    latter catches accidental many-to-one class collapse before training.
    """

    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("label mapping must be a non-empty mapping")
    normalized: dict[int, int] = {}
    for raw_source, raw_target in mapping.items():
        if (
            isinstance(raw_source, bool)
            or isinstance(raw_target, bool)
            or not isinstance(raw_source, (int, np.integer))
            or not isinstance(raw_target, (int, np.integer))
        ):
            raise ValueError("label mapping source and target IDs must be integer values")
        source = int(raw_source)
        target = int(raw_target)
        if target not in UNIFIED_LABEL_IDS:
            raise ValueError(
                f"label mapping target {target} is not a valid unified label"
            )
        normalized[source] = target
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("label mapping contains duplicate unified targets")
    return normalized


def remap_labels(mask: np.ndarray, mapping: Mapping[Any, Any]) -> np.ndarray:
    """Map a source mask to the unified schema without changing its geometry."""

    array = np.asarray(mask)
    normalized = validate_label_mapping(mapping)
    observed = {int(value) for value in np.unique(array)}
    unknown = sorted(observed.difference(normalized))
    if unknown:
        raise ValueError(
            f"mask contains unknown label values {unknown}; mapping has "
            f"{sorted(normalized)}"
        )
    result = np.empty(array.shape, dtype=np.int64)
    for source, target in normalized.items():
        result[array == source] = target
    return result


def validate_unified_labels(mask: np.ndarray) -> None:
    """Raise when a post-remap mask contains an invalid class ID."""

    observed = {int(value) for value in np.unique(np.asarray(mask))}
    invalid = sorted(observed.difference(UNIFIED_LABEL_IDS))
    if invalid:
        raise ValueError(f"mask contains labels outside the unified schema: {invalid}")
