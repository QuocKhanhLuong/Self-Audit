from __future__ import annotations

import numpy as np
import pytest

try:
    from self_audit.data.label_schema import (
        UNIFIED_LABEL_IDS,
        UNIFIED_LABEL_SCHEMA,
        remap_labels,
        validate_label_mapping,
    )
except ImportError:
    from src.self_audit.data.label_schema import (
        UNIFIED_LABEL_IDS,
        UNIFIED_LABEL_SCHEMA,
        remap_labels,
        validate_label_mapping,
    )


def test_unified_label_schema_is_single_four_class_contract() -> None:
    assert UNIFIED_LABEL_SCHEMA == {
        0: "Background",
        1: "RV",
        2: "MYO",
        3: "LV",
    }
    assert UNIFIED_LABEL_IDS == frozenset({0, 1, 2, 3})


def test_remap_labels_maps_source_ids_without_changing_shape() -> None:
    source = np.asarray([[0, 1, 2], [3, 2, 0]], dtype=np.int16)
    mapped = remap_labels(source, {0: 0, 1: 2, 2: 3, 3: 1})
    np.testing.assert_array_equal(mapped, [[0, 2, 3], [1, 3, 0]])
    assert mapped.shape == source.shape
    assert mapped.dtype == np.int64


def test_remap_labels_rejects_unknown_source_ids() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        remap_labels(np.asarray([[0, 4]], dtype=np.int16), {0: 0, 1: 1})


def test_validate_label_mapping_rejects_non_bijective_or_invalid_targets() -> None:
    with pytest.raises(ValueError, match="integer"):
        validate_label_mapping({"0": 0})
    with pytest.raises(ValueError, match="unified"):
        validate_label_mapping({0: 0, 1: 1, 2: 2, 3: 4})
    with pytest.raises(ValueError, match="duplicate"):
        validate_label_mapping({0: 0, 1: 1, 2: 1, 3: 3})
