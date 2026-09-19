from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from shared_benchmark.manifest import build_shared_manifest, validate_manifest
from shared_benchmark.self_audit_protocol import discover_self_audit_acdc
from shared_benchmark.spatial import (
    SELF_AUDIT_NORMALIZATION_VERSION,
    SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
    load_self_audit_grid_spec,
    normalize_self_audit_volume,
    read_self_audit_context_stack,
)


ROOT = Path(__file__).resolve().parents[2]


def test_self_audit_grid_is_256_whole_fov_and_loader_bilinear():
    grid = load_self_audit_grid_spec(ROOT)
    assert grid["version"] == SELF_AUDIT_SPATIAL_CONTRACT_VERSION
    assert grid["target_hw"] == [256, 256]
    assert grid["whole_fov"] is True
    assert grid["crop"] is None
    assert grid["forward_values"] == "bilinear_align_corners_false"
    assert grid["config_provenance"]["preprocessing"]["clipping_min"] == 0.5
    assert grid["config_provenance"]["preprocessing"]["clipping_max"] == 99.5


def test_image_only_receipt_drives_ed_es_all_slice_inventory(tmp_path: Path):
    image_root = tmp_path / "images"
    image_path = image_root / "training" / "patient001" / "patient001_frame01.nii"
    image_path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.arange(4 * 5 * 3, dtype=np.int16).reshape(4, 5, 3), np.eye(4)), str(image_path))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({
        "schema_version": "self_audit.acdc.frame_selection.v1",
        "seed": 42,
        "records": [{
            "patient_id": "patient001", "split": "train",
            "relative_path": "training/patient001/patient001_frame01.nii",
            "frame_index": 1, "frame_label": "ED",
        }],
    }), encoding="utf-8")
    source = discover_self_audit_acdc(image_root, selection_path)
    assert len(source["records"]) == 3
    assert {row["slice_index"] for row in source["records"]} == {0, 1, 2}
    assert source["split_provenance"]["test_patients"] == 0
    assert source["discovery_contract"]["mask_files_read"] == 0
    grid = load_self_audit_grid_spec(ROOT)
    manifest = build_shared_manifest(source, grid, fixture=False, scientific=True)
    validate_manifest(manifest)
    assert all(row["split"] == "train" for row in manifest["records"])
    record = manifest["records"][1]
    stack = read_self_audit_context_stack(record, source_root=image_root)
    expected_volume = normalize_self_audit_volume(np.moveaxis(np.arange(4 * 5 * 3, dtype=np.int16).reshape(4, 5, 3), 2, 0))
    assert np.array_equal(stack, expected_volume[[0, 1, 2]])
    assert SELF_AUDIT_NORMALIZATION_VERSION == "self_audit.volume_percentile_clip_0p5_99p5_zscore.v1"


def test_self_audit_receipt_rejects_test_membership(tmp_path: Path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "schema_version": "self_audit.acdc.frame_selection.v1", "seed": 42,
        "records": [{"patient_id": "patient001", "split": "test", "relative_path": "x.nii", "frame_index": 1, "frame_label": "ED"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="no test cohort"):
        discover_self_audit_acdc(tmp_path, selection)
