from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from shared_benchmark.manifest import build_shared_manifest, validate_manifest
from shared_benchmark.self_audit_protocol import discover_self_audit_acdc
from shared_benchmark.spatial import (
    SELF_AUDIT_HISTORICAL_224_NORMALIZATION_VERSION,
    SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION,
    SELF_AUDIT_NORMALIZATION_VERSION,
    SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
    grid_hash,
    load_self_audit_historical_224_grid_spec,
    load_self_audit_grid_spec,
    normalize_self_audit_volume,
    read_self_audit_historical_224_context_stack,
    read_self_audit_context_stack,
    resize_values_to_grid,
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


def test_historical_224_baseline_grid_is_distinct_from_current_self_audit_grid():
    from shared_benchmark.semantic_contract import FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256

    grid = load_self_audit_historical_224_grid_spec(ROOT)
    assert grid["version"] == SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION
    assert grid["target_hw"] == [224, 224]
    assert grid["whole_fov"] is True
    assert grid["crop"] is None
    assert grid["forward_values"] == "historical_preprocess_224_no_post_loader_resize"
    assert grid_hash(grid) == FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256
    assert grid["config_provenance"]["network_resize_after_loader"] == "none_baseline_only"


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


def test_historical_224_image_branch_then_loader_normalization_has_no_post_resize(tmp_path: Path):
    from skimage.transform import resize

    image_root = tmp_path / "images"
    image_path = image_root / "training" / "patient001" / "patient001_frame01.nii"
    image_path.parent.mkdir(parents=True)
    source_hwz = np.linspace(-3.0, 9.0, 4 * 5 * 3, dtype=np.float64).reshape(4, 5, 3)
    nib.save(nib.Nifti1Image(source_hwz, np.eye(4)), str(image_path))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({
        "schema_version": "self_audit.acdc.frame_selection.v1", "seed": 42,
        "records": [{
            "patient_id": "patient001", "split": "train",
            "relative_path": "training/patient001/patient001_frame01.nii",
            "frame_index": 1, "frame_label": "ED",
        }],
    }), encoding="utf-8")
    source = discover_self_audit_acdc(image_root, selection_path)
    grid = load_self_audit_historical_224_grid_spec(ROOT)
    manifest = build_shared_manifest(source, grid, fixture=False, scientific=True)
    record = manifest["records"][1]
    observed = read_self_audit_historical_224_context_stack(record, source_root=image_root)

    # Independent spelling of the two historical image transforms.  This
    # verifies ordering, skimage arguments, float32 persisted-array behavior,
    # and that the baseline contract does not append a 224->256 resize.
    low, high = np.percentile(source_hwz, (0.5, 99.5))
    preprocessed = np.clip(source_hwz, low, high)
    preprocessed = (preprocessed - preprocessed.mean()) / preprocessed.std()
    resized_hwz = np.empty((224, 224, 3), dtype=np.float32)
    for z in range(3):
        resized_hwz[:, :, z] = resize(
            preprocessed[:, :, z], (224, 224), order=1, preserve_range=True,
            anti_aliasing=True, mode="reflect",
        )
    expected = normalize_self_audit_volume(np.moveaxis(resized_hwz, 2, 0))[[0, 1, 2]]
    assert observed.shape == (3, 224, 224)
    assert np.array_equal(observed, expected)
    assert np.array_equal(resize_values_to_grid(__import__("torch").from_numpy(observed), grid).numpy(), observed)
    assert SELF_AUDIT_HISTORICAL_224_NORMALIZATION_VERSION.endswith("zscore.v1")


def test_self_audit_receipt_rejects_test_membership(tmp_path: Path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "schema_version": "self_audit.acdc.frame_selection.v1", "seed": 42,
        "records": [{"patient_id": "patient001", "split": "test", "relative_path": "x.nii", "frame_index": 1, "frame_label": "ED"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="no test cohort"):
        discover_self_audit_acdc(tmp_path, selection)
