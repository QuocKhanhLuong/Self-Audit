from __future__ import annotations

from pathlib import Path

import torch

from shared_benchmark.spatial import (
    SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION,
    build_grid_spec,
    load_pinned_grid_spec,
    load_self_audit_compat_224_grid_spec,
    resize_values_to_grid,
)


def test_shared_grid_is_whole_fov_and_uses_masked_area_primitive():
    grid = build_grid_spec((5, 7), config_provenance={"source": "fixture"})
    values = torch.arange(3 * 9 * 13, dtype=torch.float32).reshape(3, 9, 13)
    resized = resize_values_to_grid(values, grid)
    assert tuple(resized.shape) == (3, 5, 7)
    assert grid["whole_fov"] is True
    assert grid["forward_values"] == "masked_area_normalized_convolution"
    assert grid["forward_masks"] == "nearest_exact"


def test_pinned_freemask_configs_bind_the_scientific_224_grid():
    repo_root = Path(__file__).resolve().parents[2]
    grid = load_pinned_grid_spec(repo_root)
    assert grid["target_hw"] == [224, 224]
    assert grid["config_provenance"]["config_files"] == [
        "configs/maskfree_acdc_150.yaml", "configs/maskfree_mnms_150.yaml",
    ]


def test_self_audit_compat_grid_binds_224_bilinear_contract():
    repo_root = Path(__file__).resolve().parents[2]
    grid = load_self_audit_compat_224_grid_spec(repo_root)
    assert grid["version"] == SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION
    assert grid["target_hw"] == [224, 224]
    assert grid["whole_fov"] is True
    assert grid["crop"] is None
    assert grid["forward_values"] == "bilinear_align_corners_false"
    assert grid["config_provenance"]["self_audit_protocol"]["preprocessing"]["clipping_min"] == 0.5
    assert grid["config_provenance"]["self_audit_protocol"]["preprocessing"]["clipping_max"] == 99.5
    assert grid["config_provenance"]["compat_grid"]["image_size"] == 224
