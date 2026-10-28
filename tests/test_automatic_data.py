"""Targeted acceptance tests for automatic data loading, complete support extraction, and feature caching.

Covers:
- Complete multislice (Z > 1) and multiframe (T > 1) support volume extraction.
- Unequal native spatial dimensions non-multiples of 32 (e.g. H=35, W=41 and H=37, W=45).
- Rejection of malformed frame IDs, non-canonical aliases, and out-of-bounds indices BEFORE mask I/O.
- Explicit all-ignored support volume policy (rejects frames where all voxels are 255 unless allowed).
- Native extraction per shape/chunk followed by padding with explicit boolean geometric validity masks.
- Exact end-to-end forward vs head-on-cache parity for NativeStudent with fixed SemanticWindow across all shapes.
- Closed failure on nonfinite logits/features (guarding against IEEE 754 NaN > tolerance bug).
- Bounded native voxels and padded cache bytes rejection BEFORE large allocations.
- Conservative controller workspace estimation for m<=4, d=32, c=4.
- Invalidation on model weights, non-tensor window config, training mode, and mutated tensors
  (rec.images, rec.labels, cache.features, cache.valid_mask, cache.labels).
- Evaluation-mode frozen feature state and RNG preservation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from self_audit.automatic_data import (
    DEFAULT_MAX_CACHE_BYTES,
    DEFAULT_MAX_CONTROLLER_WORKSPACE_BYTES,
    SupportFeatureCache,
    SupportVolumeRecord,
    compute_model_feature_fingerprint,
    compute_support_feature_cache,
    estimate_controller_workspace_bytes,
    estimate_padded_cache_bytes,
    extract_complete_support_volumes,
    get_model_feature_config,
    hash_tensor,
)
from self_audit.data import TrainingData, write_json
from self_audit.model import NativeStudent, TinyEncoder, digest
from self_audit.window import validate_window_config


def create_variable_shape_fixture(
    folder: Path,
    seed: int = 42,
) -> tuple[Path, str]:
    """Create a synthetic training dataset with multislice, multiframe, and unequal nonmultiples of 32."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Patient configs: unequal native H/W that are NOT multiples of 32
    # synthetic_000: T=3 frames, Z=3 slices, H=35, W=41
    # synthetic_001: T=2 frames, Z=4 slices, H=37, W=45
    # synthetic_002 (unlabeled): T=2, Z=3, H=35, W=41
    patient_specs = [
        {"id": "synthetic_000", "T": 3, "Z": 3, "H": 35, "W": 41, "labeled_frames": [0, 1]},
        {"id": "synthetic_001", "T": 2, "Z": 4, "H": 37, "W": 45, "labeled_frames": [0]},
        {"id": "synthetic_002", "T": 2, "Z": 3, "H": 35, "W": 41, "labeled_frames": []},
    ]

    labeled_manifest = []
    unlabeled_manifest = []

    for spec in patient_specs:
        pid = spec["id"]
        T, Z, H, W = spec["T"], spec["Z"], spec["H"], spec["W"]
        # Normalized cine image: [T, Z, H, W]
        cine = rng.uniform(0.1, 0.9, size=(T, Z, H, W)).astype(np.float32)
        img_path = folder / f"{pid}_cine.npy"
        np.save(img_path, cine)
        img_sha = digest(img_path)

        record: dict[str, Any] = {
            "id": pid,
            "image": img_path.name,
            "image_sha256": img_sha,
        }

        if spec["labeled_frames"]:
            masks_dict = {}
            for fid in spec["labeled_frames"]:
                # Multislice mask with valid classes {0, 1, 2, 3} and some ignored 255
                mask = np.zeros((Z, H, W), dtype=np.uint8)
                mask[:, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1
                mask[:, H // 3 : 2 * H // 3, W // 3 : 2 * W // 3] = 2
                mask[0, 0, 0] = 255  # ignored voxel
                mask_path = folder / f"{pid}_frame_{fid}_mask.npy"
                np.save(mask_path, mask)
                masks_dict[str(fid)] = {
                    "path": mask_path.name,
                    "sha256": digest(mask_path),
                }
            record["masks"] = masks_dict
            labeled_manifest.append(record)
        else:
            unlabeled_manifest.append(record)

    manifest_path = folder / "train.json"
    manifest_data = {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": True,
        "label_budget": len(labeled_manifest),
        "labeled": labeled_manifest,
        "unlabeled": unlabeled_manifest,
        "evaluation_ids": [],
    }
    write_json(manifest_path, manifest_data)
    return manifest_path, digest(manifest_path)


def test_multislice_multiframe_complete_support_volumes(tmp_path):
    """Verify complete volume retention (all Z slices) and multi-frame support loading."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)

    records = extract_complete_support_volumes(data, ["synthetic_000", "synthetic_001"])
    # synthetic_000 has 2 frames (0, 1), synthetic_001 has 1 frame (0) -> 3 records total
    assert len(records) == 3

    # Check synthetic_000 frame 0
    rec0 = records[0]
    assert rec0.patient_id == "synthetic_000"
    assert rec0.frame_id == 0
    assert rec0.num_slices == 3
    assert rec0.height == 35
    assert rec0.width == 41
    assert rec0.images.shape == (3, 3, 35, 41)
    assert rec0.labels.shape == (3, 35, 41)

    # Check synthetic_000 frame 1
    rec1 = records[1]
    assert rec1.patient_id == "synthetic_000"
    assert rec1.frame_id == 1
    assert rec1.num_slices == 3
    assert rec1.height == 35
    assert rec1.width == 41

    # Check synthetic_001 frame 0
    rec2 = records[2]
    assert rec2.patient_id == "synthetic_001"
    assert rec2.frame_id == 0
    assert rec2.num_slices == 4
    assert rec2.height == 37
    assert rec2.width == 45

    # Check label accounting
    assert "synthetic_000" in data.label_reads
    assert "synthetic_001" in data.label_reads
    assert "synthetic_002" not in data.label_reads  # Unlabeled never read


def test_malformed_frame_ids_and_bounds_reject_before_mask_io(tmp_path):
    """Reject non-integer frame aliases, negative indices, and out-of-bounds frames BEFORE mask I/O."""
    data_dir = tmp_path / "data_malformed"
    m_path, m_sha = create_variable_shape_fixture(data_dir)
    data = TrainingData(m_path, m_sha, synthetic=True)

    # We test that extract_complete_support_volumes rejects malformed frame metadata BEFORE mask I/O
    # Even if mask path points to an invalid/nonexistent file
    bad_specs = [
        ("ED", "non-integer frame ID"),
        ("01", "non-canonical frame ID alias"),
        ("-1", "out of bounds for temporal length"),
        ("10", "out of bounds for temporal length"),
    ]

    for bad_key, pattern in bad_specs:
        # Inject bad frame into patient synthetic_000's masks
        data.labeled[0]["masks"] = {
            bad_key: {
                "path": "non_existent_mask_file_never_read.npy",
                "sha256": "0" * 64,
            }
        }
        with pytest.raises(ValueError, match=pattern):
            extract_complete_support_volumes(data, ["synthetic_000"])


def test_all_ignored_support_policy_explicit(tmp_path):
    """Enforce explicit policy rejecting support volumes where all voxels are 255."""
    data_dir = tmp_path / "data_all_ignored"
    m_path, m_sha = create_variable_shape_fixture(data_dir)
    data = TrainingData(m_path, m_sha, synthetic=True)

    # Overwrite mask for synthetic_000 frame 0 with all 255
    r0 = data.labeled[0]
    m0_info = r0["masks"]["0"]
    m0_resolved = data.resolve(m0_info["path"])
    mask_arr = np.load(m0_resolved)
    all_255 = np.full_like(mask_arr, 255)
    np.save(m0_resolved, all_255)
    new_sha = digest(m0_resolved)
    m0_info["sha256"] = new_sha

    # Clear cache in TrainingData so it loads the modified array
    data.cache.clear()

    # Default: fails closed on all-ignored frame
    with pytest.raises(ValueError, match="rejected by explicit all-ignored support policy"):
        extract_complete_support_volumes(data, ["synthetic_000"])

    # Opt-in override: explicitly allowed if requested
    records = extract_complete_support_volumes(data, ["synthetic_000"], allow_all_ignored=True)
    assert len(records) >= 1
    # First record corresponds to frame 0 which is entirely 255
    assert (records[0].labels == 255).all()


def test_native_student_fixed_window_parity_unequal_nonmultiples32(tmp_path):
    """Verify exact parity between NativeStudent forward and head-on-cache for unequal nonmultiples of 32."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)

    # Instantiate NativeStudent with TinyEncoder and fixed SemanticWindow
    window_cfg = validate_window_config({
        "mode": "fixed",
        "dim": 32,
        "radius_min": 0.5,
        "radius_max": 4.0,
        "offset_limit": 2.0,
        "initial_radius": 1.0,
        "gamma_init": 0.001,
    })
    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32), window=window_cfg)
    model.eval()

    records = extract_complete_support_volumes(data, ["synthetic_000", "synthetic_001"])
    cache = compute_support_feature_cache(model, records, device="cpu", max_extract_batch=2)

    # Padded spatial grid matches maximum H and W across all volumes
    # synthetic_000 is 35x41, synthetic_001 is 37x45 -> max_H=37, max_W=45
    assert cache.features.shape[-2:] == (37, 45)
    assert cache.labels.shape[-2:] == (37, 45)
    assert cache.valid_mask.shape[-2:] == (37, 45)

    # Total slices = 3 + 3 + 4 = 10 slices
    assert cache.features.shape[0] == 10

    # Probe samples recorded for BOTH distinct shapes (35, 41) and (37, 45)
    probe_shapes = {(p["H"], p["W"]) for p in cache.probe_samples}
    assert (35, 41) in probe_shapes
    assert (37, 45) in probe_shapes

    # Parity check passes across all probe samples
    max_diff = cache.verify_parity(model, tolerance=1e-5)
    assert max_diff < 1e-5


def test_padded_cache_bytes_and_workspace_bounds_rejection_before_allocation(tmp_path):
    """Enforce both native voxel and padded byte bounds rejection BEFORE large allocations."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)

    # 1. Native voxels limit rejection in extract_complete_support_volumes
    with pytest.raises(ValueError, match="exceed declared limit"):
        extract_complete_support_volumes(data, ["synthetic_000"], max_support_voxels=50)

    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32))
    records = extract_complete_support_volumes(data, ["synthetic_000", "synthetic_001"])

    # 2. Padded cache bytes limit rejection BEFORE allocation in compute_support_feature_cache
    b_total = sum(r.num_slices for r in records)
    max_h = max(r.height for r in records)
    max_w = max(r.width for r in records)
    expected_cache_bytes = estimate_padded_cache_bytes(b_total, max_h, max_w)
    assert expected_cache_bytes > 0

    with pytest.raises(ValueError, match="exceed declared max_cache_bytes"):
        compute_support_feature_cache(
            model,
            records,
            max_cache_bytes=expected_cache_bytes - 1,
        )

    # 2b. Default 2 GiB limit rejection when max_cache_bytes is None on oversized geometry
    class BigMockRecord:
        num_slices = 200
        height = 1000
        width = 1000
        patient_id = "mock_big"
        frame_id = 0

    with pytest.raises(ValueError, match="exceed declared max_cache_bytes"):
        compute_support_feature_cache(
            model,
            [BigMockRecord()],  # type: ignore
            max_cache_bytes=None,
        )

    # 3. Controller workspace estimate limit rejection
    expected_workspace = estimate_controller_workspace_bytes(b_total, max_h, max_w, m=4, d=32, c=4)
    assert expected_workspace > expected_cache_bytes

    with pytest.raises(ValueError, match="controller workspace estimate"):
        compute_support_feature_cache(
            model,
            records,
            max_controller_workspace_bytes=expected_workspace - 1,
        )

    # 4. Successful cache creation includes workspace estimate in fingerprint
    cache = compute_support_feature_cache(model, records)
    assert cache.fingerprint["controller_workspace_estimate_bytes"] == expected_workspace
    assert cache.fingerprint["padded_cache_bytes"] == expected_cache_bytes


def test_cache_invalidation_on_mutated_inputs_and_contents(tmp_path):
    """Verify SupportFeatureCache detects mutations in records, cache features, labels, and valid_mask."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)

    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32))
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)

    # Fresh cache verifies fingerprint cleanly
    assert cache.verify_fingerprint(model) is True

    # 1. Mutate record.images in-place
    with torch.no_grad():
        records[0].images[0, 0, 0, 0].add_(1.0)
    assert records[0].verify_integrity() is False
    assert cache.verify_fingerprint(model) is False
    with torch.no_grad():
        records[0].images[0, 0, 0, 0].sub_(1.0)  # Revert value; version is still mutated
    assert cache.verify_fingerprint(model) is False

    # Refresh cache for next mutation tests
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    assert cache.verify_fingerprint(model) is True

    # 2. Mutate record.labels in-place
    records[0].labels[0, 0, 0] = 3
    assert records[0].verify_integrity() is False
    assert cache.verify_fingerprint(model) is False

    # 3. Mutate cache.features in-place
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    with torch.no_grad():
        cache.features[0, 0, 0, 0].add_(0.5)
    assert cache.verify_fingerprint(model) is False

    # 4. Mutate cache.valid_mask in-place
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    cache.valid_mask[0, 0, 0] = False
    assert cache.verify_fingerprint(model) is False

    # 5. Mutate cache.labels in-place
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    cache.labels[0, 0, 0] = 255
    assert cache.verify_fingerprint(model) is False


def test_cache_invalidation_on_model_weights_and_non_tensor_config(tmp_path):
    """Verify SupportFeatureCache detects model weight changes, non-tensor window config, and training mode."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)

    window_cfg = validate_window_config({
        "mode": "fixed",
        "dim": 32,
        "radius_min": 0.5,
        "radius_max": 4.0,
        "offset_limit": 2.0,
        "initial_radius": 1.0,
        "gamma_init": 0.001,
    })
    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32), window=window_cfg)
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)

    assert cache.verify_fingerprint(model) is True

    # 1. Mutate feature parameter -> invalidates fingerprint
    orig_weight = model.encoder.stages[0].weight.clone()
    with torch.no_grad():
        model.encoder.stages[0].weight.add_(0.1)
    assert cache.verify_fingerprint(model) is False
    with torch.no_grad():
        model.encoder.stages[0].weight.copy_(orig_weight)

    # 2. Mutate head parameter -> does NOT invalidate feature cache (head is trainable)
    with torch.no_grad():
        model.head.weight.add_(0.5)
    assert cache.verify_fingerprint(model) is True

    # 3. Mutate non-tensor window config
    model.window.radius_min = 1.5
    assert cache.verify_fingerprint(model) is False
    model.window.radius_min = 0.5
    assert cache.verify_fingerprint(model) is True

    # 4. Remove window module entirely
    saved_window = model.window
    model.window = None
    assert cache.verify_fingerprint(model) is False
    model.window = saved_window
    assert cache.verify_fingerprint(model) is True

    # 5. Switching model to train mode invalidates fingerprint
    model.train(True)
    assert cache.verify_fingerprint(model) is False
    model.eval()
    assert cache.verify_fingerprint(model) is True


def test_nonfinite_parity_fails_closed(tmp_path):
    """Verify that nonfinite direct logits or cached features fail closed (NaN > tol must not pass)."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)

    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32))
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)

    # Corrupt probe feature with NaN
    cache.probe_samples[0]["f"][0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="Nonfinite cached logits detected in parity check"):
        cache.verify_parity(model, tolerance=1e-5)

    # Corrupt probe feature with Inf
    cache.probe_samples[0]["f"][0, 0, 0, 0] = float("inf")
    with pytest.raises(RuntimeError, match="Nonfinite cached logits detected in parity check"):
        cache.verify_parity(model, tolerance=1e-5)


def test_frozen_feature_state_and_rng_preservation(tmp_path):
    """Verify that validation does not mutate RNG stream, parameter versions, or gradients."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)

    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32))
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)

    # Record parameter versions
    versions_before = {name: p._version for name, p in model.named_parameters()}

    # Set torch RNG seed and check state preservation across verify_parity
    torch.manual_seed(9999)
    rng_state_before = torch.get_rng_state()

    cache.verify_parity(model)

    rng_state_after = torch.get_rng_state()
    assert torch.equal(rng_state_before, rng_state_after)

    # Check parameter versions are strictly identical
    versions_after = {name: p._version for name, p in model.named_parameters()}
    assert versions_before == versions_after

    # Model remains in eval mode
    assert model.training is False


def test_metadata_deep_copy_and_no_dict_reference_sharing(tmp_path):
    """Verify SupportFeatureCache fingerprint and records do not share dict references (deepcopy required)."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)

    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32))
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)

    # 1. Content fingerprint is an independent copy, not the same dict object
    rec_fp = cache.fingerprint["records_fingerprints"][0]
    assert rec_fp is not records[0].content_fingerprint
    assert rec_fp["view_metadata"] is not records[0].view_metadata

    # 2. Patient IDs and Frame IDs are independent snapshots
    assert cache.fingerprint["patient_ids"] is not cache.patient_ids
    assert cache.fingerprint["frame_ids"] is not cache.frame_ids

    # 3. SupportVolumeRecord.__deepcopy__ produces independent instance with cloned tensors
    rec_copy = copy.deepcopy(records[0])
    assert rec_copy is not records[0]
    assert rec_copy.images is not records[0].images
    assert rec_copy.labels is not records[0].labels
    assert rec_copy.view_metadata is not records[0].view_metadata
    assert rec_copy.verify_integrity() is True


def test_direct_regression_record_metadata_mutations(tmp_path):
    """Verify that any mutation to record metadata (frame_id, view_metadata, etc.) invalidates integrity and cache."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)
    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32))

    # Test mutation of frame_id
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    records[0].frame_id = 99
    assert records[0].verify_integrity() is False
    assert cache.verify_fingerprint(model) is False

    # Test mutation of view_metadata existing key
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    records[0].view_metadata["T"] = 999
    assert records[0].verify_integrity() is False
    assert cache.verify_fingerprint(model) is False

    # Test mutation of view_metadata new key
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    records[0].view_metadata["unauthorized"] = True
    assert records[0].verify_integrity() is False
    assert cache.verify_fingerprint(model) is False

    # Test mutation of patient_id
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    records[0].patient_id = "synthetic_999"
    assert records[0].verify_integrity() is False
    assert cache.verify_fingerprint(model) is False

    # Test mutation of image_sha256
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    records[0].image_sha256 = "0" * 64
    assert records[0].verify_integrity() is False
    assert cache.verify_fingerprint(model) is False

    # Test mutation of mask_sha256
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    records[0].mask_sha256 = "0" * 64
    assert records[0].verify_integrity() is False
    assert cache.verify_fingerprint(model) is False

    # Test mutation of source_path
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    records[0].source_path = "/tampered/cine.npy"
    assert records[0].verify_integrity() is False
    assert cache.verify_fingerprint(model) is False

    # Test mutation of mask_path
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)
    records[0].mask_path = "/tampered/mask.npy"
    assert records[0].verify_integrity() is False
    assert cache.verify_fingerprint(model) is False


def test_direct_regression_cache_metadata_probes_and_replacements(tmp_path):
    """Verify cache detects changes to patient_ids, frame_ids, replacement inputs, and probe samples."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)
    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32))

    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)

    # 1. Mutate cache.patient_ids element
    cache.patient_ids[0] = "tampered_pid"
    assert cache.verify_fingerprint(model) is False
    cache.patient_ids[0] = "synthetic_000"

    # 1b. Replace cache.patient_ids list entirely
    cache.patient_ids = ["tampered"] * len(cache.patient_ids)
    assert cache.verify_fingerprint(model) is False
    cache.patient_ids = list(cache._initial_patient_ids)
    assert cache.verify_fingerprint(model) is True

    # 2. Mutate cache.frame_ids element
    cache.frame_ids[0] = 999
    assert cache.verify_fingerprint(model) is False
    cache.frame_ids[0] = 0

    # 2b. Replace cache.frame_ids list entirely
    cache.frame_ids = [999] * len(cache.frame_ids)
    assert cache.verify_fingerprint(model) is False
    cache.frame_ids = list(cache._initial_frame_ids)
    assert cache.verify_fingerprint(model) is True

    # 3. Mutate probe tensor x
    orig_x = cache.probe_samples[0]["x"].clone()
    cache.probe_samples[0]["x"][0, 0, 0, 0].add_(1.0)
    assert cache.verify_fingerprint(model) is False
    cache.probe_samples[0]["x"].copy_(orig_x)
    assert cache.verify_fingerprint(model) is True

    # 4. Mutate probe feature tensor f
    orig_f = cache.probe_samples[0]["f"].clone()
    cache.probe_samples[0]["f"][0, 0, 0, 0].add_(0.5)
    assert cache.verify_fingerprint(model) is False
    cache.probe_samples[0]["f"].copy_(orig_f)
    assert cache.verify_fingerprint(model) is True

    # 5. Replacement inputs passed to verify_fingerprint
    clean_records = extract_complete_support_volumes(data, ["synthetic_000"])
    # Untouched replacement records verify cleanly
    assert cache.verify_fingerprint(model, clean_records) is True

    # Modified replacement records fail closed
    tampered_records = extract_complete_support_volumes(data, ["synthetic_000"])
    tampered_records[0].frame_id = 88
    assert cache.verify_fingerprint(model, tampered_records) is False

    # Replacement records with mismatched count fail closed
    assert cache.verify_fingerprint(model, clean_records[:1]) is False


def test_parity_leftover_batch_sizes_and_every_unique_bchw_shape(tmp_path):
    """Verify parity checks cover every unique BCHW shape including leftover batch sizes."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)
    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32))

    records = extract_complete_support_volumes(data, ["synthetic_000", "synthetic_001"])
    # synthetic_000: Z=3, H=35, W=41 (frames 0 and 1)
    # synthetic_001: Z=4, H=37, W=45 (frame 0)
    # With max_extract_batch=2:
    # synthetic_000 (Z=3) is split into chunks of size 2 and size 1 (leftover batch of 1!)
    # synthetic_001 (Z=4) is split into chunks of size 2 and size 2
    cache = compute_support_feature_cache(model, records, max_extract_batch=2)

    # Collect all BCHW shapes probed
    probed_bchw = {(p["B"], 3, p["H"], p["W"]) for p in cache.probe_samples}

    # Must contain both full chunk (2, 3, 35, 41) AND leftover chunk (1, 3, 35, 41)!
    assert (2, 3, 35, 41) in probed_bchw
    assert (1, 3, 35, 41) in probed_bchw
    assert (2, 3, 37, 45) in probed_bchw

    # Each probe sample has a valid boolean mask matching BCHW shape
    for probe in cache.probe_samples:
        assert probe["valid_mask"].dtype == torch.bool
        assert probe["valid_mask"].shape == (probe["B"], probe["H"], probe["W"])

    # Parity passes across all unique shapes including leftovers
    max_diff = cache.verify_parity(model, tolerance=1e-5)
    assert max_diff < 1e-5


def test_verify_parity_finite_positive_tolerance_and_mode_preservation(tmp_path):
    """Verify parity requires finite positive tolerance and preserves model mode on failure via try/finally."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)
    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32))

    records = extract_complete_support_volumes(data, ["synthetic_000"])
    cache = compute_support_feature_cache(model, records)

    # 1. Non-positive or non-finite tolerance fails closed with ValueError
    for bad_tol in [0.0, -1e-5, -1.0, float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError, match="tolerance must be a finite positive float"):
            cache.verify_parity(model, tolerance=bad_tol)

    # 2. Mode preservation on failure via try/finally
    # Put model into training mode
    model.train(True)
    assert model.training is True

    # Inject corrupt NaN feature in probe to force parity failure
    cache.probe_samples[0]["f"][0, 0, 0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="Nonfinite cached logits"):
        cache.verify_parity(model, tolerance=1e-5)

    # Crucial: model.training MUST still be True because try/finally restored it!
    assert model.training is True


def test_conservative_controller_workspace_accounting_and_finite_bounds(tmp_path):
    """Verify workspace byte estimation conservatively accounts for all controller buffers and enforces finite bounds."""
    # 1. Workspace per voxel accounts for features, logits, probs, weights, targets, responses, JVP, and margins
    workspace_bytes = estimate_controller_workspace_bytes(b_total=1, max_h=100, max_w=100, m=4, d=32, c=4)
    voxels = 1 * 100 * 100
    per_voxel = workspace_bytes / voxels
    # Per voxel must be conservative (> 700 B/voxel, accounting for float64 weights/targets and JVP)
    assert per_voxel >= 700

    # 2. Invalid parameters fail closed
    with pytest.raises(ValueError, match="positive integers"):
        estimate_controller_workspace_bytes(0, 10, 10)
    with pytest.raises(ValueError, match="m >= 0"):
        estimate_controller_workspace_bytes(1, 10, 10, m=-1)

    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)
    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32))
    records = extract_complete_support_volumes(data, ["synthetic_000"])

    # 3. Finite declared bound: None enforces default finite limit (no None production bound)
    cache = compute_support_feature_cache(model, records, max_controller_workspace_bytes=None)
    assert cache.fingerprint["controller_workspace_estimate_bytes"] > 0

    # 4. Exceeding max_controller_workspace_bytes fails closed BEFORE allocation
    est = estimate_controller_workspace_bytes(
        sum(r.num_slices for r in records),
        max(r.height for r in records),
        max(r.width for r in records),
    )
    with pytest.raises(ValueError, match="controller workspace estimate"):
        compute_support_feature_cache(model, records, max_controller_workspace_bytes=est - 1)


def test_total_native_voxels_and_frame_validations_reject_before_mask_io(tmp_path):
    """Verify total native voxels and frame metadata validation across all patients reject BEFORE any mask I/O."""
    data_dir = tmp_path / "data_pre_io"
    m_path, m_sha = create_variable_shape_fixture(data_dir)
    data = TrainingData(m_path, m_sha, synthetic=True)

    # Point all masks to nonexistent files
    for r in data.labeled:
        for m_info in r["masks"].values():
            m_info["path"] = "nonexistent_mask_never_read.npy"

    # 1. Total native voxel budget exceeds max_support_voxels:
    # Fails closed BEFORE opening nonexistent mask files
    with pytest.raises(ValueError, match="exceed declared limit"):
        extract_complete_support_volumes(data, ["synthetic_000"], max_support_voxels=50)

    # 2. Patient frame validation failure on second patient fails BEFORE reading any masks
    data.labeled[1]["masks"] = {
        "bad_frame": {"path": "nonexistent.npy", "sha256": "0" * 64}
    }
    with pytest.raises(ValueError, match="malformed non-integer frame ID"):
        extract_complete_support_volumes(data, ["synthetic_000", "synthetic_001"])

    # 3. Exceeding authorized label budget fails closed before mask I/O
    data.manifest["label_budget"] = 1
    with pytest.raises(ValueError, match="exceeds authorized label budget"):
        extract_complete_support_volumes(
            data,
            ["synthetic_000", "synthetic_001"],
            max_support_voxels=10_000_000,
        )


def test_image_normalization_and_finiteness_verified(tmp_path):
    """Verify extract_complete_support_volumes rejects non-finite or unnormalized images."""
    data_dir = tmp_path / "data_norm"
    m_path, m_sha = create_variable_shape_fixture(data_dir)
    data = TrainingData(m_path, m_sha, synthetic=True)

    r0 = data.labeled[0]
    img_path = data.resolve(r0["image"])
    orig_img = np.load(img_path)

    # 1. Image with NaN fails closed
    nan_img = orig_img.copy()
    nan_img[0, 0, 0, 0] = np.nan
    np.save(img_path, nan_img)
    r0["image_sha256"] = digest(img_path)
    data.cache.clear()

    with pytest.raises(ValueError, match="non-finite"):
        extract_complete_support_volumes(data, ["synthetic_000"])

    # 2. Image with unnormalized values outside [0, 1] fails closed
    unnorm_img = orig_img.copy()
    unnorm_img[0, 0, 0, 0] = 5.0
    np.save(img_path, unnorm_img)
    r0["image_sha256"] = digest(img_path)
    data.cache.clear()

    with pytest.raises(ValueError, match="not normalized in"):
        extract_complete_support_volumes(data, ["synthetic_000"])


def test_zero_u_mask_access(tmp_path):
    """Verify zero U mask access: U records have no mask paths and U patients are never in label reads."""
    m_path, m_sha = create_variable_shape_fixture(tmp_path / "data")
    data = TrainingData(m_path, m_sha, synthetic=True)

    # Verify U records have no masks
    for u_rec in data.unlabeled:
        assert "masks" not in u_rec
        assert "mask_path" not in u_rec

    # Attempting to declare U patient as support fails closed
    with pytest.raises(ValueError, match="overlaps with unlabeled pool"):
        extract_complete_support_volumes(data, ["synthetic_002"])

    # Normal support extraction touches only labeled patients
    records = extract_complete_support_volumes(data, ["synthetic_000"])
    assert "synthetic_000" in data.label_reads
    assert "synthetic_002" not in data.label_reads
