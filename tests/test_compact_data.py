"""Unit tests for CompactTrainingData, K=1 reuse, and audit frame streaming."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from self_audit.compact_data import (
    CompactTrainingData,
    build_compact_training_data,
    create_compact_audit_stream,
    synthetic_compact_fixture,
)
from self_audit.data import TrainingData, label_subset
from self_audit.model import digest


def test_label_subset_opt_in_k1():
    patients = ["p001", "p002", "p003", "p004"]
    # Legacy default allow_k1=False rejects count < 2
    with pytest.raises(ValueError, match="2 <= label budget"):
        label_subset(patients, 1, 42)

    # Opt-in allow_k1=True allows count = 1
    selected = label_subset(patients, 1, 42, allow_k1=True)
    assert len(selected) == 1
    assert selected[0] in patients


def test_legacy_training_data_rejects_k1_by_default(tmp_path):
    manifest_path, manifest_sha = synthetic_compact_fixture(tmp_path / "k1_data", n_labeled=1, n_unlabeled=2)
    # TrainingData default allow_k1=False must reject K=1
    with pytest.raises(ValueError, match="at least two labeled patients"):
        TrainingData(manifest_path, manifest_sha, synthetic=True)

    # TrainingData with opt-in allow_k1=True accepts K=1
    td = TrainingData(manifest_path, manifest_sha, synthetic=True, allow_k1=True)
    assert len(td.lids) == 1
    assert len(td.uids) == 2


def test_compact_training_data_k1_reuse_no_fake_patients(tmp_path):
    manifest_path, manifest_sha = synthetic_compact_fixture(tmp_path / "k1_fixture", n_labeled=1, n_unlabeled=3)
    data = CompactTrainingData(manifest_path, manifest_sha, synthetic=True, allow_k1=True)

    assert len(data.lids) == 1
    assert len(data.uids) == 3
    # Exactly one labeled patient, no duplicated IDs
    assert len(set(data.lids + data.uids)) == 4

    rng = np.random.default_rng(123)
    # In K=1 mode, li == ai == 0 (supervised patient reused without fake patients)
    for step in range(5):
        li, ai, ui = data.sample_indices(step, rng)
        assert li == 0
        assert ai == 0
        assert 0 <= ui < 3


def test_compact_training_data_k2_preserves_disjointness(tmp_path):
    manifest_path, manifest_sha = synthetic_compact_fixture(tmp_path / "k2_fixture", n_labeled=2, n_unlabeled=2)
    data = CompactTrainingData(manifest_path, manifest_sha, synthetic=True)

    assert len(data.lids) == 2
    rng = np.random.default_rng(456)
    for step in range(10):
        li, ai, ui = data.sample_indices(step, rng)
        assert li != ai, "K>=2 must maintain disjoint supervised and audit patients"
        assert 0 <= ui < 2


def test_stream_whole_labeled_frames(tmp_path):
    manifest_path, manifest_sha = synthetic_compact_fixture(tmp_path / "stream_fixture", n_labeled=1, n_unlabeled=1)
    data = CompactTrainingData(manifest_path, manifest_sha, synthetic=True)

    # Stream frames for patient 0
    frames = list(data.stream_whole_labeled_frames(0, device="cpu"))
    assert len(frames) > 0

    first = frames[0]
    assert set(first.keys()) == {"x", "y", "valid", "frame", "slice", "patient"}
    assert isinstance(first["x"], torch.Tensor)
    assert first["x"].ndim == 4
    assert first["x"].shape[1] == 3  # 3-slice context
    assert isinstance(first["y"], torch.Tensor)
    assert first["y"].ndim == 3  # (1, H, W)
    assert isinstance(first["valid"], torch.Tensor)
    assert first["valid"].dtype == torch.bool
    assert first["patient"] == data.lids[0]


def test_create_audit_batches_repeatable(tmp_path):
    manifest_path, manifest_sha = synthetic_compact_fixture(tmp_path / "audit_fixture", n_labeled=1, n_unlabeled=1)
    data = CompactTrainingData(manifest_path, manifest_sha, synthetic=True)

    audit_batches = data.create_audit_batches(0, device="cpu")
    # Verify repeatable callable
    run1 = list(audit_batches())
    run2 = list(audit_batches())
    assert len(run1) == len(run2)
    assert len(run1) > 0
    assert torch.equal(run1[0]["x"], run2[0]["x"])
    assert torch.equal(run1[0]["y"], run2[0]["y"])


def test_sample_batch_labeled_and_unlabeled(tmp_path):
    manifest_path, manifest_sha = synthetic_compact_fixture(tmp_path / "batch_fixture", n_labeled=1, n_unlabeled=2)
    data = CompactTrainingData(manifest_path, manifest_sha, synthetic=True)
    rng = np.random.default_rng(789)

    # Labeled batch
    x_l, y_l, meta_l, valid_l = data.sample_batch(0, rng, batch_size=2, labeled=True, crop_size=16, return_valid=True)
    assert x_l.shape == (2, 3, 16, 16)
    assert y_l.shape == (2, 16, 16)
    assert valid_l.shape == (2, 16, 16)
    assert "samples" in meta_l
    assert len(meta_l["samples"]) == 2

    # Unlabeled batch
    x_u, meta_u, valid_u = data.sample_batch(0, rng, batch_size=2, labeled=False, crop_size=16, return_valid=True)
    assert x_u.shape == (2, 3, 16, 16)
    assert valid_u.shape == (2, 16, 16)
    assert "samples" in meta_u


def test_audit_batches_interop_with_compact_control(tmp_path):
    from self_audit.compact_control import apply_compact_update, build_class_candidates
    from self_audit.model import build_model

    manifest_path, manifest_sha = synthetic_compact_fixture(tmp_path / "control_interop", n_labeled=1, n_unlabeled=1)
    data = CompactTrainingData(manifest_path, manifest_sha, synthetic=True)

    model, _ = build_model({"backbone": "tiny_test"}, synthetic=True)
    audit_callable = data.create_audit_batches(0, device="cpu")

    # Generate dummy candidates
    dummy_x = torch.zeros(1, 3, 32, 32)
    dummy_targets = torch.full((1, 4, 32, 32), 0.25)
    dummy_valid = torch.ones(1, 32, 32, dtype=torch.bool)
    candidates = build_class_candidates(model, dummy_x, dummy_targets, dummy_valid, lr=0.01)

    receipt = apply_compact_update(
        model,
        candidates,
        method="scalar_guard",
        audit_batches=audit_callable,
        config={"max_trials": 2, "tolerance": 1e-4},
    )
    assert "status" in receipt
    assert "accepted" in receipt



def test_compact_input_mmaps_are_bounded_and_active_views_survive_eviction(tmp_path):
    from self_audit.compact_data import CompactTrainingData, synthetic_compact_fixture
    import numpy as np
    manifest, sha = synthetic_compact_fixture(tmp_path / "bounded", n_labeled=1, n_unlabeled=5, size=16)
    data = CompactTrainingData(manifest, sha, synthetic=True, max_array_handles=1)
    assert len(data.cache) == 1
    first = data.labeled[0]
    active = data.array(first["image"], first["image_sha256"])
    expected = active.copy()
    for row in data.unlabeled:
        data.array(row["image"], row["image_sha256"])
        assert len(data.cache) == 1
    np.testing.assert_array_equal(active, expected)
    # Holding a cine while loading its frame mask remains safe with a one-map cache.
    assert list(data.stream_whole_labeled_frames(0))
    data.close()
    assert not data.cache
