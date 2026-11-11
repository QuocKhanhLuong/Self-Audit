"""Synthetic contracts for image-only A-3D semi-supervised inputs."""

import copy
import json
from pathlib import Path

import pytest

from self_audit.candidate_a3d.config import validate_config
from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
from self_audit.model import digest


def _smoke_config():
    return json.loads(Path("configs/candidate_a3d/smoke.json").read_text())


def test_unlabeled_loader_covers_all_native_frames_without_mask_reads(tmp_path):
    fixture = synthetic_fixture(
        tmp_path / "fixture", n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(3, 4, 5, 6),
    )
    data = CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)
    keys = data.unlabeled_frame_keys()
    assert keys == [(0, 0), (0, 1), (0, 2)]
    for key in keys:
        image, geometry, metadata = data.load_unlabeled_frame(key)
        assert tuple(image.shape) == (1, 1, 4, 5, 6)
        assert geometry["context"] == "full_volume"
        assert metadata == {"patient": "train_unlabeled_000", "frame": key[1]}
    assert data.unlabeled_image_reads == {"train_unlabeled_000"}
    assert data.label_reads == set()
    assert not any("masks" in record for record in data.unlabeled)


def test_unlabeled_loader_rejects_out_of_range_and_bad_image(tmp_path):
    fixture = synthetic_fixture(
        tmp_path / "fixture", n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(1, 3, 4, 5),
    )
    data = CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)
    with pytest.raises(IndexError, match="outside cached geometry"):
        data.load_unlabeled_frame((0, 1))
    manifest = json.loads(fixture["train"].read_text())
    record = manifest["unlabeled"][0]
    image_path = fixture["train"].parent / record["image"]
    import numpy as np

    image = np.load(image_path, allow_pickle=False)
    image = image.copy()
    image[0, 0, 0, 0] = np.nan
    np.save(image_path, image)
    record["image_sha256"] = digest(image_path)
    bad_manifest = fixture["train"].parent / "bad_unlabeled.json"
    bad_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="finite"):
        CandidateA3DData(bad_manifest, digest(bad_manifest), synthetic=True)


def test_semi_supervised_config_is_explicit_and_strict():
    config = _smoke_config()
    legacy = validate_config(copy.deepcopy(config))
    assert "semi_supervised" not in legacy

    config["semi_supervised"] = {
        "enabled": True,
        "ema_decay": 0.99,
        "teacher_weight": 1.0,
        "student_weight": 1.0,
        "rampup_epochs": 10,
        "confidence_threshold": 0.7,
        "background_weight": 0.1,
        "steps_per_epoch": 0,
        "intensity_gamma": [0.7, 1.5],
        "intensity_noise_std": 0.03,
    }
    validated = validate_config(config)
    assert validated["semi_supervised"]["enabled"] is True
    assert validated["semi_supervised"]["intensity_gamma"] == [0.7, 1.5]

    unknown = copy.deepcopy(config)
    unknown["semi_supervised"]["typo"] = 1
    with pytest.raises(ValueError, match="exact declared fields"):
        validate_config(unknown)

    zero_weight = copy.deepcopy(config)
    zero_weight["semi_supervised"]["teacher_weight"] = 0.0
    with pytest.raises(ValueError, match="teacher_weight"):
        validate_config(zero_weight)
