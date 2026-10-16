import json

import numpy as np
import pytest
import torch
from self_audit_guidance.io import digest, freeze

from integration.bank import PRODUCER_HASHES, load_training_context


def test_historical_bank_to_image_feature_interface(tmp_path, monkeypatch):
    image = np.arange(2 * 3 * 47 * 53, dtype=np.float32).reshape(2, 3, 47, 53) / 100 - 2
    features = (np.arange(2 * 3 * 7 * 9 * 64).reshape(2, 3, 7, 9, 64) % 100).astype(
        np.float16
    )
    q = np.full((2, 3, 7, 9, 4), 0.25, np.float32)
    np.savez(
        tmp_path / "TRAIN_A.npz",
        image=image,
        features=features,
        q=q,
        r0=q,
        reliability=np.zeros(q.shape[:-1]),
        labels=np.full(q.shape[:-1], 255, np.uint8),
    )
    freeze(
        tmp_path,
        {
            "role": "guidance_bank",
            "split": "train",
            "patients": ["TRAIN_A"],
            "locked_splits": {"train": ["TRAIN_A"], "val": ["VAL_B"], "test": []},
            "teacher": {"frozen": True, "source": "manufactured_test"},
            "source_sha256": PRODUCER_HASHES,
            "image_records": {
                "TRAIN_A": {
                    "image_sha256": "1" * 64,
                    "shape_xyzt": [53, 47, 3, 2],
                    "affine": np.eye(4).tolist(),
                    "spatial_unit": "unknown",
                }
            },
        },
    )
    before = (tmp_path / "FROZEN.json").read_bytes()
    expected = digest(tmp_path / "FROZEN.json")
    result = load_training_context(
        tmp_path, "TRAIN_A", [(0, 1), (1, 2)], expected_bank_sha256=expected
    )
    assert result["images"].shape == (2, 3, 47, 53)
    assert result["teacher_features"].shape == (2, 64, 7, 9)
    np.testing.assert_array_equal(
        result["teacher_features"][0].numpy(), features[0, 1].transpose(2, 0, 1)
    )
    assert 0 <= result["images"].min() <= result["images"].max() <= 1
    assert result["geometry"].native_hw == (47, 53)
    assert result["semantic_admission"] is False
    assert "q" not in result and "labels" not in result
    assert (tmp_path / "FROZEN.json").read_bytes() == before
    with pytest.raises(ValueError, match="TRAIN"):
        load_training_context(
            tmp_path, "VAL_B", [(0, 1)], expected_bank_sha256=expected
        )
    with pytest.raises(ValueError, match="locked input"):
        load_training_context(
            tmp_path, "TRAIN_A", [(0, 1)], expected_bank_sha256="0" * 64
        )
    from integration import bank as adapter

    original_loader = adapter.load_bank_patient

    def mutating_loader(*args):
        data = original_loader(*args)
        manifest = json.loads((tmp_path / "FROZEN.json").read_text())
        manifest["status"] = "CHANGED_DURING_READ"
        (tmp_path / "FROZEN.json").write_text(json.dumps(manifest))
        return data

    monkeypatch.setattr(adapter, "load_bank_patient", mutating_loader)
    with pytest.raises(ValueError, match="changed while reading"):
        load_training_context(
            tmp_path, "TRAIN_A", [(0, 1)], expected_bank_sha256=expected
        )


def test_invalid_nonfinite_cine_refused():
    from integration.geometry import normalize_cine

    with pytest.raises(ValueError):
        normalize_cine(torch.full((1, 1, 2, 2), float("nan")), "1" * 64)
