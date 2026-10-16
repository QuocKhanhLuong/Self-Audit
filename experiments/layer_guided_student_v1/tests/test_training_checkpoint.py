"""Local contract tests for the new layer-guided training artifact."""

import copy

import numpy as np
import pytest
import torch
from integration import artifacts
from layer_upgrade import LayerGuidedStudent
from torch import nn
from training_v1 import checkpoint as ckpt
from training_v1 import config


class TinyEncoder(nn.Module):
    channels = (8, 16, 32, 64)

    def __init__(self):
        super().__init__()
        widths = (3, *self.channels)
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    widths[index],
                    widths[index + 1],
                    4 if index == 0 else 2,
                    stride=4 if index == 0 else 2,
                )
                for index in range(4)
            ]
        )

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


PROVENANCE = {
    "schema": "layer_guided.backbone_approval.v1",
    "approved": True,
    "model": "dinov3_convnext_tiny",
    "lineage": "official_self_supervised_dinov3",
    "repo": "/fixture/official-dinov3",
    "weights": "/fixture/dinov3.pth",
    "weights_sha256": "a" * 64,
    "verified_weights_sha256": "a" * 64,
    "source_commit": "b" * 40,
    "review_record": "TEST_FIXTURE_ONLY",
    "upstream_lineage": "declared_self_supervised_not_independently_audited",
}


def make_model(recipe=None):
    recipe = recipe or config.constructor_recipe(config.DEFAULT_RECIPE)
    model = LayerGuidedStudent(TinyEncoder(), decoder_channels=(16, 24, 32, 40)).eval()
    model.integration_identity = {
        "recipe": copy.deepcopy(recipe),
        "backbone": copy.deepcopy(PROVENANCE),
        "sources": artifacts.source_hashes(),
    }
    return model


@pytest.fixture
def approved_model(monkeypatch):
    def construct(recipe, approval, device="cpu"):
        model = make_model(recipe).to(device).eval()
        return model, copy.deepcopy(PROVENANCE)

    monkeypatch.setattr(ckpt.artifacts, "construct_student", construct)
    return make_model()


def lineage(*, optimizer_steps=0, semantic_updates=0, allowed=None):
    return {
        "bank_manifest_sha256": "1" * 64,
        "teacher": {
            "checkpoint_sha256": "2" * 64,
            "frozen": True,
            "source": "SYNTHETIC_TEST_TEACHER",
        },
        "locked_splits": {
            "train": ["TRAIN_A"],
            "val": ["VAL_B"],
            "test": ["TEST_C"],
        },
        "train_patients": ["TRAIN_A"],
        "optimizer_steps": optimizer_steps,
        "semantic_updates": semantic_updates,
        "allowed_classes": [True, True, True, True] if allowed is None else allowed,
    }


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)


def save_fixture(tmp_path, model, *, steps=0, semantic_updates=0):
    optimizer = optimizer_for(model)
    if steps:
        images = torch.rand(1, 3, 32, 32)
        optimizer.zero_grad(set_to_none=True)
        model(images)["logits"].square().mean().backward()
        optimizer.step()
    return ckpt.save_checkpoint(
        model,
        optimizer,
        tmp_path / "checkpoint",
        config.DEFAULT_RECIPE,
        lineage(optimizer_steps=steps, semantic_updates=semantic_updates),
        training_state={"epoch": 1 if steps else 0, "step": steps},
    )


def test_zero_step_artifact_is_initialized_only_and_representation_refuses_named_output(
    tmp_path, approved_model
):
    digest = save_fixture(tmp_path, approved_model)
    restored, metadata = ckpt.load_checkpoint(
        tmp_path / "checkpoint", digest, {}, device="cpu"
    )
    assert metadata["schema"] == ckpt.SCHEMA
    assert metadata["status"] == ckpt.STATUS_INITIALIZED
    assert metadata["reload"]["mode"] == "inference_only"
    with pytest.raises(ValueError, match="cannot produce named labels"):
        ckpt.predict_cine(
            restored,
            metadata,
            np.zeros((1, 2, 8, 8), dtype=np.float32),
            image_sha256="c" * 64,
        )


def test_semantic_optimizer_step_roundtrips_logits_and_same_preprocess_prediction(
    tmp_path, approved_model
):
    torch.manual_seed(20261016)
    expected_input = torch.rand(1, 3, 31, 29)
    optimizer = optimizer_for(approved_model)
    optimizer.zero_grad(set_to_none=True)
    approved_model(expected_input)["logits"].square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        expected_logits = approved_model(expected_input)["logits"].clone()
    digest = ckpt.save_checkpoint(
        approved_model,
        optimizer,
        tmp_path / "checkpoint",
        config.DEFAULT_RECIPE,
        lineage(optimizer_steps=1, semantic_updates=1),
        training_state={"epoch": 1, "step": 1},
    )
    restored, metadata = ckpt.load_checkpoint(
        tmp_path / "checkpoint", digest, {}, device="cpu"
    )
    assert metadata["status"] == ckpt.STATUS_PSEUDO
    with torch.no_grad():
        torch.testing.assert_close(
            restored(expected_input)["logits"], expected_logits, rtol=0, atol=0
        )

    cine = np.arange(2 * 3 * 9 * 11, dtype=np.float32).reshape(2, 3, 9, 11)
    first = ckpt.predict_cine(
        restored,
        metadata,
        cine,
        batch_size=1,
        min_probability=0.0,
        min_margin=0.0,
        image_sha256="d" * 64,
    )
    second = ckpt.predict_cine(
        restored,
        metadata,
        cine,
        batch_size=2,
        min_probability=0.0,
        min_margin=0.0,
        image_sha256="d" * 64,
    )
    for left, right in zip(first[:2], second[:2]):
        np.testing.assert_array_equal(left, right)
    # CPU convolution reduction order can differ with the batch dimension.
    # Fixed-state/same-input roundtrip above remains bitwise exact; confidence
    # and entropy across batch sizes obey float32 numerical parity.
    for left, right in zip(first[2:4], second[2:4]):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    assert first[0].shape == (2, 3, 9, 11)
    assert first[1].shape == first[0].shape
    assert first[2].shape == first[0].shape
    assert first[3].shape == first[0].shape
    assert first[4]["image_sha256"] == "d" * 64
    assert first[4]["context_order"] == "previous_center_next_z"
    assert first[4]["bank_or_training_gate_mutated"] is False


def test_dense_argmax_is_separate_from_probability_and_allowed_class_gate(
    tmp_path, approved_model
):
    digest = ckpt.save_checkpoint(
        approved_model,
        optimizer_for(approved_model),
        tmp_path / "checkpoint",
        config.DEFAULT_RECIPE,
        lineage(
            optimizer_steps=1, semantic_updates=1, allowed=[True, False, True, True]
        ),
        training_state={"epoch": 1, "step": 1},
    )
    restored, metadata = ckpt.load_checkpoint(tmp_path / "checkpoint", digest, {})
    dense, gated, *_ = ckpt.predict_cine(
        restored,
        metadata,
        np.ones((1, 2, 7, 9), dtype=np.float32),
        min_probability=0.0,
        min_margin=0.0,
        image_sha256="e" * 64,
    )
    assert set(np.unique(dense)).issubset({0, 1, 2, 3})
    assert set(np.unique(gated)).issubset({0, 2, 3, ckpt.UNKNOWN})


@pytest.mark.parametrize("mutator", ["manifest", "payload", "source"])
def test_wrong_hash_or_source_drift_is_rejected(
    tmp_path, approved_model, mutator, monkeypatch
):
    digest = save_fixture(tmp_path, approved_model)
    folder = tmp_path / "checkpoint"
    if mutator == "manifest":
        with pytest.raises(ValueError, match="manifest digest"):
            ckpt.load_checkpoint(folder, "f" * 64, {})
    elif mutator == "payload":
        (folder / "model.pt").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="payload digest"):
            ckpt.load_checkpoint(folder, digest, {})
    else:
        original = ckpt.training_config.source_hashes
        monkeypatch.setattr(
            ckpt.training_config,
            "source_hashes",
            lambda: {**original(), "training_v1/config.py": "f" * 64},
        )
        with pytest.raises(ValueError, match="source hashes"):
            ckpt.load_checkpoint(folder, digest, {})


def test_old_schema_is_not_accepted(tmp_path):
    folder = tmp_path / "legacy"
    folder.mkdir()
    torch.save({"model": {}, "width": 32}, folder / "last.pt")
    with pytest.raises(ValueError, match="new training checkpoint inventory"):
        ckpt.load_checkpoint(folder, "a" * 64, {})


def test_lineage_and_resume_contract_reject_false_claims(tmp_path, approved_model):
    optimizer = optimizer_for(approved_model)
    with pytest.raises(ValueError, match="cannot exceed"):
        ckpt.save_checkpoint(
            approved_model,
            optimizer,
            tmp_path / "invalid",
            config.DEFAULT_RECIPE,
            lineage(optimizer_steps=0, semantic_updates=1),
            training_state={"epoch": 0, "step": 0},
        )
    with pytest.raises(ValueError, match="exact resume"):
        ckpt.save_checkpoint(
            approved_model,
            optimizer,
            tmp_path / "invalid_resume",
            config.DEFAULT_RECIPE,
            lineage(),
            training_state={"epoch": 0, "step": 0, "rng_states": {}},
        )
