"""A-3D teacher contracts on manufactured CPU tensors only."""

import sys
import types

import pytest
import torch

import self_audit.candidate_a3d.teacher as teacher_module
from self_audit.candidate_a3d.teacher import (
    DualBranchTeacher3D,
    MedNeXtL3DAdapter,
    _segmentation_checkpoint_payload,
    build_teacher3d,
    set_teacher_stage,
    teacher_loss,
)


def config():
    return {
        "backbone": "tiny_test3d",
        "input_channels": 1,
        "feature_channels": [8, 16, 24, 32],
        "class_mapping": ["BG", "RV", "MYO", "LV"],
    }


def geometry(shape, spacing=(2.0, 1.0, 0.8)):
    return {
        "shape": tuple(shape), "spacing_zyx": tuple(spacing),
        "affine_zyx": [[spacing[0], 0, 0, 0], [0, spacing[1], 0, 0],
                       [0, 0, spacing[2], 0], [0, 0, 0, 1]],
        "context": "full_volume",
    }


def test_teacher_is_true_3d_padded_and_cropped_with_two_processing_paths():
    teacher, provenance = build_teacher3d(config(), synthetic=True)
    assert isinstance(teacher, DualBranchTeacher3D)
    assert provenance["quality"] == "SYNTHETIC_RANDOM_INIT"
    x = torch.rand(1, 1, 5, 7, 9)
    output = teacher(x, geometry=geometry(x.shape[-3:]))
    assert set(output) == {"region_logits", "boundary_logits", "sdf"}
    assert output["region_logits"].shape == (1, 4, 5, 7, 9)
    assert output["boundary_logits"].shape == output["region_logits"].shape
    assert output["sdf"].shape == output["region_logits"].shape
    assert output["region_logits"].data_ptr() != output["boundary_logits"].data_ptr()
    assert torch.isfinite(output["sdf"]).all()
    assert float(output["sdf"].detach().min()) >= -1 and float(output["sdf"].detach().max()) <= 1


def test_teacher_loss_is_3d_and_backpropagates_both_branches():
    teacher, _ = build_teacher3d(config(), synthetic=True)
    x = torch.rand(1, 1, 6, 8, 10)
    y = torch.zeros(1, 6, 8, 10, dtype=torch.long)
    y[:, 2:4, 3:6, 4:7] = 2
    output = teacher(x, geometry=geometry(x.shape[-3:]))
    target_sdf = torch.zeros_like(output["sdf"])
    loss = teacher_loss(output, y, target_sdf, lambda_distance=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in teacher.region_path.parameters())
    assert any(parameter.grad is not None for parameter in teacher.boundary_path.parameters())


def test_spacing_conditions_boundary_without_changing_original_region_path():
    teacher, _ = build_teacher3d(config(), synthetic=True)
    teacher.eval()
    x = torch.rand(1, 1, 5, 7, 9)
    with torch.inference_mode():
        first = teacher(x, geometry=geometry(x.shape[-3:], spacing=(1.0, 1.0, 1.0)))
        second = teacher(x, geometry=geometry(x.shape[-3:], spacing=(4.0, 1.0, 0.5)))
    torch.testing.assert_close(first["region_logits"], second["region_logits"])
    assert not torch.equal(first["boundary_logits"], second["boundary_logits"])
    assert not torch.equal(first["sdf"], second["sdf"])


def test_external_adapter_preserves_original_region_decoder_separately_from_boundary():
    class Stage(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv3 = torch.nn.Conv3d(channels, channels, 1)

        def forward(self, x):
            return self.conv3(x)

    class Out(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_out = torch.nn.Conv3d(8, 4, 1)

        def forward(self, x):
            return self.conv_out(x)

    class FakeNativeMedNeXt(torch.nn.Module):
        do_ds = False
        _candidate_a_native_v1 = True

        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Conv3d(1, 8, 1)
            for name in ("enc_block_0", "down_0", "enc_block_1", "down_1",
                         "enc_block_2", "down_2", "enc_block_3", "down_3",
                         "bottleneck", "up_3", "dec_block_3", "up_2",
                         "dec_block_2", "up_1", "dec_block_1", "up_0", "dec_block_0"):
                setattr(self, name, Stage(8))
            self.out_0 = Out()

        def forward(self, x):
            x = self.stem(x)
            r0 = self.enc_block_0(x); x = self.down_0(r0)
            r1 = self.enc_block_1(x); x = self.down_1(r1)
            r2 = self.enc_block_2(x); x = self.down_2(r2)
            r3 = self.enc_block_3(x); x = self.down_3(r3)
            x = self.bottleneck(x)
            x = self.dec_block_3(self.up_3(x) + r3)
            x = self.dec_block_2(self.up_2(x) + r2)
            x = self.dec_block_1(self.up_1(x) + r1)
            return self.out_0(self.dec_block_0(self.up_0(x) + r0))

    adapter = MedNeXtL3DAdapter(FakeNativeMedNeXt(), source_repo="/local/mednext",
                                source_commit="verified", weights_sha256=None,
                                features_api="native_mednext_v1_sequence",
                                feature_channels=(8, 8, 8, 8, 8))
    teacher = DualBranchTeacher3D(adapter, input_channels=1)
    x = torch.rand(1, 1, 6, 8, 10)
    out = teacher(x, geometry=geometry(x.shape[-3:]))
    padded, original_shape = teacher._pad(x)
    assert tuple(padded.shape[-3:]) == (32, 32, 32)
    expected = adapter.forward_region(padded)[..., :original_shape[0], :original_shape[1], :original_shape[2]]
    torch.testing.assert_close(out["region_logits"], expected)
    assert out["region_logits"].shape == (1, 4, 6, 8, 10)
    assert out["boundary_logits"].shape == out["region_logits"].shape
    assert teacher._external_region_path


def test_warmup_freezes_original_trunk_and_region_but_not_boundary():
    teacher, _ = build_teacher3d(config(), synthetic=True)
    set_teacher_stage(teacher, "teacher_warmup")
    assert not any(parameter.requires_grad for parameter in teacher.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in teacher.region_path.parameters())
    assert any(parameter.requires_grad for parameter in teacher.boundary_path.parameters())
    teacher.train()
    assert not teacher.encoder.training and not teacher.region_path.training
    set_teacher_stage(teacher, "joint", freeze=True)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert not teacher.training


def test_student_or_teacher_checkpoint_metadata_cannot_change_ontology():
    teacher, _ = build_teacher3d(config(), synthetic=True)
    with pytest.raises(ValueError, match="class mapping"):
        build_teacher3d(config(), synthetic=True,
                        checkpoint_state=teacher.state_dict(),
                        checkpoint_metadata={"model_class": "DualBranchTeacher3D",
                                             "class_mapping": ["BG", "LV", "MYO", "RV"]})
    with pytest.raises(ValueError, match="model_class"):
        build_teacher3d(config(), synthetic=True,
                        checkpoint_state=teacher.state_dict(),
                        checkpoint_metadata={"model_class": "Other", "class_mapping": ["BG", "RV", "MYO", "LV"]})


def test_patient_build_rejects_tiny_backbone_and_unverified_refineformer():
    with pytest.raises(ValueError, match="synthetic"):
        build_teacher3d(config(), synthetic=False)
    with pytest.raises(ValueError, match="not verified"):
        build_teacher3d({**config(), "backbone": "refineformer3d_standard"}, synthetic=True)


def test_mednext_configuration_fails_closed_without_local_checkout_or_weights():
    mednext = {
        "backbone": "mednext_l_k3", "input_channels": 1,
        "source_repo": "/missing/mednext", "source_commit": "0" * 40,
        "weights": None, "weights_sha256": None,
    }
    with pytest.raises((FileNotFoundError, ValueError), match="MedNeXt|source"):
        build_teacher3d(mednext, synthetic=False)


def _validated_source_config(source_repo, source_commit=None):
    return {
        "backbone": "mednext_l_k3", "input_channels": 1,
        "source_repo": str(source_repo),
        "source_commit": source_commit or teacher_module.UPSTREAM_MAIN_SHA,
        "weights": None, "weights_sha256": None,
    }


def test_mednext_source_commit_is_explicitly_verified(tmp_path, monkeypatch):
    source_repo = tmp_path / "mednext"
    source_repo.mkdir()
    monkeypatch.setattr(teacher_module, "_git_head", lambda _: teacher_module.UPSTREAM_MAIN_SHA)
    with pytest.raises(ValueError, match="verified only"):
        build_teacher3d(_validated_source_config(source_repo, "f" * 40), synthetic=False)


def test_mednext_rejects_untracked_or_tracked_source_changes(tmp_path, monkeypatch):
    source_repo = tmp_path / "mednext"
    source_repo.mkdir()
    monkeypatch.setattr(teacher_module, "_git_head", lambda _: teacher_module.UPSTREAM_MAIN_SHA)
    monkeypatch.setattr(teacher_module, "_git_dirty", lambda _: True)
    with pytest.raises(ValueError, match="tracked or untracked"):
        build_teacher3d(_validated_source_config(source_repo), synthetic=False)


def test_mednext_rejects_cached_foreign_module_before_import(tmp_path, monkeypatch):
    source_repo = tmp_path / "mednext"
    source_repo.mkdir()
    monkeypatch.setattr(teacher_module, "_git_head", lambda _: teacher_module.UPSTREAM_MAIN_SHA)
    monkeypatch.setattr(teacher_module, "_git_dirty", lambda _: False)
    foreign = types.ModuleType("nnunet_mednext")
    foreign.__file__ = "/foreign/nnunet_mednext/__init__.py"
    monkeypatch.setitem(sys.modules, "nnunet_mednext", foreign)
    with pytest.raises(RuntimeError, match="outside"):
        build_teacher3d(_validated_source_config(source_repo), synthetic=False)


def test_segmentation_weights_require_dense_train_provenance():
    with pytest.raises(ValueError, match="raw/unknown-cohort"):
        _segmentation_checkpoint_payload({"out_0.conv_out.weight": torch.zeros(4, 1, 1, 1, 1)})
    state = {"out_0.conv_out.weight": torch.zeros(4, 1, 1, 1, 1)}
    payload = {
        "state_dict": state,
        "metadata": {
            "labeled_ids": ["p1", "p2"], "unlabeled_ids": ["p3"],
            "label_budget": 2, "data_sha256": "a" * 64,
            "class_mapping": ["BG", "RV", "MYO", "LV"],
        },
    }
    loaded, metadata = _segmentation_checkpoint_payload(payload)
    assert loaded is state
    assert metadata["label_budget"] == 2
    with pytest.raises(ValueError, match="label_budget"):
        _segmentation_checkpoint_payload({**payload, "metadata": {**payload["metadata"], "label_budget": 1}})
