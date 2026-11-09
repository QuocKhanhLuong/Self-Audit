import copy
import hashlib

import pytest
import torch

from self_audit.model import build_model
from self_audit.candidate_a.teacher import (
    CLASS_NAMES,
    DualBranchTeacher,
    build_teacher,
    set_teacher_stage,
    teacher_loss,
)


def recipe():
    return {
        "backbone": "tiny_test",
        "class_mapping": list(CLASS_NAMES),
        "num_classes": 4,
    }


def test_teacher_shared_encoder_and_separate_branches_receive_gradients():
    torch.manual_seed(3)
    teacher, provenance = build_teacher(recipe(), synthetic=True)
    assert isinstance(teacher, DualBranchTeacher)
    assert provenance["segmentation_pretrained"] is False
    calls = []
    handle = teacher.encoder.register_forward_hook(lambda *_: calls.append(1))
    try:
        x = torch.rand(2, 3, 29, 35)
        outputs = teacher(x)
    finally:
        handle.remove()
    assert calls == [1], "region and boundary branches must share one encoder pass"
    assert set(outputs) == {"region_logits", "boundary_logits", "sdf"}
    assert outputs["region_logits"].shape == (2, 4, 29, 35)
    assert outputs["boundary_logits"].shape == (2, 4, 29, 35)
    assert outputs["sdf"].shape == (2, 4, 29, 35)
    assert float(outputs["sdf"].detach().min()) >= -1.0
    assert float(outputs["sdf"].detach().max()) <= 1.0

    labels = torch.zeros(2, 29, 35, dtype=torch.long)
    labels[:, 8:20, 10:24] = 2
    sdf_target = torch.zeros_like(outputs["sdf"])
    loss = teacher_loss(outputs, labels, sdf_target, 0.25)
    assert torch.isfinite(loss)
    loss.backward()

    region_ids = {id(parameter) for parameter in teacher.region_branch.parameters()}
    boundary_ids = {id(parameter) for parameter in teacher.boundary_branch.parameters()}
    assert region_ids.isdisjoint(boundary_ids)
    assert teacher.region_head.weight.grad is not None
    assert teacher.boundary_head.weight.grad is not None
    assert teacher.sdf_head.weight.grad is not None
    assert teacher.region_branch.path[0].weight.grad is not None
    assert teacher.boundary_branch.detail[0].weight.grad is not None
    assert teacher.encoder.stages[0].weight.grad is not None


def test_teacher_loss_rejects_nonfinite_and_wrong_ontology():
    teacher, _ = build_teacher(recipe(), synthetic=True)
    outputs = teacher(torch.rand(1, 3, 32, 32))
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, 0, 0] = 255
    target = torch.zeros_like(outputs["sdf"])
    assert torch.isfinite(teacher_loss(outputs, labels, target, 0.0))
    with pytest.raises(ValueError, match="unknown class"):
        bad = labels.clone()
        bad[:, 0, 1] = 7
        teacher_loss(outputs, bad, target, 0.0)
    with pytest.raises(ValueError, match="NaN or Inf"):
        bad_outputs = dict(outputs)
        bad_outputs["sdf"] = outputs["sdf"].clone()
        bad_outputs["sdf"][0, 0, 0, 0] = float("nan")
        teacher_loss(bad_outputs, labels, target, 0.0)


def test_stage_controls_encoder_without_disabling_new_branches():
    teacher, _ = build_teacher(recipe(), synthetic=True)
    set_teacher_stage(teacher, "warmup")
    assert not any(parameter.requires_grad for parameter in teacher.encoder.parameters())
    assert all(parameter.requires_grad for parameter in teacher.input_adapter.parameters())
    assert any(parameter.requires_grad for parameter in teacher.shared_lateral.parameters())
    assert any(parameter.requires_grad for parameter in teacher.region_branch.parameters())
    assert any(parameter.requires_grad for parameter in teacher.boundary_branch.parameters())
    teacher.train()
    assert not any(parameter.requires_grad for parameter in teacher.encoder.parameters())

    set_teacher_stage(teacher, "pretrain")
    assert all(parameter.requires_grad for parameter in teacher.encoder.parameters())
    set_teacher_stage(teacher, "joint", freeze=True)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    set_teacher_stage(teacher, "joint")
    assert all(parameter.requires_grad for parameter in teacher.parameters())
    with pytest.raises(ValueError, match="stage"):
        set_teacher_stage(teacher, "bad")


def test_native_initialization_requires_explicit_ontology_and_four_class_head():
    base, _ = build_model({"backbone": "tiny_test"}, synthetic=True)
    state = copy.deepcopy(base.state_dict())
    metadata = {"model_class": "NativeStudent"}
    with pytest.raises(ValueError, match="explicit class_mapping"):
        build_teacher({"backbone": "tiny_test"}, synthetic=True, native_state=state, native_metadata=metadata)

    teacher, provenance = build_teacher(
        recipe(), synthetic=True, native_state=state, native_metadata=metadata
    )
    assert provenance["native_initialization"]["state_scope"] == "shared_encoder_and_native_projection"
    assert provenance["segmentation_pretrained"] is False
    torch.testing.assert_close(
        teacher.encoder.stages[0].weight,
        base.encoder.stages[0].weight,
    )

    bad_state = copy.deepcopy(state)
    bad_state["head.weight"] = bad_state["head.weight"][:3]
    with pytest.raises(ValueError, match="exactly four classes"):
        build_teacher(recipe(), synthetic=True, native_state=bad_state, native_metadata=metadata)


def test_full_teacher_checkpoint_schema_and_hash_are_strict(tmp_path):
    teacher, _ = build_teacher(recipe(), synthetic=True)
    payload = {
        "schema": "self_audit.teacher.v1",
        "model_class": "DualBranchTeacher",
        "num_classes": 4,
        "class_mapping": list(CLASS_NAMES),
        "state_dict": teacher.state_dict(),
    }
    checkpoint = tmp_path / "teacher.pt"
    torch.save(payload, checkpoint)
    hasher = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = recipe() | {"segmentation_checkpoint_sha256": hasher}
    loaded, provenance = build_teacher(config, synthetic=True, segmentation_checkpoint=checkpoint)
    assert provenance["segmentation_pretrained"] is True
    assert loaded.segmentation_ready is True

    bad_config = recipe() | {"segmentation_checkpoint_sha256": "0" * 64}
    with pytest.raises(ValueError, match="hash mismatch"):
        build_teacher(bad_config, synthetic=True, segmentation_checkpoint=checkpoint)

    native_payload = payload | {"schema": "self_audit.checkpoint.v2"}
    native_checkpoint = tmp_path / "native.pt"
    torch.save(native_payload, native_checkpoint)
    native_hash = hashlib.sha256(native_checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="native v2"):
        build_teacher(
            recipe() | {"segmentation_checkpoint_sha256": native_hash},
            synthetic=True,
            segmentation_checkpoint=native_checkpoint,
        )
