import pytest
import torch

from self_audit.candidate_fuse.teacher import (
    FuseTeacher,
    build_teacher,
    configure_teacher_step,
    teacher_parameter_groups,
    teacher_loss,
)
from self_audit.candidate_fuse.config import default_config as fuse_default_config


def _config():
    return fuse_default_config(synthetic=True)


def test_fuse_teacher_single_forward_features_and_freeze():
    teacher, _ = build_teacher(_config(), synthetic=True)
    assert isinstance(teacher, FuseTeacher)
    assert teacher.feature_channels == (4, 8, 12, 16, 20)
    assert tuple(p[0] for p in teacher.feature_phases) == (
        (1, 1, 1), (1, 2, 2), (1, 4, 4), (2, 8, 8), (2, 16, 16))
    calls = 0
    original = teacher.model.encoder.forward
    def counted(x):
        nonlocal calls
        calls += 1
        return original(x)
    teacher.model.encoder.forward = counted
    image = torch.randn(1, 1, 16, 32, 32)
    geometry = {"shape": (16, 32, 32), "spacing_zyx": (1., 1., 1.),
                "affine_zyx": torch.eye(4).tolist(), "context": "full_volume"}
    result = teacher(image, geometry)
    assert calls == 1
    assert len(result["features"]) == 5
    teacher.freeze_for_joint()
    teacher.train(True)
    assert not teacher.training and not teacher.model.training
    assert all(not p.requires_grad for p in teacher.parameters())
    assert not any(p.grad is not None for p in teacher.parameters())


def test_freeze_for_joint_clears_existing_gradients():
    teacher, _ = build_teacher(_config(), synthetic=True)
    for parameter in teacher.parameters():
        parameter.grad = torch.ones_like(parameter)
    teacher.freeze_for_joint()
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_teacher_groups_deduplicate_shared_encoder_and_schedule():
    teacher, _ = build_teacher(_config(), synthetic=True)
    groups = teacher_parameter_groups(teacher, _config())
    flat = [p for group in groups for p in group["params"]]
    assert len(flat) == len({id(p) for p in flat}) == len(list(teacher.parameters()))
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    geometry = {"shape": (16, 32, 32), "spacing_zyx": (1., 1., 1.),
                "affine_zyx": torch.eye(4).tolist(), "context": "full_volume"}
    image = torch.randn(1, 1, 16, 32, 32)
    target = torch.randint(0, 4, (1, 16, 32, 32))
    phase = configure_teacher_step(teacher, optimizer, 0, 8000, _config())
    assert phase["phase"] == "T-a"
    optimizer.zero_grad(set_to_none=True)
    teacher_loss(teacher(image, geometry, return_features=False), target, geometry).backward()
    assert any(p.grad is not None for p in teacher.model.boundary_path.parameters())
    assert all(p.grad is None for p in teacher.model.encoder.parameters())
    phase = configure_teacher_step(teacher, optimizer, 400, 8000, _config())
    assert phase["phase"] == "T-b"
    optimizer.zero_grad(set_to_none=True)
    teacher_loss(teacher(image, geometry, return_features=False), target, geometry).backward()
    assert any(p.grad is not None for p in teacher.model.boundary_path.parameters())
    assert any(p.grad is not None for p in teacher.model.encoder.parameters())
    assert configure_teacher_step(teacher, optimizer, 7999, 8000, _config())["learning_rates"]["encoder"] < 1e-10


def test_teacher_strict_checkpoint_rejects_alias_divergence_and_missing_head():
    teacher, _ = build_teacher(_config(), synthetic=True)
    state = teacher.state_dict()
    divergent = dict(state)
    key = "model.encoder.stem.convs.0.conv.weight"
    divergent[key] = divergent[key].clone() + 1
    with pytest.raises(ValueError, match="divergent shared encoder aliases"):
        teacher.load_state_dict(divergent)
    missing = dict(state)
    missing.pop(next(key for key in missing if key.startswith("model.boundary_path.")))
    with pytest.raises(RuntimeError, match="Missing key"):
        teacher.load_state_dict(missing)


def test_fuse_real_teacher_keeps_strict_pretrained_contract():
    config = _config()
    config["model"]["teacher"]["pretrained"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Phase-0 checkpoint hash"):
        build_teacher(config, synthetic=False)


def test_teacher_tb_schedule_matches_q3d_boundaries():
    teacher, _ = build_teacher(_config(), synthetic=True)
    optimizer = torch.optim.AdamW(teacher_parameter_groups(teacher, _config()), weight_decay=1e-4)
    expected = {
        400: (1.0 / 380.0),
        779: (380.0 / 380.0),
        780: (0.5 * (1.0 + __import__("math").cos(__import__("math").pi / 7220.0))),
        7999: 0.0,
    }
    for update, scale in expected.items():
        info = configure_teacher_step(teacher, optimizer, update, 8000, _config())
        assert info["phase"] == "T-b"
        assert info["learning_rates"]["encoder"] == pytest.approx(1e-4 * scale, abs=1e-15)
        assert info["learning_rates"]["heads"] == pytest.approx(1e-3 * scale, abs=1e-15)
