"""Focused CPU-only PG model contracts (synthetic architecture fixtures only)."""

from __future__ import annotations

import torch
from torch import nn

from self_audit.candidate_a3d.teacher import MedNeXtL3DAdapter, UPSTREAM_MAIN_SHA
from self_audit.candidate_pg.student_pg import PGStudent
from self_audit.candidate_pg.teacher_mednext import (
    MEDNEXT_CHANNELS,
    MEDNEXT_DOWN_STRIDES,
    MEDNEXT_PHASE_STEPS,
    MedNeXtPGTeacher,
    _PGAnisotropicMedNeXt,
    derive_mednext_feature_phases,
)
from self_audit.candidate_q3d.losses import q3d_teacher_loss


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int]):
        super().__init__()
        # Official MedNeXtDownBlock samples in depthwise conv1 and changes
        # channels in its stride-1 conv3 projection.
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, 1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(in_channels, out_channels, 1)
        self.do_res = False
        self.resample_do_res = False

    def forward(self, x):
        return self.conv3(self.act(self.conv2(self.norm(self.conv1(x)))))


class _Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int]):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.ConvTranspose3d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, 1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(in_channels, out_channels, 1)
        self.do_res = False
        self.resample_do_res = False

    def forward(self, x):
        x = self.conv3(self.act(self.conv2(self.norm(self.conv1(x)))))
        sz, sy, sx = self.stride
        return nn.functional.pad(x, (1 if sx > 1 else 0, 0, 1 if sy > 1 else 0, 0, 1 if sz > 1 else 0, 0))


class _NativeMedNeXtFixture(nn.Module):
    """Shape-faithful official-v1 module sequence; not a patient model."""

    _candidate_a_native_v1 = True
    do_ds = False

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv3d(1, 32, 3, padding=1)
        self.enc_block_0 = nn.Conv3d(32, 32, 3, padding=1)
        self.down_0 = _Down(32, 64, MEDNEXT_DOWN_STRIDES[0])
        self.enc_block_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.down_1 = _Down(64, 128, MEDNEXT_DOWN_STRIDES[1])
        self.enc_block_2 = nn.Conv3d(128, 128, 3, padding=1)
        self.down_2 = _Down(128, 256, MEDNEXT_DOWN_STRIDES[2])
        self.enc_block_3 = nn.Conv3d(256, 256, 3, padding=1)
        self.down_3 = _Down(256, 512, MEDNEXT_DOWN_STRIDES[3])
        self.bottleneck = nn.Conv3d(512, 512, 3, padding=1)
        self.up_3 = _Up(512, 256, MEDNEXT_DOWN_STRIDES[3])
        self.dec_block_3 = nn.Conv3d(256, 256, 3, padding=1)
        self.up_2 = _Up(256, 128, MEDNEXT_DOWN_STRIDES[2])
        self.dec_block_2 = nn.Conv3d(128, 128, 3, padding=1)
        self.up_1 = _Up(128, 64, MEDNEXT_DOWN_STRIDES[1])
        self.dec_block_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.up_0 = _Up(64, 32, MEDNEXT_DOWN_STRIDES[0])
        self.dec_block_0 = nn.Conv3d(32, 32, 3, padding=1)
        self.out_0 = nn.Conv3d(32, 4, 1)

    def forward(self, x):  # pragma: no cover - adapter owns this path
        x = self.stem(x)
        r0 = self.enc_block_0(x)
        x = self.down_0(r0)
        r1 = self.enc_block_1(x)
        x = self.down_1(r1)
        r2 = self.enc_block_2(x)
        x = self.down_2(r2)
        r3 = self.enc_block_3(x)
        x = self.down_3(r3)
        x = self.bottleneck(x)
        x = self.dec_block_3(self.up_3(x) + r3)
        x = self.dec_block_2(self.up_2(x) + r2)
        x = self.dec_block_1(self.up_1(x) + r1)
        return self.out_0(self.dec_block_0(self.up_0(x) + r0))


def _teacher_config():
    return {
        "model": {
            "teacher": {
                "arch": "mednext_l",
                "kernel": 3,
                "channels": list(MEDNEXT_CHANNELS),
                "down_strides": [list(value) for value in MEDNEXT_DOWN_STRIDES],
                "feature_phases": [list(value) for value in MEDNEXT_PHASE_STEPS],
            },
            "student": {"channels": [16, 32, 64, 128, 256], "num_classes": 4},
        }
    }


def _geometry(shape):
    return {
        "shape": tuple(shape),
        "spacing_zyx": (2.0, 1.0, 0.8),
        "affine_zyx": torch.eye(4).tolist(),
        "context": "full_volume",
    }


def _teacher():
    adapter = MedNeXtL3DAdapter(
        _NativeMedNeXtFixture(), source_repo="/synthetic/verified-mednext",
        source_commit=UPSTREAM_MAIN_SHA, weights_sha256=None,
        features_api="native_mednext_v1_sequence", feature_channels=MEDNEXT_CHANNELS,
    )
    return MedNeXtPGTeacher(_teacher_config(), model=adapter)


def test_mednext_phases_are_derived_from_live_anisotropic_convs():
    model = _NativeMedNeXtFixture()
    phases = derive_mednext_feature_phases(model)
    assert tuple(phase[0] for phase in phases) == MEDNEXT_PHASE_STEPS
    assert all(tuple(offset) == (0.0, 0.0, 0.0) for _, offset in phases)


def test_anisotropic_full_decoder_handles_odd_native_depth_and_high_side_padding():
    source = _NativeMedNeXtFixture()
    core = _PGAnisotropicMedNeXt(source, MEDNEXT_DOWN_STRIDES)
    phases = derive_mednext_feature_phases(core)
    assert tuple(phase[0] for phase in phases) == MEDNEXT_PHASE_STEPS
    # The PG teacher pads to (D multiple 2, H/W multiples 16), not an
    # isotropic 16-cube; decoder skip shapes must still close exactly.
    image = torch.rand(1, 1, 10, 32, 32)
    features, logits = core.forward_with_features(image)
    assert [tuple(value.shape[-3:]) for value in features] == [
        (10, 32, 32), (10, 16, 16), (10, 8, 8), (5, 4, 4), (5, 2, 2)
    ]
    assert logits.shape[-3:] == image.shape[-3:]


def test_anisotropic_core_to_eval_forward_backward_uses_native_module_apply():
    core = _PGAnisotropicMedNeXt(_NativeMedNeXtFixture(), MEDNEXT_DOWN_STRIDES)
    # ``nn.Module.to`` and ``eval`` must dispatch through the inherited
    # ``nn.Module._apply``; a helper with that name would break this path.
    core.to(device="cpu", dtype=torch.float32).eval()
    image = torch.rand(1, 1, 10, 32, 32, requires_grad=True)
    features, logits = core.forward_with_features(
        image,
        skip_dropout=1.0,
        dropout_generator=torch.Generator(device="cpu").manual_seed(42),
    )
    assert len(features) == 5
    assert logits.shape == (1, 4, 10, 32, 32)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert any(parameter.grad is not None for parameter in core.parameters())


def test_teacher_full_decoder_boundary_compatibility_and_freeze():
    teacher = _teacher()
    image = torch.rand(1, 1, 5, 16, 16)
    output = teacher(image, _geometry((5, 16, 16)), return_features=True)
    assert set(output) == {"region_logits", "boundary_logits", "boundary_logits4", "sdf", "sdf_all", "features"}
    assert output["region_logits"].shape == (1, 4, 5, 16, 16)
    assert output["boundary_logits"].shape == output["region_logits"].shape
    assert output["sdf"].shape == (1, 3, 5, 16, 16)
    assert output["sdf_all"].shape == output["region_logits"].shape
    assert len(output["features"]) == 5
    assert set(teacher.as_q3d_outputs(output)) == {"region_logits", "boundary_logits", "sdf"}
    labels = torch.zeros(1, 5, 16, 16, dtype=torch.long)
    loss, _ = q3d_teacher_loss(teacher.as_q3d_outputs(output), labels, torch.zeros_like(output["sdf_all"]))
    loss.backward()
    assert any(parameter.grad is not None for parameter in teacher.boundary_path.parameters())
    assert any(parameter.grad is not None for parameter in teacher.boundary_path.sdf_head.parameters())
    assert any(parameter.grad is not None for parameter in teacher.boundary_path.sdf_background_head.parameters())
    teacher.freeze()
    teacher.train(True)
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_pg_student_dropout_stage3_and_no_mimic():
    student = PGStudent(_teacher_config())
    assert not student.include_mimic
    assert len(student.mimic_heads) == 0
    image = torch.rand(1, 1, 5, 16, 16)
    geometry = _geometry((5, 16, 16))
    output = student(image, geometry, skip_dropout=1.0, return_stage3=True)
    assert output["logits"].shape == (1, 4, 5, 16, 16)
    assert output["stage3"].shape[1] == 128
    assert output["mimic_loss"].detach().item() == 0.0


def test_pg_student_code_branch_is_train_only_and_teacher_features_detached():
    class Branch(nn.Module):
        def forward(self, stage3, valid_mask=None):
            return stage3.mean(dim=1, keepdim=True)

    student = PGStudent(_teacher_config()).attach_code_branch(Branch())
    geometry = _geometry((5, 16, 16))
    image = torch.rand(1, 1, 5, 16, 16)
    features = [
        torch.rand(1, channels, depth, height, width, requires_grad=True)
        for channels, (depth, height, width) in zip(
            MEDNEXT_CHANNELS, ((5, 16, 16), (5, 8, 8), (5, 4, 4), (3, 2, 2), (3, 1, 1))
        )
    ]
    keep = torch.ones(1, 1, 1, 1, 1)
    output = student(image, geometry, teacher_features=features, keep_mask=keep, return_stage3=True, return_code=True)
    assert "code_branch" in output
    output["logits"].mean().backward()
    assert all(feature.grad is None for feature in features)
    student.eval()
    assert "code_branch" not in student(image, geometry)
