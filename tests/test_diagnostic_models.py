import torch
import pytest
from torch import nn

from self_audit.bridge_alignment import BridgeAlignmentConfig, BridgeFitSample, fit_bridge_alignment
from self_audit.compact_model import CompactStudent
from self_audit.diagnostic_models import (
    CompactStudentTap,
    TeacherSuffixBridge,
    check_bridge_h_dependence,
    extract_teacher_pre_refine0,
    feature_support_from_valid,
)
from self_audit.model import NativeStudent


class TinyNativeEncoder(nn.Module):
    channels = (8, 8, 8, 8)

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 8, 3, stride=4, padding=1),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),
        ])

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


class HistoricalCompactAlias(nn.Module):
    """Strict-loader-style class alias exposing the compact model contract."""

    def __init__(self, source):
        super().__init__()
        self.encoder = source.encoder
        self.decoder = source.decoder
        self.head = source.head
        self.widths = source.widths
        self.in_channels = source.in_channels

    def _pad_geometry(self, x):
        h, w = x.shape[-2:]
        stride = 2 ** (len(self.widths) - 1)
        padded = torch.nn.functional.pad(x, (0, (-w) % stride, 0, (-h) % stride), mode="replicate")
        return padded, h, w

    def forward(self, x, valid_mask=None):
        padded, h, w = self._pad_geometry(x)
        return self.head(self.decoder(self.encoder(padded))[..., :h, :w])


HistoricalCompactAlias.__module__ = "_gradient_bridge_historical_0f54455b.compact_model"


class HistoricalNativeStudentAlias(nn.Module):
    """Historical NativeStudent module layout without a teacher_memory field."""

    def __init__(self, source):
        super().__init__()
        for name in ("input_adapter", "encoder", "lateral", "up", "refine", "detail", "fuse", "head"):
            setattr(self, name, getattr(source, name))
        self.window = None


HistoricalNativeStudentAlias.__module__ = "_gradient_bridge_historical_0f54455b.model"


def _state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _assert_state(before, model):
    after = model.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name


def _alignment(student, teacher, tap, train_inputs):
    samples = []
    for index, x in enumerate(train_inputs):
        student_context = tap.extract(x)
        teacher_target = extract_teacher_pre_refine0(teacher, x)
        samples.append(BridgeFitSample(
            student_h=student_context.h,
            teacher_pre_refine0=teacher_target,
            valid_mask=torch.ones((1, *teacher_target.shape[-2:]), dtype=torch.bool),
            patient_id=f"train-patient-{index}",
            source_id=f"train-patient-{index}/frame-0/slice-{index}",
            split="TRAIN",
        ))
    return fit_bridge_alignment(
        samples,
        BridgeAlignmentConfig(max_source_ids=2, max_train_patients=2, max_pixels=256, ridge=1e-3),
        {"student_sha256": "a" * 64, "teacher_sha256": "b" * 64,
         "train_manifest_sha256": "c" * 64, "source_commit": "fixture"},
    )


@pytest.fixture
def strict_loaded_historical_aliases():
    torch.manual_seed(301)
    current_student = CompactStudent(widths=(8, 16, 24, 32)).eval()
    current_teacher = NativeStudent(TinyNativeEncoder(), widths=(8, 8, 8, 8)).eval()
    return (
        HistoricalCompactAlias(current_student).eval(),
        current_student,
        HistoricalNativeStudentAlias(current_teacher).eval(),
        current_teacher,
    )


def test_compact_bottleneck_tap_and_native_decoder_d0_are_exact_on_odd_geometry():
    torch.manual_seed(13)
    model = CompactStudent(widths=(8, 16, 24, 32)).eval()
    tap = CompactStudentTap(model)
    x = torch.rand(2, 3, 31, 37)
    valid = torch.ones((2, 31, 37), dtype=torch.bool)
    before, rng_before = _state(model), torch.get_rng_state().clone()
    context = tap.extract(x, valid_mask=valid)
    logits = tap.decode(context)
    expected = model(x, valid_mask=valid)

    assert context.h.shape == (2, 32, 4, 5)
    assert len(context.skips) == 3
    assert all(not skip.requires_grad for skip in context.skips)
    assert not context.h.requires_grad
    assert context.input_hw == (31, 37)
    assert context.padded_hw == (32, 40)
    assert logits.shape == (2, 4, 31, 37)
    assert torch.equal(logits, expected)
    assert torch.equal(rng_before, torch.get_rng_state())
    _assert_state(before, model)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_tap_refuses_training_mode_without_mutating_mode():
    model = CompactStudent(widths=(8, 16, 24, 32)).train()
    with pytest.raises(ValueError, match="must already be in eval mode"):
        CompactStudentTap(model)
    assert model.training is True


def test_feature_support_uses_context_padded_canvas_for_odd_native_shapes():
    model = CompactStudent(widths=(8, 16, 24, 32)).eval()
    tap = CompactStudentTap(model)
    x = torch.rand(1, 3, 33, 35)
    valid = torch.ones((1, 33, 35), dtype=torch.bool)
    context = tap.extract(x, valid)
    support = feature_support_from_valid(context.valid_mask, context.padded_hw, context.h)
    assert context.padded_hw == (40, 40)
    assert context.h.shape[-2:] == (5, 5)
    assert int(support.sum()) == 16
    assert support[0, :4, :4].all()
    assert not support[0, 4].any()
    assert not support[0, :, 4].any()


def test_historical_alias_modules_are_accepted_by_structural_protocols(strict_loaded_historical_aliases):
    student, current_student, teacher, current_teacher = strict_loaded_historical_aliases
    assert not isinstance(student, CompactStudent)
    assert not isinstance(teacher, NativeStudent)
    assert not hasattr(teacher, "teacher_memory")

    tap = CompactStudentTap(student)
    train_inputs = [torch.rand(1, 3, 29, 35), torch.rand(1, 3, 29, 35)]
    context = tap.extract(train_inputs[0])
    assert torch.equal(tap.decode(context), student(train_inputs[0]))
    target_alias = extract_teacher_pre_refine0(teacher, train_inputs[0])
    target_current = extract_teacher_pre_refine0(current_teacher, train_inputs[0])
    assert torch.equal(target_alias, target_current)

    artifact = _alignment(student, teacher, tap, train_inputs)
    bridge = TeacherSuffixBridge(teacher, artifact)
    logits = bridge(context.h, context.input_hw)
    assert logits.shape == (1, 4, 29, 35)
    assert torch.isfinite(logits).all()


def test_teacher_suffix_uses_h_verified_p_and_zero_detail_without_full_decoder_bypass():
    torch.manual_seed(22)
    student = CompactStudent(widths=(8, 16, 24, 32)).eval()
    tap = CompactStudentTap(student)
    teacher = NativeStudent(TinyNativeEncoder(), widths=(8, 8, 8, 8)).eval()
    train_inputs = [torch.rand(1, 3, 29, 35), torch.rand(1, 3, 29, 35)]
    artifact = _alignment(student, teacher, tap, train_inputs)
    assert artifact.metadata["target_tap"] == "NativeStudent.pre_refine[0] input"
    assert artifact.metadata["raw_detail_conditioning"] is False
    teacher_before = _state(teacher)
    for parameter in teacher.parameters():
        parameter.grad = torch.full_like(parameter, 0.037)
    teacher_grads_before = [parameter.grad.clone() for parameter in teacher.parameters()]
    with pytest.raises(ValueError, match="identity mismatch: teacher_sha256"):
        TeacherSuffixBridge(teacher, artifact, expected_identities={"teacher_sha256": "d" * 64})
    rng_before = torch.get_rng_state().clone()
    bridge = TeacherSuffixBridge(teacher, artifact, expected_identities={
        "student_sha256": "a" * 64,
        "teacher_sha256": "b" * 64,
        "train_manifest_sha256": "c" * 64,
    })
    assert torch.equal(rng_before, torch.get_rng_state())
    assert not bridge.training

    detail_calls = []
    refine0_calls = []
    full_decoder_calls = []
    detail_hook = teacher.detail.register_forward_hook(lambda *_: detail_calls.append(True))
    refine0_hook = teacher.refine[0].register_forward_hook(lambda *_: refine0_calls.append(True))
    later_refine_hooks = [teacher.refine[index].register_forward_hook(lambda *_: full_decoder_calls.append(True))
                          for index in (1, 2, 3)]
    x = torch.cat([torch.rand(1, 3, 29, 35), torch.rand(1, 3, 29, 35)], dim=0)
    context = tap.extract(x)
    logits = bridge(context.h, context.input_hw)
    detail_hook.remove()
    refine0_hook.remove()
    for hook in later_refine_hooks:
        hook.remove()
    assert logits.shape == (2, 4, 29, 35)
    assert detail_calls == []
    assert refine0_calls == [True]
    assert full_decoder_calls == []
    _assert_state(teacher_before, teacher)

    sensitivity_rng_before = torch.get_rng_state().clone()
    dependence = check_bridge_h_dependence(
        bridge, context.h, context.input_hw, ("patient-a", "patient-b"), seed=91,
    )
    assert torch.equal(sensitivity_rng_before, torch.get_rng_state())
    assert set(dependence.donor_indices) == {0, 1}
    assert dependence.donor_indices[0] != 0 and dependence.donor_indices[1] != 1
    assert dependence.keep_vs_zero_mean_abs.shape == (2,)
    assert dependence.keep_vs_shuffle_mean_abs.shape == (2,)
    assert dependence.h_vjp_norm.shape == (2,)
    assert torch.isfinite(dependence.h_vjp_norm).all()
    assert torch.all(dependence.keep_vs_zero_mean_abs > 0)
    assert torch.all(dependence.keep_vs_shuffle_mean_abs > 0)
    assert torch.all(dependence.h_vjp_norm > 0)
    _assert_state(teacher_before, teacher)
    for parameter, grad_before in zip(teacher.parameters(), teacher_grads_before, strict=True):
        assert torch.equal(parameter.grad, grad_before)


def test_random_projector_and_non_native_teacher_graph_are_rejected():
    teacher = NativeStudent(TinyNativeEncoder(), widths=(8, 8, 8, 8)).eval()
    try:
        TeacherSuffixBridge(teacher, nn.Conv2d(32, 8, 1))
    except TypeError as error:
        assert "random P is rejected" in str(error)
    else:
        raise AssertionError("unfitted random adapter was accepted")


def test_alignment_target_is_the_exact_native_pre_refine0_cut():
    torch.manual_seed(8)
    teacher = NativeStudent(TinyNativeEncoder(), widths=(8, 8, 8, 8)).eval()
    x = torch.rand(1, 3, 27, 34)
    target = extract_teacher_pre_refine0(teacher, x)
    h, w = x.shape[-2:]
    pad = (0, (-w) % 32, 0, (-h) % 32)
    norm = torch.nn.functional.pad(teacher.input_adapter(x), pad, mode="replicate")
    stages = teacher.encoder(norm)
    d = teacher.refine[3](teacher.lateral[3](stages[3]))
    for index in (2, 1):
        skip = teacher.lateral[index](stages[index])
        d = teacher.refine[index](skip + teacher.up[index](
            torch.nn.functional.interpolate(d, size=skip.shape[-2:], mode="bilinear", align_corners=False)))
    skip0 = teacher.lateral[0](stages[0])
    expected = skip0 + teacher.up[0](
        torch.nn.functional.interpolate(d, size=skip0.shape[-2:], mode="bilinear", align_corners=False))
    assert torch.equal(target, expected)
