from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from self_audit.candidate_pg.codebook import StudentCodeBranch, TeacherCodebook
from self_audit.candidate_pg.reliability import ClassAdaptiveThreshold
from self_audit.candidate_pg.student_pg import PGStudent
from self_audit.candidate_pg.training_step import student_training_step

torch.set_num_threads(1)


@dataclass
class _Frame:
    image: torch.Tensor
    target: torch.Tensor | None
    geometry: dict
    patient_id: str


_PHASES = (
    ((1, 1, 1), (0.0, 0.0, 0.0)),
    ((1, 2, 2), (0.0, 0.0, 0.0)),
    ((1, 4, 4), (0.0, 0.0, 0.0)),
    ((2, 8, 8), (0.0, 0.0, 0.0)),
    ((2, 16, 16), (0.0, 0.0, 0.0)),
)


class _Teacher(nn.Module):
    feature_phases = _PHASES
    teacher_phases = _PHASES

    def __init__(self, *, p_class: int = 0, strength: float = 4.0):
        super().__init__()
        self.proj = nn.Conv3d(1, 4, 1)
        self.p_class = p_class
        self.strength = float(strength)

    def forward(self, image, geometry, *, return_features=True):
        b, _, d, h, w = image.shape
        logits = image.new_full((b, 4, d, h, w), -self.strength)
        logits[:, self.p_class] = self.strength
        channels = (32, 64, 128, 256, 512)
        features = []
        for channels_i, (step, _offset) in zip(channels, _PHASES):
            shape = tuple(max(1, (int(image.shape[2 + axis]) + step[axis] - 1) // step[axis]) for axis in range(3))
            features.append(image.new_zeros((b, channels_i, *shape)))
        sdf = image.new_zeros((b, 3, d, h, w))
        return {"region_logits": logits, "sdf": sdf, "features": tuple(features)}


class _SpyStudent(nn.Module):
    """Small gradient-probe student that distinguishes fusion from clean paths."""

    phases = _PHASES

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))
        self.fusion_calls = 0
        self.teacher_feature_calls = 0
        self.keep_masks = []

    def forward(self, image, geometry, *, teacher_features=None, keep_mask=None,
                return_stage3=False, return_code=False, skip_dropout=0.0,
                dropout_generator=None):
        if teacher_features is not None:
            self.teacher_feature_calls += 1
            self.keep_masks.append(None if keep_mask is None else keep_mask.detach().clone())
            # Fusion weak target deliberately predicts class 1 while the
            # frozen teacher predicts class 0; y must come from p_W.
            logits = image.new_zeros((image.shape[0], 4, *image.shape[-3:]))
            logits[:, 1] = 5.0 + self.logit
            gate = [0.0] * 5
        else:
            self.fusion_calls += 1
            logits = image.new_zeros((image.shape[0], 4, *image.shape[-3:]))
            logits[:, 0] = 5.0 + self.logit
            gate = [0.0] * 5
        result = {"logits": logits, "gate_means": gate, "features": (image,) * 5}
        if return_stage3:
            result["stage3"] = image.repeat(1, 128, 1, 1, 1)
        return result


def _config(*, pg: bool, cbkd: bool, warmup: int = 800, n_views: int = 1) -> dict:
    return {
        "student": {"teacher_dropout_on_labeled": {"start": 0.0, "end": 0.5}, "fp_skip_dropout": 0.5},
        "training": {"updates": 8000},
        "augmentation": {
            "n_strong_views": n_views,
            "cutmix": {"area_fraction": [0.25, 0.5], "align_px": 1, "full_depth": True},
        },
        "pg": {
            "enabled": pg,
            "stream_weights": {"s1": 0.25, "s2": 0.25, "fp": 0.5},
            "lambda_u": {"max": 1.0, "warmup_updates_labeled_only": warmup, "ramp_fraction": 0.1},
        },
        "cbkd": {"enabled": cbkd, "warmup_updates_labeled_only": warmup, "lambda_cb": 0.5,
                 "weights": {"assign": 1.0, "qtask": 1.0, "commit": 0.25, "qpred": 0.0}},
    }


def _frames(shape=(2, 4, 4)):
    image = torch.rand(1, 1, *shape)
    target = torch.zeros((1, *shape), dtype=torch.long)
    target[..., 1:-1, 1:-1] = 1
    geometry = {"shape": tuple(shape), "spacing_zyx": (1.0, 1.0, 1.0)}
    return (
        _Frame(image, target, geometry, "L"),
        _Frame(image, None, geometry, "A"),
        _Frame(image, None, geometry, "B"),
    )


def _views(frames, *, n_views=1, box=(0, 2, 0, 2)):
    l, a, b = frames
    valid = torch.ones((1, *l.image.shape[-3:]), dtype=torch.bool)
    return (
        {"weak": l.image, "strong": l.image, "target": l.target, "valid": valid, "geometry": l.geometry},
        {"a": {"weak": a.image, "strong": a.image, "strong2": a.image.clone(), "valid": valid, "geometry": a.geometry, "box": box},
         "b": {"weak": b.image, "strong": b.image, "strong2": b.image + 0.01, "valid": valid, "geometry": b.geometry}},
    )


def _call(cfg, student, teacher, frames, views, *, update, threshold):
    lv, uv = views
    return student_training_step(
        cfg, student, teacher, None, *frames, update, threshold,
        labeled_views=lv, u_views=uv,
        keep_generator=torch.Generator(device="cpu").manual_seed(7),
        fp_generator=torch.Generator(device="cpu").manual_seed(11),
    )


def test_warmup_is_labeled_only_and_diagnostics_are_detached(tmp_path):
    del tmp_path
    frames = _frames()
    student, teacher = _SpyStudent(), _Teacher()
    result = _call(_config(pg=True, cbkd=False), student, teacher, frames, _views(frames), update=0, threshold=ClassAdaptiveThreshold(4))
    assert result["loss_pg"].detach().item() == 0.0
    assert result["loss_cbkd"].detach().item() == 0.0
    assert student.teacher_feature_calls == 1  # labelled fusion only
    assert all(not value.requires_grad for value in result["diagnostics"].values() if isinstance(value, torch.Tensor))


def test_pg_uses_fusion_weak_target_and_never_backpropagates_through_gates(tmp_path):
    del tmp_path
    frames = _frames()
    student, teacher = _SpyStudent(), _Teacher(p_class=0)
    cfg = _config(pg=True, cbkd=False, warmup=0)
    result = _call(cfg, student, teacher, frames, _views(frames), update=0, threshold=ClassAdaptiveThreshold(4))
    assert torch.equal(result["diagnostics"]["pW_target_a"], torch.ones_like(result["diagnostics"]["pW_target_a"]))
    assert torch.equal(result["diagnostics"]["teacher_target_a"], torch.zeros_like(result["diagnostics"]["teacher_target_a"]))
    result["loss_pg"].backward()
    assert student.logit.grad is not None and torch.isfinite(student.logit.grad)


def test_threshold_update_uses_fusion_pW_known_answer(tmp_path):
    del tmp_path
    frames = _frames()
    student, teacher = _SpyStudent(), _Teacher(strength=0.0)
    threshold = ClassAdaptiveThreshold(4, ema=0.5, clip=(0.0, 1.0))
    _call(_config(pg=True, cbkd=False, warmup=0), student, teacher, frames, _views(frames), update=0, threshold=threshold)
    p_w = torch.softmax(torch.tensor([0.0, 5.0, 0.0, 0.0]), dim=0)
    expected_tau_g = 0.5 * 0.25 + 0.5 * float(p_w.max())
    assert float(threshold.tau_g) == pytest.approx(expected_tau_g, rel=0.0, abs=1e-6)
    assert float(threshold.p_tilde[1]) > float(threshold.p_tilde[0])


def test_pg_mixes_unequal_native_partner_after_explicit_packet_crop_pad(tmp_path):
    del tmp_path
    shape_a, shape_b = (2, 4, 4), (3, 6, 5)
    image_l = torch.rand(1, 1, *shape_a)
    target_l = torch.zeros((1, *shape_a), dtype=torch.long)
    image_a = torch.rand(1, 1, *shape_a)
    image_b = torch.rand(1, 1, *shape_b)
    geom_a = {"shape": shape_a, "spacing_zyx": (1.0, 1.0, 1.0)}
    geom_b = {"shape": shape_b, "spacing_zyx": (1.0, 1.0, 1.0)}
    frames = (_Frame(image_l, target_l, geom_a, "L"), _Frame(image_a, None, geom_a, "A"), _Frame(image_b, None, geom_b, "B"))
    valid_a = torch.ones((1, *shape_a), dtype=torch.bool)
    valid_b = torch.ones((1, *shape_b), dtype=torch.bool)
    labeled = {"weak": image_l, "strong": image_l, "target": target_l, "valid": valid_a, "geometry": geom_a}
    unlabeled = {
        "a": {"weak": image_a, "strong": image_a, "valid": valid_a, "geometry": geom_a, "box": (0, 2, 0, 2)},
        "b": {"weak": image_b, "strong": image_b, "valid": valid_b, "geometry": geom_b},
    }
    result = student_training_step(
        _config(pg=True, cbkd=False, warmup=0), _SpyStudent(), _Teacher(), None, *frames, 0, ClassAdaptiveThreshold(4),
        labeled_views=labeled, u_views=unlabeled,
        keep_generator=torch.Generator().manual_seed(7), fp_generator=torch.Generator().manual_seed(11),
    )
    assert torch.isfinite(result["loss"])
    assert tuple(result["diagnostics"]["pW_target_a"].shape) == (1, *shape_a)


def test_teacher_targets_and_learning_streams_are_gradient_isolated(tmp_path):
    del tmp_path
    frames = _frames()
    student, teacher = _SpyStudent(), _Teacher()
    result = _call(_config(pg=True, cbkd=False, warmup=0), student, teacher, frames, _views(frames), update=0, threshold=ClassAdaptiveThreshold(4))
    result["loss"].backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    # L fusion + U fusion A/B are the only calls carrying teacher features;
    # strong s1/s2 and weak-A FP are clean student-only calls.
    assert student.teacher_feature_calls == 3


def test_labeled_keep_rng_is_switch_independent(tmp_path):
    del tmp_path
    frames = _frames()
    views = _views(frames)
    student_a, student_b = _SpyStudent(), _SpyStudent()
    teacher_a, teacher_b = _Teacher(), _Teacher()
    _call(_config(pg=False, cbkd=False, warmup=800), student_a, teacher_a, frames, views, update=0, threshold=ClassAdaptiveThreshold(4))
    _call(_config(pg=True, cbkd=True, warmup=800), student_b, teacher_b, frames, views, update=0, threshold=ClassAdaptiveThreshold(4))
    assert len(student_a.keep_masks) == len(student_b.keep_masks) == 1
    torch.testing.assert_close(student_a.keep_masks[0], student_b.keep_masks[0])


def test_actual_pg_student_labeled_gates_get_gradients_but_pg_does_not(tmp_path):
    del tmp_path
    shape = (2, 16, 16)
    frames = _frames(shape)
    cfg = {
        "model": {"student": {"channels": [16, 32, 64, 128, 256], "num_classes": 4}},
        **_config(pg=True, cbkd=False, warmup=0, n_views=1),
    }
    student = PGStudent(cfg)
    teacher = _Teacher()
    lv, uv = _views(frames, box=(0, 16, 0, 16))
    result = student_training_step(cfg, student, teacher, None, *frames, 0, ClassAdaptiveThreshold(4),
                                   labeled_views=lv, u_views=uv,
                                   keep_generator=torch.Generator().manual_seed(7),
                                   fp_generator=torch.Generator().manual_seed(11))
    gate_params = [parameter for module in student.fusions for parameter in module.parameters()]
    pg_grads = torch.autograd.grad(result["loss_pg"], gate_params, allow_unused=True)
    assert all(value is None or torch.equal(value, torch.zeros_like(value)) for value in pg_grads)
    sup_grads = torch.autograd.grad(result["loss_sup"], gate_params, allow_unused=True)
    assert any(value is not None and torch.any(value != 0) for value in sup_grads)


def test_cbkd_codebook_is_frozen_and_clean_a_only(tmp_path):
    del tmp_path
    shape = (2, 16, 16)
    frames = _frames(shape)
    cfg = {
        "model": {"student": {"channels": [16, 32, 64, 128, 256], "num_classes": 4}},
        **_config(pg=False, cbkd=True, warmup=0, n_views=1),
    }
    student = PGStudent(cfg)
    teacher = _Teacher()
    teacher_codebook = TeacherCodebook({"teacher_channels": 256, "embedding_dim": 64, "codebook_size": 2})
    teacher_codebook.freeze()
    student.attach_code_branch(StudentCodeBranch({"student_channels": 128, "embedding_dim": 64}, teacher_codebook))
    lv, uv = _views(frames, box=(0, 16, 0, 16))
    result = student_training_step(cfg, student, teacher, teacher_codebook, *frames, 0, ClassAdaptiveThreshold(4),
                                   labeled_views=lv, u_views=uv,
                                   keep_generator=torch.Generator().manual_seed(7),
                                   fp_generator=torch.Generator().manual_seed(11))
    result["loss_cbkd"].backward()
    assert any(parameter.grad is not None and torch.any(parameter.grad != 0) for parameter in student.code_branch.P_S.parameters())
    assert all(parameter.grad is None for parameter in teacher_codebook.parameters())
    assert result["diagnostics"]["code_perplexity"] is not None
    assert result["diagnostics"]["code_dead"] is not None
    assert all(value is None or torch.equal(value, torch.zeros_like(value)) for value in
               torch.autograd.grad(result["loss_cbkd"], [p for m in student.fusions for p in m.parameters()], allow_unused=True))
