import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from self_audit.candidate_fuse.augmentation import StrongInplaneAugment
from self_audit.candidate_fuse.engine import (
    _restore_rng, _rng_state, joint_loss, prepare_teacher_batch,
)


GEOMETRY = {"shape": (2, 8, 8), "spacing_zyx": (1., 1., 1.),
    "affine_zyx": torch.eye(4).tolist(), "context": "full_volume"}


def _frames():
    image = torch.zeros((1, 1, 2, 8, 8), dtype=torch.float32)
    image[:, :, :, 2:6, 2:6] = .8
    target = torch.zeros((1, 2, 8, 8), dtype=torch.long)
    target[:, :, 2:6, 2:6] = 1
    return (SimpleNamespace(image=image, target=target, geometry=dict(GEOMETRY),
                patient_id="L", frame_index=0),
            SimpleNamespace(image=image.clone(), target=None, geometry=dict(GEOMETRY),
                patient_id="U", frame_index=1))


def test_run_seed_42_matches_augmentor_contract_and_t_batch2_draws_independently():
    augmentor = StrongInplaneAugment(42)
    expected = StrongInplaneAugment(42)
    assert torch.equal(augmentor.generator.get_state(), expected.generator.get_state())
    labeled, _ = _frames()
    second = SimpleNamespace(image=labeled.image.clone(), target=labeled.target.clone(),
        geometry=dict(GEOMETRY), patient_id="L", frame_index=1)
    augmentor = StrongInplaneAugment(42)
    image, target, geometry = prepare_teacher_batch([labeled, second], "cpu",
        py_rng=random.Random(42), augmentation=augmentor)
    assert image.shape == (2, 1, 2, 8, 8) and target.shape == (2, 2, 8, 8)
    assert geometry["shape"] == GEOMETRY["shape"]
    # Two sequential draws must advance the one private stream independently.
    expected = StrongInplaneAugment(42)
    first_expected, first_target, first_geometry = expected.teacher_view(
        labeled.image, labeled.target, dict(GEOMETRY))
    second_expected, second_target, second_geometry = expected.teacher_view(
        second.image, second.target, dict(GEOMETRY))
    torch.testing.assert_close(image, torch.cat((first_expected, second_expected)))
    torch.testing.assert_close(target, torch.cat((first_target, second_target)))
    assert first_geometry == second_geometry == geometry


def test_actual_strong_augmentor_joint_teacher_and_student_views_share_spatial():
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__(); self.p = nn.Parameter(torch.tensor(.2), requires_grad=False)
            self.views = []
        def forward(self, image, geometry, *, return_features=True):
            assert not torch.is_grad_enabled()
            self.views.append(image.clone())
            logits = self.p * torch.ones((1, 4, *image.shape[-3:]))
            return {"region_logits": logits, "features": tuple(image for _ in range(5))}

    class Student(nn.Module):
        def __init__(self): super().__init__(); self.p=nn.Parameter(torch.tensor(.1)); self.views=[]
        def forward(self, image, geometry, *, teacher_features=None, keep_mask=None):
            self.views.append(image.clone())
            logits = self.p * torch.ones((1, 4, *image.shape[-3:]))
            return {"logits": logits, "mimic_loss": logits.sum()*0,
                    "gate_means": [0.]*5}

    labeled, unlabeled = _frames()
    p = {key: 0. for key in StrongInplaneAugment(42).probabilities}
    p.update(flip_y=1., gamma=1., brightness=1.)
    augmentor = StrongInplaneAugment(42, probabilities=p)
    expected = StrongInplaneAugment(42, probabilities=p)
    t_l = expected.spatial(labeled.image, labeled.target, dict(GEOMETRY))[0]
    t_u = expected.spatial(unlabeled.image, None, dict(GEOMETRY))[0]
    teacher, student = Teacher().eval(), Student()
    from self_audit.candidate_fuse.config import default_config
    loss, _ = joint_loss(student, teacher, labeled, unlabeled, update=1, total_updates=2,
        config=default_config(synthetic=True), generator=torch.Generator().manual_seed(42),
        augmentation=augmentor)
    torch.testing.assert_close(teacher.views[0], t_l)
    torch.testing.assert_close(teacher.views[1], t_u)
    assert not torch.equal(student.views[0], teacher.views[0])
    assert not torch.equal(student.views[1], teacher.views[1])
    loss.backward()
    assert student.p.grad is not None and teacher.p.grad is None


def test_enabled_augmentation_resume_restores_real_augmentor_exactly():
    augmentor = StrongInplaneAugment(42)
    py_rng = random.Random(42)
    generator = torch.Generator().manual_seed(42)
    state = _rng_state(generator, py_rng, augmentor)
    frame, _ = _frames()
    expected = augmentor.joint_views(frame.image, frame.target, dict(GEOMETRY))
    augmentor.joint_views(frame.image, frame.target, dict(GEOMETRY))
    _restore_rng(state, generator, py_rng, augmentor)
    actual = augmentor.joint_views(frame.image, frame.target, dict(GEOMETRY))
    for a, b in zip(actual[:3], expected[:3]):
        if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    assert actual[3] == expected[3]
    missing = {key: value for key, value in state.items() if key != "strong_augmentation"}
    with pytest.raises(ValueError, match="strong_augmentation"):
        _restore_rng(missing, generator, py_rng, StrongInplaneAugment(42))
