import torch

from self_audit_pseudolabel.system_v3 import (
    AdaptiveAnnotationStudent,
    CinePseudoTeacher,
    UNKNOWN,
    pseudo_supervision_loss,
)


def _cine(batch=1, h=32, w=32):
    torch.manual_seed(17)
    cur = torch.rand(batch, 3, h, w)
    prev = torch.roll(cur, -1, -1)
    nxt = torch.roll(cur, 1, -1)
    return prev, cur, nxt


def test_teacher_shapes_and_abstention_contract():
    teacher = CinePseudoTeacher(width=8, appearance_dim=16, motion_dim=8, fused_dim=24, k=8)
    prev, cur, nxt = _cine()
    out = teacher(prev, cur, nxt)
    assert out["pseudo_label"].shape == (1, 32, 32)
    assert out["soft_label"].shape == (1, 4, 32, 32)
    assert out["region_prob"].shape == (1, 8, 8, 8)
    assert set(torch.unique(out["pseudo_label"]).tolist()).issubset({0, 1, 2, 3, UNKNOWN})


def test_student_resource_profiles_change_compute_not_output_contract():
    student = AdaptiveAnnotationStudent(width=32, window_k=4).eval()
    _, cur, _ = _cine()
    with torch.no_grad():
        compact = student(cur, profile="compact")
        balanced = student(cur, profile="balanced")
    assert compact["a0_logits"].shape == (1, 4, 32, 32)
    assert compact["final_logits"].shape == (1, 4, 32, 32)
    assert len(compact["stages"]) == 1
    assert len(balanced["stages"]) == 2
    assert torch.equal(compact["a0_logits"], compact["final_logits"])


def test_pseudo_supervision_ignores_unknown_pixels():
    student = AdaptiveAnnotationStudent(width=32, window_k=4)
    _, cur, _ = _cine()
    out = student(cur, profile="compact")
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[:, 8:24, 8:24] = True
    loss = pseudo_supervision_loss(out, target, valid)
    assert torch.isfinite(loss)
    loss.backward()
