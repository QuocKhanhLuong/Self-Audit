"""SliceSSMStudent: budget, shapes, physical-step semantics, padding, alignment, shift supervision, export, dispatch."""
from __future__ import annotations

import pytest
import torch

from self_audit.candidate_pg import shape_atoms as sa
from self_audit.candidate_pg.slicessm_student import (
    PARAMETER_CAP, PhysZSSM, SliceAlign, SliceSSMStudent, _warp_inplane, apply_slice_shifts, build_slicessm_student,
    export_slicessm_student, load_slicessm_export, sample_slice_shifts, shift_recovery_loss,
)
from self_audit.candidate_pg.student_pg import PGStudent, build_student


def _geo(d, h, w, sz=10.0):
    return {"shape": (d, h, w), "spacing_zyx": [sz, 1.4, 1.4]}


def test_parameter_cap_and_no_z_downsampling():
    m = build_slicessm_student({})
    assert sum(p.numel() for p in m.parameters()) <= PARAMETER_CAP
    assert not hasattr(m, "z_transformer")
    assert [s[0] for s in m.phases] == [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 8), (1, 16, 16)]


@pytest.mark.parametrize("d", [1, 6, 13])
def test_shapes_for_depths_and_unaligned_inplane_size(d):
    m = SliceSSMStudent().eval()
    out = m(torch.rand(1, 1, d, 37, 53), _geo(d, 37, 53))
    assert out["logits"].shape == (1, 4, d, 37, 53) and torch.isfinite(out["logits"]).all()


def test_missing_spacing_is_an_error():
    with pytest.raises(ValueError, match="spacing"):
        SliceSSMStudent().eval()(torch.rand(1, 1, 4, 32, 32), {"shape": (4, 32, 32)})


def test_physical_step_depends_on_spacing_and_none_mode_does_not():
    torch.manual_seed(0)
    phys = PhysZSSM(16, spacing_mode="physical").eval()
    x = torch.randn(1, 16, 7, 3, 3)
    a, b = phys(x, torch.tensor([5.0])), phys(x, torch.tensor([10.0]))
    assert not torch.allclose(a, b)
    none = PhysZSSM(16, spacing_mode="none").eval()
    none.load_state_dict(phys.state_dict())
    assert torch.allclose(none(x, torch.tensor([5.0])), none(x, torch.tensor([10.0])))
    # physical mode at s_ref equals the none mode (factor 1)
    assert torch.allclose(phys(x, torch.tensor([10.0])), none(x, torch.tensor([7.0])), atol=1e-6)


def test_padded_slices_do_not_leak_into_valid_ones():
    torch.manual_seed(0)
    blk = PhysZSSM(16).eval()
    x = torch.randn(1, 16, 8, 2, 2)
    valid = torch.tensor([[True] * 5 + [False] * 3])
    x2 = x.clone()
    x2[:, :, 5:] = torch.randn_like(x2[:, :, 5:]) * 10
    a, b = blk(x, torch.tensor([8.0]), valid), blk(x2, torch.tensor([8.0]), valid)
    assert torch.allclose(a[:, :, :5], b[:, :, :5], atol=1e-5)
    assert not torch.allclose(blk(x, torch.tensor([8.0])), blk(x2, torch.tensor([8.0])))


def test_slice_align_is_identity_at_init():
    al = SliceAlign(16).eval()
    x = torch.randn(1, 16, 5, 6, 6)
    out, flow = al(x)
    assert torch.equal(out, x) and flow.abs().max() == 0 and flow.shape == (1, 2, 4, 6, 6)


def test_warp_matches_integer_shift():
    x = torch.zeros(1, 1, 8, 8)
    x[0, 0, 3, 4] = 1.0
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0], flow[:, 1] = 1.0, -2.0  # sample at (y+1, x-2)
    y = _warp_inplane(x, flow)
    assert y[0, 0, 2, 6] == 1.0 and y.sum() == 1.0


def test_apply_slice_shifts_moves_content_and_labels_identically():
    img = torch.zeros(1, 1, 2, 10, 10)
    img[0, 0, :, 4, 4] = 1.0
    lab = (img > 0).long()
    shifts = torch.tensor([[[0.0, 0.0], [2.0, -1.0]]])
    si, inside = apply_slice_shifts(img, shifts)
    sl, _ = apply_slice_shifts(lab, shifts, mode="nearest")
    assert si[0, 0, 0, 4, 4] == 1.0 and si[0, 0, 1, 6, 3] == 1.0
    assert sl.dtype == torch.long and sl[0, 0, 1, 6, 3] == 1 and sl.sum() == 2
    assert inside[0, 0, 1, :2].sum() == 0 and inside[0, 0, 0].all()


def test_shift_recovery_loss_is_zero_for_exact_offsets_and_positive_otherwise():
    shifts = torch.tensor([[[0.0, 0.0], [8.0, -16.0], [8.0, 0.0]]])
    clean = torch.randn(1, 2, 2, 4, 4)
    exact = clean.clone()
    exact[:, 0, 0] += 1.0; exact[:, 1, 0] -= 2.0; exact[:, 1, 1] += 2.0
    assert shift_recovery_loss(exact, clean, shifts, stride=8) < 1e-12
    assert shift_recovery_loss(clean, clean, shifts, stride=8) > 0.1


def test_sample_slice_shifts_range_and_determinism():
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    a, b = sample_slice_shifts(2, 9, 6, generator=g1), sample_slice_shifts(2, 9, 6, generator=g2)
    assert torch.equal(a, b) and a.shape == (2, 9, 2) and a.abs().max() <= 6


def test_gradients_reach_every_parameter_including_offset_head():
    torch.manual_seed(0)
    m = SliceSSMStudent().train()
    g = _geo(6, 32, 32)
    x = torch.rand(2, 1, 6, 32, 32)
    out = m(x, g)
    shifts = sample_slice_shifts(2, 6, 8, generator=torch.Generator().manual_seed(0))
    xs, _ = apply_slice_shifts(x, shifts)
    aux = shift_recovery_loss(m.slice_flows(xs, g), m.slice_flows(x, g), shifts)
    loss = out["logits"].mean() + out["sdf"].mean() + sum(a.mean() for a in out["aux_logits"]) + aux
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert all(n.startswith("stage3_tap") for n in dead), dead
    assert m.align3.offset[-1].weight.grad.abs().sum() > 0


def test_code_branch_contract_via_stage3_tap():
    cb = sa.ShapeAtomCodebook().freeze()
    m = SliceSSMStudent().train().attach_code_branch(sa.StudentAtomBranch(cb))
    out = m(torch.rand(1, 1, 4, 32, 32), _geo(4, 32, 32), return_stage3=True, return_code=True)
    assert out["stage3"].shape[1] == 128 and "code_branch" in out


def test_export_strips_train_only_heads_and_cold_load_matches(tmp_path):
    torch.manual_seed(0)
    m = SliceSSMStudent({"student": {"slicessm": {"spacing_mode": "none"}}}).eval()
    info = export_slicessm_student(m, tmp_path / "s.pt", epoch=3)
    loaded = load_slicessm_export(tmp_path / "s.pt")
    assert info["parameters_deployed"] == sum(p.numel() for p in loaded.parameters()) and loaded.spacing_mode == "none"
    x, g = torch.rand(1, 1, 5, 40, 36), _geo(5, 40, 36, 7.5)
    assert torch.allclose(m(x, g)["logits"], loaded(x, g)["logits"], atol=1e-6)


def test_build_student_dispatch_and_default_unchanged():
    assert isinstance(build_student({"student": {"arch": "slicessm"}}), SliceSSMStudent)
    assert type(build_student({})) is PGStudent


def test_rejects_teacher_feature_fusion():
    with pytest.raises(ValueError, match="fusion"):
        SliceSSMStudent()(torch.rand(1, 1, 2, 16, 16), _geo(2, 16, 16), teacher_features=[], keep_mask=torch.ones(1, 1, 1, 1, 1))


def test_shift_aux_config_and_deterministic_loss():
    from self_audit.candidate_pg.slicessm_student import shift_aux_config, slice_shift_aux
    assert shift_aux_config({}) is None
    assert shift_aux_config({"student": {"slicessm": {"shift_aux": {"weight": 0.0}}}}) is None
    assert shift_aux_config({"student": {"slicessm": {"shift_aux": {"weight": 0.1}}}}) == {"weight": 0.1, "max_px": 8, "start_update": 0}
    torch.manual_seed(0)
    m = SliceSSMStudent().train()
    with torch.no_grad():  # non-zero offsets so the loss is informative
        m.align3.offset[-1].weight.normal_(0, 0.01)
    pairs = [(torch.rand(1, 1, 5, 40, 40), _geo(5, 40, 40))]
    a = slice_shift_aux(m, pairs, seed=42, update=7, max_px=8)
    b = slice_shift_aux(m, pairs, seed=42, update=7, max_px=8)
    assert torch.equal(a, b) and a > 0
    a.backward()
    assert m.align3.offset[-1].weight.grad.abs().sum() > 0 and m.zssm4[0].out_proj.weight.grad is None
    assert slice_shift_aux(PGStudent({}), pairs, seed=42, update=7, max_px=8) is None


def test_ablated_student_imports_full_s1_artifact_dropping_only_removed_modules(tmp_path):
    from self_audit.candidate_pg.cinema_init import load_encoder_artifact
    full = SliceSSMStudent()
    enc = dict(full.state_dict())  # the real S1 artifact stores the whole student state
    path = tmp_path / "student_init.pt"
    torch.save({"schema": "self_audit.candidate_pg.cinema_init.v1", "model": enc}, path)
    ablated = SliceSSMStudent({"student": {"slicessm": {"align": False, "ssm_blocks": [0, 0]}}})
    info = load_encoder_artifact(path, ablated)
    assert info["dropped_ablated_keys"] == sum(1 for k in enc if k.startswith(("align3", "zssm")))
    assert torch.equal(ablated.e3[0].conv.weight, full.e3[0].conv.weight)
    enc["bogus.weight"] = torch.zeros(1)
    torch.save({"schema": "self_audit.candidate_pg.cinema_init.v1", "model": enc}, path)
    with pytest.raises(ValueError, match="unexpected"):
        load_encoder_artifact(path, ablated)


def test_replicate_seeds_validate_and_others_are_rejected(tmp_path):
    from self_audit.candidate_pg import engine
    from self_audit.candidate_pg.config import REPLICATE_SEEDS, default_config, validate_config

    assert REPLICATE_SEEDS == engine.TRAIN_SEEDS == (42, 43, 44)
    for seed in (42, 43, 44):
        cfg = default_config()
        cfg["seed"] = seed
        assert validate_config(cfg)["seed"] == seed
    cfg = default_config()
    cfg["seed"] = 45
    with pytest.raises(ValueError):
        validate_config(cfg)
    with pytest.raises(ValueError, match="seed"):
        engine.run_training(cfg, tmp_path, stage="arms")
