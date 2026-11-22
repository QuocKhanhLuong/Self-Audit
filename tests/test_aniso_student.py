"""AnisoStudent: params, shapes, z-attention masking, gradients, export/cold load, and bit-identical default student."""
from __future__ import annotations

import pytest
import torch

from self_audit.candidate_pg import shape_atoms as sa
from self_audit.candidate_pg.aniso_student import AnisoStudent, ZTransformer, export_aniso_student, load_aniso_export, param_breakdown
from self_audit.candidate_pg.config import default_config
from self_audit.candidate_pg.student_pg import PGStudent, build_student


def _geo(d, h, w):
    return {"shape": (d, h, w), "spacing_zyx": [10.0, 1.4, 1.4]}


def test_parameter_budget_and_no_z_downsampling():
    m = AnisoStudent()
    assert param_breakdown(m)["total"] <= 4_000_000
    assert all(s[0] == (1, s[0][1], s[0][2]) or s[0][0] == 1 for s in m.phases)  # z step stays 1 at every stage
    assert [s[0][1] for s in m.phases] == [1, 2, 4, 8, 16]


@pytest.mark.parametrize("d", [6, 10, 18])
def test_shapes_for_different_depths_and_unaligned_inplane_size(d):
    m = AnisoStudent().eval()
    x = torch.rand(1, 1, d, 37, 53)
    out = m(x, _geo(d, 37, 53))
    assert out["logits"].shape == (1, 4, d, 37, 53) and len(out["features"]) == 5 and torch.isfinite(out["logits"]).all()


def test_z_transformer_ignores_padded_slices():
    zt = ZTransformer(32, layers=2, heads=4).eval()
    x = torch.randn(1, 32, 8, 3, 3)
    valid = torch.tensor([[True] * 5 + [False] * 3])
    a = zt(x, valid)
    x2 = x.clone()
    x2[:, :, 5:] = torch.randn_like(x2[:, :, 5:]) * 10  # change only the padded slices
    b = zt(x2, valid)
    assert torch.allclose(a[:, :, :5], b[:, :, :5], atol=1e-5) and a[:, :, 5:].abs().max() == 0
    assert not torch.allclose(zt(x), zt(x2))  # without the mask the padded slices do leak


def test_gradient_reaches_every_stage_and_train_only_outputs_exist():
    m = AnisoStudent().train()
    out = m(torch.rand(2, 1, 6, 32, 32), _geo(6, 32, 32))
    assert out["sdf"].shape == (2, 3, 6, 32, 32) and len(out["aux_logits"]) == 2
    loss = out["logits"].mean() + out["sdf"].mean() + sum(a.mean() for a in out["aux_logits"])
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert dead == [] or all(n.startswith("stage3_tap") for n in dead), dead


def test_code_branch_contract_via_stage3_tap():
    cb = sa.ShapeAtomCodebook().freeze()
    m = AnisoStudent().train().attach_code_branch(sa.StudentAtomBranch(cb))
    out = m(torch.rand(1, 1, 4, 32, 32), _geo(4, 32, 32), return_stage3=True, return_code=True)
    assert out["stage3"].shape[1] == 128 and "code_branch" in out
    out["code_branch"]["z"].sum().backward()
    assert m.stage3_tap.weight.grad is not None


def test_export_strips_train_only_modules_and_cold_loads_identically(tmp_path):
    m = AnisoStudent().eval()
    m.attach_code_branch(sa.StudentAtomBranch(sa.ShapeAtomCodebook().freeze()))
    info = export_aniso_student(m, tmp_path / "a.pt", epoch=3)
    dep = load_aniso_export(tmp_path / "a.pt")
    assert not any(k.startswith(("sdf_head", "aux_heads", "stage3_tap", "code_branch")) for k in dep.state_dict())
    assert info["parameters_deployed"] < param_breakdown(m)["total"]
    x = torch.rand(1, 1, 5, 32, 32)
    with torch.no_grad():
        assert torch.allclose(m(x, _geo(5, 32, 32))["logits"], dep(x, _geo(5, 32, 32))["logits"], atol=1e-5)


def test_default_arch_is_unchanged_and_aniso_is_selected_by_config():
    cfg = default_config(synthetic=True)
    torch.manual_seed(0)
    a = build_student(cfg)
    torch.manual_seed(0)
    b = PGStudent(cfg)
    assert type(a) is PGStudent and all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    cfg2 = dict(cfg)
    cfg2["student"] = dict(cfg["student"], arch="aniso")
    assert isinstance(build_student(cfg2), AnisoStudent)


def test_screening_arm_configs_differ_only_in_architecture_cinema_revision_and_epochs(tmp_path):
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_w2s_ssl4mis as ssl
    import w2s_make_configs_ssl4mis as mk

    out, lock_path = ssl._synthetic_prepared(tmp_path)

    def flat(x, p=""):
        if isinstance(x, dict):
            r = {}
            for k, v in x.items():
                r.update(flat(v, f"{p}/{k}"))
            return r
        return {p: json.dumps(x)}

    s0 = flat(mk.build("S0", out, lock_path, 0.99, epochs=150))
    unet = flat(mk.build("S0unet150", out, lock_path, 0.99, epochs=150))
    aniso = flat(mk.build("S0aniso", out, lock_path, 0.99, epochs=150))
    assert unet == s0  # the UNETR++-lite arm is exactly S0 at 150 epochs
    diff = sorted(k for k in set(aniso) | set(s0) if aniso.get(k) != s0.get(k))
    assert diff == ["/cinema_init/hf_revision", "/student/arch"], diff


def test_inter_slice_consistency_is_one_for_a_stack_of_identical_slices_and_lower_when_a_slice_flips():
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from w2s_ssl4mis_eval import inter_slice_consistency

    lab = np.zeros((6, 32, 32), dtype=np.int64)
    lab[:, 10:20, 10:20] = 3
    rec = lambda p: [{"prediction": torch.from_numpy(p), "target": torch.from_numpy(lab)}]
    same = inter_slice_consistency(rec(lab.copy()))
    assert same["prediction"] == pytest.approx(1.0) and same["ratio"] == pytest.approx(1.0)
    bad = lab.copy()
    bad[3] = 0
    assert inter_slice_consistency(rec(bad))["ratio"] < 0.9
