"""CBKD-SA (cbkd.variant = cbkd_sa) in the A4 fusion step, grouped clipping, S3-SA stage.

Software evidence only (synthetic CPU toys).
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from self_audit.candidate_pg import codebook as v1, shape_atoms as sa
from self_audit.candidate_pg.config import default_config, validate_config
from self_audit.candidate_pg.engine import _grouped_backward
from self_audit.candidate_pg.reliability import ClassAdaptiveThreshold
from self_audit.candidate_pg.student_pg import PGStudent
from self_audit.candidate_pg.training_step import student_training_step

torch.set_num_threads(1)
SHAPE = (2, 32, 32)
GEOM = {"shape": SHAPE, "spacing_zyx": [5.0, 1.4, 1.4]}
PHASES = ((((1, 1, 1), (0., 0., 0.)), ((1, 2, 2), (0., 0., 0.)), ((1, 4, 4), (0., 0., 0.)), ((2, 8, 8), (0., 0., 0.)), ((2, 16, 16), (0., 0., 0.))))


class _Teacher(nn.Module):
    feature_phases = PHASES
    teacher_phases = PHASES

    def __init__(self, strength=6.0):
        super().__init__()
        self.strength = strength
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, image, geometry, *, return_features=True, **_):
        b, _, d, h, w = image.shape
        logits = image.new_full((b, 4, d, h, w), -self.strength)
        logits[:, 1] = self.strength
        g = torch.Generator().manual_seed(0)
        feats = []
        for ch, (step, _o) in zip((32, 64, 128, 256, 512), PHASES):
            shape = tuple(max(1, (n + s - 1) // s) for n, s in zip((d, h, w), step))
            feats.append(torch.randn(b, ch, *shape, generator=g) * 0.1)
        return {"region_logits": logits + self.p, "sdf": image.new_zeros(b, 3, d, h, w), "features": tuple(feats)}


def _config(updates=100):
    cfg = default_config(synthetic=True)
    cfg["pg"]["enabled"] = True
    cfg["pg"]["lambda_u"]["warmup_updates_labeled_only"] = 0
    cfg["cbkd"].update({"enabled": True, "variant": "cbkd_sa", "warmup_updates_labeled_only": 0, "ramp_updates": 4, "lambda_cb": 0.5})
    cfg["training"]["updates"] = updates
    return cfg


def _codebook():
    torch.manual_seed(0)
    cb = sa.ShapeAtomCodebook(patch=(1, 8, 8))
    cb.tau_star.fill_(0.5)
    return cb.freeze()


def _student(cb):
    torch.manual_seed(0)
    return PGStudent(default_config(synthetic=True)).attach_code_branch(sa.StudentAtomBranch(cb))


def _run(cfg, student, teacher, cb, update=10):
    g = torch.Generator().manual_seed(3)
    def img(): return torch.rand(1, 1, *SHAPE, generator=g)
    valid = torch.ones(1, *SHAPE, dtype=torch.bool)
    tgt = torch.randint(0, 4, (1, *SHAPE), generator=g)
    Frame = type("F", (), {})
    lf = Frame(); lf.image, lf.target, lf.geometry = img(), tgt, GEOM
    ua, ub = Frame(), Frame(); ua.image, ua.target, ua.geometry = img(), None, GEOM; ub.image, ub.target, ub.geometry = img(), None, GEOM
    lv = {"weak": lf.image, "strong": lf.image * 0.95, "valid": valid, "geometry": GEOM, "target": tgt}
    def uv(f): return {"weak": f.image, "strong": (f.image * 0.9).clone(), "strong2": (f.image * 0.8).clone(), "valid": valid, "geometry": GEOM}
    uvs = {"a": uv(ua), "b": uv(ub), "boxes": [(0, 16, 0, 16), (8, 24, 8, 24)]}
    return student_training_step(cfg, student, teacher, cb, lf, ua, ub, update, ClassAdaptiveThreshold(4),
                                 labeled_views=lv, u_views=uvs, keep_generator=g, fp_generator=g)


def test_a4_sa_step_gradients_and_diagnostics() -> None:
    cb, teacher = _codebook(), _Teacher()
    student = _student(cb)
    student.train()
    out = _run(_config(), student, teacher, cb, update=10)
    d = out["diagnostics"]
    assert d["lambda_cb"] == pytest.approx(0.5) and d["lambda_u"] > 0
    for key in ("loss_pm", "loss_atom", "loss_commit", "m_T_mean", "student_code_perplexity", "loss_pm_l", "loss_pm_u"):
        assert key in d and torch.isfinite(torch.as_tensor(d[key]).float())
    assert d["student_code_perplexity"] > 1.0  # test 10: not collapsed at init
    out["loss"].backward()
    assert student.code_branch.P_S.weight.grad is not None and student.code_branch.P_S.weight.grad.abs().sum() > 0
    assert student.logits_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in cb.parameters()) and teacher.p.grad is None
    assert not student.code_branch.C.requires_grad
    assert all(p.grad is None for p in student.code_branch.D_A.parameters())


def test_sa_onset_is_staggered_and_ramped() -> None:
    cfg = _config(); cfg["cbkd"].update({"warmup_updates_labeled_only": 1200, "ramp_updates": 400}); cfg["pg"]["lambda_u"]["warmup_updates_labeled_only"] = 800
    from self_audit.candidate_pg.training_step import _config_lambda_cb, _config_lambda_u

    assert _config_lambda_u(cfg, 800) > 0 and _config_lambda_cb(cfg, 800) == 0.0
    assert _config_lambda_cb(cfg, 1199) == 0.0 and 0 < _config_lambda_cb(cfg, 1200) < 0.01
    assert _config_lambda_cb(cfg, 1600) == pytest.approx(0.5)


def test_grouped_clip_factors_and_per_term_norms() -> None:
    cb, teacher = _codebook(), _Teacher()
    student = _student(cb); student.train()
    out = _run(_config(), student, teacher, cb, update=10)
    info = _grouped_backward(student, out, 1e-3, per_term=True)
    assert info["clip_factor_main"] < 1 and info["clip_factor_cbkd"] < 1  # tiny clip forces both groups
    assert set(info) >= {"norm_main", "norm_cbkd", "norm_sup", "norm_pg"}
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in student.parameters() if p.grad is not None))
    assert float(total) <= 2e-3 + 1e-6  # two groups, each clipped to 1e-3
    out2 = _run(_config(), student, teacher, cb, update=10)
    info2 = _grouped_backward(student, out2, 1e9, per_term=False)
    assert info2["clip_factor_main"] == 1.0 and "norm_sup" not in info2


def test_scale_robustness_and_v1_saturation_regression() -> None:
    # v1 regression: probability below 1e-12 gives exactly the clamp value 27.63 and zero gradient.
    rho_t = torch.zeros(1, 1, 1, 1, 4); rho_t[..., 0] = 1.0
    logits = torch.tensor([-60.0, 0.0, 0.0, 0.0]).reshape(1, 1, 1, 1, 4).requires_grad_(True)
    rho_s = logits.softmax(-1)
    loss = v1.assign_kl(rho_t, rho_s)
    loss.backward()
    assert loss.item() == pytest.approx(27.631, abs=1e-2) and logits.grad.abs().max() == 0
    # CBKD-SA: realistic feature scale (|z| ~ 10-30) still gives finite, nonzero gradients.
    cb = _codebook()
    branch = sa.StudentAtomBranch(cb)
    x = (torch.randn(1, 128, 2, 4, 4) * 20).requires_grad_(True)
    out = branch(x)
    tgt = torch.randint(0, 4, (1, 2, 32, 32))
    loss, _ = sa.atom_loss(out, tgt, None, torch.ones(1, 2, 32, 32, dtype=torch.bool), (2, 32, 32), (1, 8, 8))
    (loss + sa.commitment(out["z"], out["e"], None)).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.allclose(out["z"].norm(dim=1), torch.ones(1, 2, 4, 4), atol=1e-4)


def test_a4_sa_config_differs_from_a4_only_in_cbkd_fields() -> None:
    a4 = default_config(); a4["pg"]["enabled"] = True; a4["cbkd"]["enabled"] = True
    sa_cfg = copy.deepcopy(a4); sa_cfg["cbkd"].update({"variant": "cbkd_sa", "warmup_updates_labeled_only": 1200, "ramp_updates": 400})
    validate_config(sa_cfg)
    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            o.update(flat(v, f"{p}.{k}" if p else k) if isinstance(v, dict) else {(f"{p}.{k}" if p else k): v})
        return o
    fa, fb = flat(a4), flat(sa_cfg)
    assert {k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)} == {"cbkd.variant", "cbkd.warmup_updates_labeled_only", "cbkd.ramp_updates"}


def test_s3_sa_stage_runs_resumes_and_freezes(tmp_path) -> None:
    """Tiny end-to-end S3-SA on a synthetic TRAIN view: runs, checkpoints, resumes, writes a frozen artifact."""

    from types import SimpleNamespace

    class View:
        def __init__(self):
            g = torch.Generator().manual_seed(1)
            self.frames = {}
            for p in range(3):
                for t in range(2):
                    img = torch.rand(1, 1, *SHAPE, generator=g)
                    tgt = torch.randint(0, 4, (1, *SHAPE), generator=g)
                    self.frames[(f"patient{p:03d}", t, True)] = SimpleNamespace(image=img, target=tgt, geometry=GEOM, patient_id=f"patient{p:03d}")
                    self.frames[(f"patient{p:03d}", t, False)] = SimpleNamespace(image=img.clone(), target=None, geometry=GEOM, patient_id=f"patient{p:03d}")
        def labeled_keys(self): return [(f"patient{p:03d}", 0) for p in range(2)]
        def unlabeled_keys(self): return [(f"patient{p:03d}", t) for p in range(3) for t in range(2)]
        def load(self, key, labeled): return self.frames[(key[0], key[1], labeled)]

    class Sampler:
        def __init__(self, *_a, **_k): self.i = 0
        def next_step(self):
            self.i += 1
            return ("patient000", 0), ("patient001", self.i % 2), ("patient002", self.i % 2)
        def state_dict(self): return {"i": self.i}
        def load_state_dict(self, s): self.i = s["i"]

    import self_audit.candidate_pg.data as data_mod
    data_mod_PG = data_mod.PGSampler
    data_mod.PGSampler = Sampler
    try:
        view = View()
        bundle = SimpleNamespace(train=view, val=view)
        cfg = default_config(synthetic=True)
        cfg["codebook"]["updates"] = 4
        cfg["codebook"]["codebook"].update({"size": 16, "dim": 8})
        student = PGStudent(default_config(synthetic=True))
        teacher = _Teacher()
        r1 = sa.run_shape_atoms(cfg, tmp_path, bundle, view, teacher, student, max_updates=2, checkpoint_every=2, tau_frames=2)
        assert (tmp_path / "checkpoint.pt").is_file() and r1["codebook_frozen"] and r1["tau_star"] > 0
        r2 = sa.run_shape_atoms(cfg, tmp_path, bundle, view, teacher, student, resume=True, checkpoint_every=2, tau_frames=2)
        assert r2["updates"] == 4
        model = sa.load_shape_atoms(r2["artifact"])
        assert all(not p.requires_grad for p in model.parameters()) and float(model.tau_star) == pytest.approx(r2["tau_star"])
        assert "mean" in r2["val_reconstruction"] and r2["val_reconstruction"]["frames"] > 0
    finally:
        data_mod.PGSampler = data_mod_PG
