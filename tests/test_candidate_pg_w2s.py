"""W2S-Match: arbiter, schedule, stability, gated codebook KD, configs, export.

Software evidence only (synthetic CPU toys); no training, no patient data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from self_audit.candidate_pg import shape_atoms as cbkd_sa, w2s_arbiter as arb, w2s_config
from self_audit.candidate_pg.config import default_config
from self_audit.candidate_pg.export import export_student, load_student_export, verify_cold_load
from self_audit.candidate_pg.reliability import ClassAdaptiveThreshold
from self_audit.candidate_pg.student_pg import PGStudent
from self_audit.candidate_pg.training_step import student_training_step

torch.set_num_threads(1)

_PHASES = (
    ((1, 1, 1), (0.0, 0.0, 0.0)), ((1, 2, 2), (0.0, 0.0, 0.0)), ((1, 4, 4), (0.0, 0.0, 0.0)),
    ((2, 8, 8), (0.0, 0.0, 0.0)), ((2, 16, 16), (0.0, 0.0, 0.0)),
)
SHAPE = (2, 32, 32)
GEOM = {"shape": SHAPE, "spacing_zyx": [1.0, 1.0, 1.0]}


def _probs(classes_conf: list[tuple[int, float]]) -> torch.Tensor:
    """[1,4,1,1,N] probabilities: each voxel puts ``conf`` on ``cls``, rest spread evenly."""
    cols = []
    for cls, conf in classes_conf:
        p = torch.full((4,), (1.0 - conf) / 3.0)
        p[cls] = conf
        cols.append(p)
    return torch.stack(cols, dim=1).reshape(1, 4, 1, 1, len(classes_conf))


# --------------------------------------------------------------------------- arbiter
def test_each_rule_fires_and_first_match_wins() -> None:
    #            agree     veto      override  fallback  drop      low-agree
    t = _probs([(1, 0.97), (1, 0.995), (1, 0.96), (1, 0.96), (1, 0.90), (2, 0.90)])
    s = _probs([(1, 0.50), (2, 0.60), (2, 0.97), (2, 0.60), (2, 0.60), (2, 0.90)])
    # voxel 0 agree with c_T=.97; voxel 5: same label but both < tau
    s[..., 0] = _probs([(1, 0.5)])[..., 0]
    stable = torch.tensor([[[[True, True, True, True, True, True]]]])
    valid = torch.ones(1, 1, 1, 6, dtype=torch.bool)
    y, w, r = arb.arbitrate(t, s, stable, valid, 0.3)
    assert r.flatten().tolist() == [1, 2, 3, 4, 5, 5]
    assert y.flatten().tolist()[:5] == [1, 1, 2, 1, 1]
    assert w.flatten()[:5].tolist() == pytest.approx([1.0, 1.0, 0.3, 0.7, 0.0])
    assert w.flatten()[5].item() == 0.0  # rule 1 with both confidences < tau


def test_override_requires_stability_and_veto_beats_override() -> None:
    t = _probs([(1, 0.96), (1, 0.995)])
    s = _probs([(2, 0.97), (2, 0.97)])
    valid = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    unstable = torch.zeros(1, 1, 1, 2, dtype=torch.bool)
    _, w, r = arb.arbitrate(t, s, unstable, valid, 0.5)
    assert r.flatten().tolist() == [4, 2]  # fallback (no stability), veto
    _, _, r = arb.arbitrate(t, s, ~unstable, valid, 0.5)
    assert r.flatten().tolist() == [3, 2]  # veto still wins over a stable student


def test_pad_voxels_are_zero_and_outputs_carry_no_grad() -> None:
    t = _probs([(1, 0.99), (1, 0.99)]).requires_grad_(True)
    s = _probs([(1, 0.99), (1, 0.99)]).requires_grad_(True)
    valid = torch.tensor([[[[True, False]]]])
    y, w, r = arb.arbitrate(t, s, torch.ones_like(valid), valid, 0.5)
    assert r.flatten().tolist() == [1, 0] and w.flatten().tolist() == [1.0, 0.0]
    assert not (y.requires_grad or w.requires_grad or r.requires_grad)


def test_alpha_schedule_values() -> None:
    f = lambda u: arb.alpha_schedule(u, 100)
    assert (f(0), f(20), f(100)) == (0.0, 0.0, 0.5)
    assert f(35) == pytest.approx(0.25) and f(50) == 0.5 and f(75) == 0.5


# ------------------------------------------------------------------------- stability
def test_source_aligned_stability_counts_and_consistency() -> None:
    b1, b2 = (0, 16, 0, 16), (8, 24, 8, 24)
    shape = (1, 2, 32, 32)
    ya = torch.zeros(shape, dtype=torch.long)
    yb = torch.ones(shape, dtype=torch.long)

    def views(a_ok=True, b_ok=True):
        mixed = []
        for box in (b1, b2):
            m = ya.clone()
            m[..., box[0]:box[1], box[2]:box[3]] = yb[..., box[0]:box[1], box[2]:box[3]] if b_ok else 3
            mixed.append(m)
        return mixed, ya.clone() if a_ok else torch.full(shape, 3)

    mixed, fp = views()
    st_a, st_b, per_view = arb.source_aligned_stability(ya, yb, mixed, fp, [b1, b2])
    # A: visible = fp + (outside box1) + (outside box2) -> 1, 2 or 3 views
    assert not st_a[0, 0, 12, 12]           # inside both boxes: only fp visible (1 view)
    assert bool(st_a[0, 0, 4, 4])           # inside box1 only: fp + view2 (2 views)
    assert bool(st_a[0, 0, 28, 28])         # outside both: 3 views
    # B: visible = views whose box contains x -> 0, 1 or 2 views; only "inside both" is stable
    assert bool(st_b[0, 0, 12, 12]) and not st_b[0, 0, 4, 4] and not st_b[0, 0, 28, 28]
    assert len(per_view) == 3 and torch.equal(per_view[-1], st_a)
    # mapping into view coordinates: B-source voxels use stable_b inside the box
    assert bool(per_view[0][0, 0, 12, 12]) and bool(per_view[0][0, 0, 4, 4]) is False and bool(per_view[0][0, 0, 28, 28])
    # one disagreeing visible view makes the voxel unstable
    mixed, fp = views(a_ok=False)
    st_a, _, _ = arb.source_aligned_stability(ya, yb, mixed, fp, [b1, b2])
    assert not st_a.any()


# ----------------------------------------------------------------------------- toys
@dataclass
class _Frame:
    image: torch.Tensor
    target: torch.Tensor | None
    geometry: dict
    patient_id: str = "p"


class _Teacher(nn.Module):
    feature_phases = _PHASES
    teacher_phases = _PHASES

    def __init__(self, cls: int = 1, strength: float = 6.0):
        super().__init__()
        self.cls, self.strength = cls, strength
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, image, geometry, *, return_features=True, **_):
        b, _, d, h, w = image.shape
        logits = image.new_full((b, 4, d, h, w), -self.strength)
        logits[:, self.cls] = self.strength
        g = torch.Generator().manual_seed(0)
        f3 = torch.randn(b, 256, (d + 1) // 2, (h + 7) // 8, (w + 7) // 8, generator=g)
        feats = tuple(image.new_zeros(b, c, 1, 1, 1) for c in (32, 64, 128)) + (f3, image.new_zeros(b, 512, 1, 1, 1))
        return {"region_logits": logits + self.p, "sdf": image.new_zeros(b, 3, d, h, w), "features": feats}


class _Student(nn.Module):
    phases = _PHASES

    def __init__(self, bias=(0.0, 0.0, 0.0, 0.0), branch=None):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, 1)
        nn.init.zeros_(self.conv.weight)
        self.conv.bias.data = torch.tensor(bias, dtype=torch.float32)
        self.s3 = nn.Conv3d(1, 128, 1)
        self.code_branch = branch

    def forward(self, image, geometry, *, teacher_features=None, keep_mask=None, return_stage3=False,
                return_code=False, skip_dropout=0.0, dropout_generator=None):
        out = {"logits": self.conv(image)}
        if return_stage3 or return_code:
            s3 = self.s3(image[..., ::1, ::8, ::8])
            out["stage3"] = s3
            if return_code and self.code_branch is not None:
                valid = torch.ones((s3.shape[0], *s3.shape[2:]), dtype=torch.bool)
                out["code_branch"] = self.code_branch(s3, valid_mask=valid)
        return out


def _config(mode="w2s_arbiter", cb=False, updates=100):
    cfg = w2s_config.make_config("S2" if cb else ("S0" if mode == "self" else "S2arb"))
    cfg["pg"]["target_mode"] = mode
    cfg["pg"]["lambda_u"]["warmup_updates_labeled_only"] = 0
    cfg["training"]["updates"] = updates
    if cb:
        cfg["cbkd"]["warmup_updates_labeled_only"] = 0
    return cfg


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    def img(): return torch.rand(1, 1, *SHAPE, generator=g)
    tgt = torch.randint(0, 4, (1, *SHAPE), generator=g)
    valid = torch.ones(1, *SHAPE, dtype=torch.bool)
    lf = _Frame(img(), tgt, GEOM)
    ua, ub = _Frame(img(), None, GEOM), _Frame(img(), None, GEOM)
    lv = {"weak": lf.image, "strong": lf.image.clone(), "valid": valid, "geometry": GEOM, "target": tgt}
    def uv(f, extra=False):
        d = {"weak": f.image, "strong": (f.image * 0.9).clone(), "valid": valid, "geometry": GEOM, "strong2": (f.image * 0.8).clone()}
        return d
    uvs = {"a": uv(ua), "b": uv(ub), "boxes": [(0, 16, 0, 16), (8, 24, 8, 24)]}
    return lf, ua, ub, lv, uvs


def _step(cfg, student, teacher, codebook=None, update=10, seed=0):
    lf, ua, ub, lv, uvs = _inputs(seed)
    gen = torch.Generator().manual_seed(1)
    return student_training_step(cfg, student, teacher, codebook, lf, ua, ub, update, None,
                                 labeled_views=lv, u_views=uvs, keep_generator=gen, fp_generator=gen)


def _codebook(seed=0) -> cbkd_sa.ShapeAtomCodebook:
    torch.manual_seed(seed)
    cb = cbkd_sa.ShapeAtomCodebook()
    cb.tau_star.fill_(0.5)
    return cb.freeze()


# ------------------------------------------------------------------------- step: arms
def test_arbiter_step_trains_student_only_and_reports_rules() -> None:
    teacher = _Teacher(cls=1)
    student = _Student(bias=(0, 4.0, 0, 0))
    out = _step(_config(), student, teacher, update=60)
    out["loss"].backward()
    assert student.conv.bias.grad is not None and teacher.p.grad is None
    d = out["diagnostics"]
    assert d["alpha"] == pytest.approx(0.5 * (60 / 100 - 0.2) / 0.3 if 60 / 100 < 0.5 else 0.5)
    assert sum(d["rule_fraction"].values()) == pytest.approx(1.0)
    assert d["rule_fraction"]["agree"] > 0.9  # teacher and student agree on class 1


def test_self_mode_uses_students_own_prediction_with_fixed_threshold(monkeypatch) -> None:
    from self_audit.candidate_pg import w2s_step

    captured = []
    real = w2s_step._packet_loss
    monkeypatch.setattr(w2s_step, "_packet_loss", lambda l, y, w, v: (captured.append((y.clone(), w.clone())), real(l, y, w, v))[1])
    # student: class 2 with p ~ .97 (>= .95) -> weight 1; teacher is absent entirely
    bias = (0.0, 0.0, 6.0, 0.0)  # softmax over 4 -> e^6/(e^6+3) = .9926
    out = _step(_config("self"), _Student(bias=bias), None, update=10)
    assert out["diagnostics"]["target_mode"] == "self"
    for y, w in captured:
        assert (y == 2).all() and (w == 1).all()
    captured.clear()
    _step(_config("self"), _Student(bias=(0, 0, 2.0, 0)), None, update=10)  # p = .71 < .95
    assert captured and all((w == 0).all() for _, w in captured)


def test_fixed_threshold_clip_returns_095_for_all_classes() -> None:
    th = ClassAdaptiveThreshold(4, ema=0.9, clip=[0.95, 0.95])
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        p = torch.softmax(torch.randn(1, 4, 50, 1, 1, generator=g), dim=1)
        th.update(p, torch.ones(1, 50, 1, 1, dtype=torch.bool))
    assert torch.allclose(th.thresholds, torch.full((4,), 0.95))


def test_fusion_default_does_not_dispatch_to_w2s(monkeypatch) -> None:
    from self_audit.candidate_pg import w2s_step

    def boom(*a, **k):  # pragma: no cover
        raise AssertionError("W2S path must not run for target_mode=fusion")

    monkeypatch.setattr(w2s_step, "w2s_student_training_step", boom)
    cfg = default_config(synthetic=True)
    cfg["pg"]["lambda_u"]["warmup_updates_labeled_only"] = 10 ** 6
    cfg["cbkd"]["enabled"] = False
    cfg["training"]["updates"] = 100
    lf, ua, ub, lv, uvs = _inputs()
    gen = torch.Generator().manual_seed(1)

    class _Fusion(_Student):
        def forward(self, image, geometry, *, teacher_features=None, keep_mask=None, **kw):
            return super().forward(image, geometry, **kw)

    out = student_training_step(cfg, _Fusion(), _Teacher(), None, lf, ua, ub, 0, ClassAdaptiveThreshold(4),
                                labeled_views=lv, u_views=uvs, keep_generator=gen, fp_generator=gen)
    assert torch.isfinite(out["loss"])


# ----------------------------------------------------------------- gated codebook KD
def test_pm_loss_zero_when_gate_closed_and_gradient_only_through_student() -> None:
    p = torch.softmax(torch.randn(1, 4, 2, 8, 8), dim=1).requires_grad_(True)
    q = torch.softmax(torch.randn(1, 4, 2, 8, 8), dim=1).requires_grad_(True)
    m = torch.rand(1, 2, 8, 8).requires_grad_(True)
    valid = torch.ones(1, 2, 8, 8, dtype=torch.bool)
    closed = cbkd_sa.pm_loss(p, q, m, torch.zeros(1, 2, 8, 8), valid)
    assert closed.item() == 0.0
    closed.backward()
    opened = cbkd_sa.pm_loss(p, q, m, torch.ones(1, 2, 8, 8), valid)
    opened.backward()
    assert opened.item() > 0 and p.grad is not None and p.grad.abs().sum() > 0
    assert q.grad is None and m.grad is None


def test_gate_is_zero_for_rules_3_and_5_in_step() -> None:
    # teacher low confidence + student low confidence -> every voxel is rule 5 -> L_pm exactly 0
    cb = _codebook()
    teacher = _Teacher(cls=1, strength=0.1)
    student = _Student(bias=(0, 0.5, 0, 0))
    branch = cbkd_sa.StudentAtomBranch(cb)
    student.code_branch = branch
    out = _step(_config(cb=True), student, teacher, cb, update=1400)
    d = out["diagnostics"]
    assert d["rule_fraction"]["drop"] == pytest.approx(1.0)
    assert d["loss_pm"].item() == 0.0 and d["gate_fraction"] == 0.0


def test_cbkd_step_gradients_reach_student_and_never_teacher_or_codebook() -> None:
    cb = _codebook()
    teacher = _Teacher(cls=1)
    student = _Student(bias=(0, 4.0, 0, 0), branch=cbkd_sa.StudentAtomBranch(cb))
    out = _step(_config(cb=True), student, teacher, cb, update=1400)
    d = out["diagnostics"]
    assert d["lambda_cb"] > 0 and d["gate_fraction"] > 0.9 and d["loss_pm"].item() >= 0
    assert d["code_perplexity"] is not None and d["code_perplexity"] > 1
    out["loss"].backward()
    assert student.code_branch.P_S.weight.grad is not None
    assert student.conv.bias.grad is not None
    assert all(p.grad is None for p in cb.parameters()) and teacher.p.grad is None
    assert student.code_branch.C.requires_grad is False
    assert all(p.grad is None for p in student.code_branch.D_A.parameters())


def test_lambda_cb_staggered_onset() -> None:
    assert cbkd_sa.lambda_cb(1199) == 0.0
    assert cbkd_sa.lambda_cb(1200) == pytest.approx(0.5 / 400)
    assert cbkd_sa.lambda_cb(1599) == pytest.approx(0.5) and cbkd_sa.lambda_cb(5000) == 0.5


def test_cbkd_sa_scale_and_code_usage_at_init() -> None:
    cb = _codebook()
    feat = torch.randn(1, 256, 2, 4, 4)
    out = cb(feat)
    assert torch.allclose(out["z"].norm(dim=1), torch.ones(1, 2, 4, 4), atol=1e-5)
    assert torch.allclose(cb.codes().norm(dim=1), torch.ones(128), atol=1e-5)
    assert (out["d1"] >= 0).all() and (out["d2"] >= out["d1"]).all() and (out["d2"] <= 4 + 1e-5).all()
    assert (out["margin"] >= 0).all() and (out["margin"] < 1).all()
    branch = cbkd_sa.StudentAtomBranch(cb)
    s3 = torch.randn(2, 128, 2, 8, 8)
    idx = branch(s3)["idx"]
    assert cbkd_sa.code_usage(idx, None, 128)["perplexity"] > 1.0


def test_atom_assembly_places_cell_patches_on_centered_grid() -> None:
    cell = torch.arange(4.0).reshape(1, 1, 1, 2, 2).expand(1, 1, 1, 2, 2)
    up = cbkd_sa.upsample_cells(cell, (1, 12, 12), (1, 8, 8))
    # cell i is centred on voxel 8*i: voxels [-4,4) -> cell 0, [4,12) -> cell 1
    assert up.shape == (1, 1, 1, 12, 12)
    assert up[0, 0, 0, 0, 0] == 0 and up[0, 0, 0, 3, 3] == 0
    assert up[0, 0, 0, 4, 4] == 3 and up[0, 0, 0, 11, 11] == 3
    assert up[0, 0, 0, 0, 11] == 1 and up[0, 0, 0, 11, 0] == 2


def test_fit_loss_updates_codebook_and_tau_star_calibration() -> None:
    torch.manual_seed(0)
    cb = cbkd_sa.ShapeAtomCodebook()
    feat = torch.randn(1, 256, 2, 4, 4)
    cls_t = F.one_hot(torch.randint(0, 4, (1, 2, 32, 32)), 4).permute(0, 4, 1, 2, 3).float()
    ones = torch.ones(1, 2, 32, 32)
    sdf = torch.zeros(1, 3, 2, 32, 32)
    before = cb.C.detach().clone()
    loss, terms = cbkd_sa.fit_loss(cb, feat, cls_t, ones, sdf, ones, (2, 32, 32))
    loss.backward()
    assert torch.isfinite(loss) and cb.C.grad is not None and cb.P_T.weight.grad is not None
    tau = cbkd_sa.calibrate_tau_star(cb, [feat])
    assert tau > 0 and float(cb.tau_star) == pytest.approx(tau) and torch.equal(before, cb.C.detach())


# ------------------------------------------------------------------ configs / export
def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


def test_arm_config_diff_is_exactly_the_intended_fields() -> None:
    s0, s2, s2a = (_flat(w2s_config.make_config(a)) for a in ("S0", "S2", "S2arb"))
    diff = lambda a, b: {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert diff(s0, s2a) == {"pg.target_mode"}
    assert all(k.startswith("cbkd.") for k in diff(s2a, s2)) and diff(s2a, s2)
    assert s2["training.updates"] == 8000 and s2["pg.w2s.tau"] == 0.95
    w = w2s_config.make_config("W")
    assert w["teacher"]["semi"]["threshold"]["clip"] == [0.95, 0.95]
    base = default_config()
    assert {k for k in _flat(w).keys() | _flat(base).keys() if _flat(w).get(k) != _flat(base).get(k)} == {
        "teacher.semi.threshold.clip", "teacher.select", "training.selection.split", "training.eval_splits"}


def test_invalid_w2s_configs_are_rejected() -> None:
    from self_audit.candidate_pg.config import validate_config

    cfg = w2s_config.make_config("S2")
    cfg["pg"]["target_mode"] = "fusion"
    with pytest.raises(ValueError):
        validate_config(cfg)  # cbkd_sa_gated requires the arbiter
    cfg = w2s_config.make_config("S0")
    cfg["pg"]["w2s"]["tau"] = 0.9
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = w2s_config.make_config("W")
    cfg["teacher"]["semi"]["threshold"]["clip"] = [0.5, 0.9]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_student_export_strips_codebook_branch_and_cold_loads(tmp_path) -> None:
    config = default_config(synthetic=True)
    model = PGStudent(config)
    model.attach_code_branch(cbkd_sa.StudentAtomBranch(_codebook()))
    path = tmp_path / "student.pt"
    receipt = export_student(model, config, path, verify=True)
    deployed = load_student_export(path)
    keys = list(deployed.state_dict())
    assert not any(any(t in k for t in ("code_branch", "P_S", "D_A", "teacher", "fusions.")) for k in keys)
    assert receipt["metadata"]["training_parameters_total"] > receipt["metadata"]["parameters_total"]
    volumes = [(torch.rand(1, 1, 2, 17, 19), {"shape": (2, 17, 19), "spacing_zyx": [1.0, 1.0, 1.0]}) for _ in range(2)]
    assert verify_cold_load(model, path, volumes)["status"] == "PASS"


# ---------------------------------------------------------------- saturation + watchdogs
def test_saturation_rule_and_watchdogs_fire_once() -> None:
    from self_audit.candidate_pg import w2s_checks as chk

    acc = chk.SaturationAccumulator()
    p = torch.zeros(1, 4, 1, 1, 10)
    p[:, 0] = 0.0
    p[:, 1, ..., :9] = 0.999; p[:, 0, ..., :9] = 0.001   # 9 confident FG voxels
    p[:, 1, ..., 9] = 0.6; p[:, 0, ..., 9] = 0.4            # 1 unconfident FG voxel
    acc.add(p)
    rep = acc.report()
    assert rep["overall_fraction"] == pytest.approx(0.9) and rep["tau_v"] == 0.99  # not > 0.90
    assert chk.choose_tau_v(0.91) == 0.999 and chk.choose_tau_v(0.5) == 0.99

    messages: list[str] = []
    wd = chk.Watchdogs(fg_prior=0.2, alpha_onset_update=100, emit=messages.append)
    wd.update(10, {"override_share_fg_disagree": 0.4, "fg_coverage": 0.2})
    wd.update(11, {"override_share_fg_disagree": 0.4, "fg_coverage": 0.3})
    assert len(messages) == 2  # override-high once, fg-drift once
    for u in range(100, 700, 50):
        wd.update(u, {"override_share_fg_disagree": 0.0, "alpha": 0.1, "fg_coverage": 0.2})
    assert sum("veto saturation" in m for m in messages) == 1
