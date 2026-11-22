"""M3HL teacher arms: config diff, cbkd_sa_out encoder / step / artifact round trip, teacher construction (CPU, toy tensors)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_candidate_pg_w2s as base  # noqa: E402
import test_w2s_ssl4mis as ssl  # noqa: E402
import w2s_make_configs_ssl4mis as mk  # noqa: E402

from self_audit.candidate_pg import shape_atoms as sa  # noqa: E402
from self_audit.candidate_pg.config import validate_config  # noqa: E402


class _OutTeacher(base._Teacher):
    output_space = True


def _out_codebook(seed=0):
    torch.manual_seed(seed)
    cb = sa.ShapeAtomCodebook(teacher_channels=sa.N_OUT, encoder="out")
    cb.tau_star.fill_(0.5)
    return cb.freeze()


def test_output_cell_encoder_shape_unit_norm_and_padding():
    cb = _out_codebook()
    x = torch.rand(1, 7, 3, 20, 30)  # not a multiple of the (patch) cell: high side padded
    z = cb.embed(x)
    assert z.shape == (1, 64, 3, 3, 4)  # ceil(20/8), ceil(30/8) with patch (1,8,8)
    assert torch.allclose(z.norm(dim=1), torch.ones(1, 3, 3, 4), atol=1e-5)
    out = cb(x)
    assert out["class_logits"].shape[1] == 4 and torch.isfinite(out["margin"]).all()
    idx = sa.StudentAtomBranch(cb)(torch.randn(2, 128, 2, 8, 8))["idx"]
    assert sa.code_usage(idx, None, 128)["perplexity"] > 1.0


def test_cbkd_sa_out_step_gradients_reach_only_the_student():
    cb = _out_codebook()
    teacher = _OutTeacher(cls=1)
    student = base._Student(bias=(0, 4.0, 0, 0), branch=sa.StudentAtomBranch(cb))
    cfg = base._config(cb=True)
    cfg["cbkd"]["variant"] = "cbkd_sa_out"
    out = base._step(cfg, student, teacher, cb, update=1400)
    d = out["diagnostics"]
    assert d["lambda_cb"] > 0 and torch.isfinite(torch.as_tensor(float(d["loss_pm"]))) and d["code_perplexity"] > 1
    out["loss"].backward()
    assert student.code_branch.P_S.weight.grad is not None and student.conv.bias.grad is not None
    assert all(p.grad is None for p in cb.parameters()) and teacher.p.grad is None


def test_variant_and_teacher_kind_must_match():
    cfg = base._config(cb=True)
    cfg["cbkd"]["variant"] = "cbkd_sa_out"
    cb = _out_codebook()
    student = base._Student(branch=sa.StudentAtomBranch(cb))
    with pytest.raises(ValueError, match="cbkd_sa_out requires"):
        base._step(cfg, student, base._Teacher(cls=1), cb, update=1400)  # MedNeXt-like teacher with the out variant


def test_out_codebook_artifact_round_trip(tmp_path):
    cb = _out_codebook(3)
    path = tmp_path / "codebook.pt"
    torch.save({"codebook": cb.state_dict(), "tau_star": 0.5, "patch": [1, 8, 8], "size": 128, "dim": 64, "encoder": "out", "teacher_channels": 7}, path)
    loaded = sa.load_shape_atoms(path)
    x = torch.rand(1, 7, 2, 16, 16)
    assert loaded.encoder == "out" and torch.allclose(loaded.embed(x), cb.embed(x))
    old = tmp_path / "old.pt"  # an artifact written before this change still loads as the feature encoder
    fb = sa.ShapeAtomCodebook()
    torch.save({"codebook": fb.state_dict(), "tau_star": 0.5, "patch": [1, 8, 8], "size": 128, "dim": 64}, old)
    assert sa.load_shape_atoms(old).encoder == "feature"


def test_m3hl_arm_configs_differ_from_mednext_twins_only_in_teacher_and_variant(tmp_path):
    out, lock_path = ssl._synthetic_prepared(tmp_path)
    ckpt = tmp_path / "unet_best_model.pth"
    ckpt.write_bytes(b"toy")

    def flat(x, p=""):
        if isinstance(x, dict):
            r = {}
            for k, v in x.items():
                r.update(flat(v, f"{p}/{k}"))
            return r
        return {p: json.dumps(x)}

    for twin, arm in (("S1kd", "S1kdM3"), ("S2arb", "S2arbM3"), ("S2main", "S2M3")):
        a, b = flat(mk.build(twin, out, lock_path, 0.99)), flat(mk.build(arm, out, lock_path, 0.99, ckpt))
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        assert all(k.startswith(("/pg/teacher_source", "/pg/m3hl")) or k == "/cbkd/variant" for k in diff), (arm, diff)
        assert "/pg/teacher_source" in diff
    assert json.loads(flat(mk.build("S2M3", out, lock_path, 0.99, ckpt))["/cbkd/variant"]) == "cbkd_sa_out"
    with pytest.raises(SystemExit):
        mk.build("S2M3", out, lock_path, 0.99)  # checkpoint required
    bad = mk.build("S2M3", out, lock_path, 0.99, ckpt)
    bad["pg"]["teacher_source"] = "mednext"
    with pytest.raises(ValueError):
        validate_config(bad)  # cbkd_sa_out needs the M3HL teacher


@pytest.mark.skipif(not (ROOT / "external/M3HL/code/networks/unet.py").exists(), reason="external/M3HL not cloned")
def test_engine_builds_the_pinned_m3hl_teacher_and_rejects_a_changed_checkpoint(tmp_path):
    from self_audit.candidate_pg import engine
    from self_audit.candidate_pg.teacher_m3hl import M3HLTeacher, _import_unet_hl

    out, lock_path = ssl._synthetic_prepared(tmp_path)
    ckpt = tmp_path / "unet_best_model.pth"
    torch.save(_import_unet_hl(ROOT / "external/M3HL")(in_chns=1, class_num=4).state_dict(), ckpt)
    cfg = mk.build("S2arbM3", out, lock_path, 0.99, ckpt)
    teacher, meta = engine._build_teacher(cfg, synthetic=False)
    assert isinstance(teacher, M3HLTeacher) and meta["kind"] == "m3hl"
    ckpt.write_bytes(ckpt.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        engine._build_teacher(cfg, synthetic=False)


def test_pipeline_refuses_to_start_before_t_m3hl_finishes_and_rejects_unknown_steps(tmp_path, monkeypatch):
    import m3hl_pipeline

    monkeypatch.setenv("SELF_AUDIT_ROOT", str(tmp_path))
    with pytest.raises(SystemExit, match="has not finished"):
        m3hl_pipeline.main(["--steps", "M3WAIT"])
    with pytest.raises(SystemExit, match="unknown steps"):
        m3hl_pipeline.main(["--steps", "NOPE"])
    assert m3hl_pipeline.ALL.index("S2ARBM3") < m3hl_pipeline.ALL.index("S1KDM3") < m3hl_pipeline.ALL.index("M3CB") < m3hl_pipeline.ALL.index("S2M3")


def test_official_eval_metrics_zero_for_empty_prediction_and_match_medpy():
    import numpy as np

    import m3hl_official_eval as ev

    gt = np.zeros((2, 8, 8), dtype=np.uint8)
    gt[:, 2:6, 2:6] = 1
    assert ev.official_case_metrics(np.zeros_like(gt), gt, 1) == (0.0, 0.0, 0.0, 0.0)
    dice, jc, hd, asd = ev.official_case_metrics(gt.copy(), gt, 1)
    assert dice == 1.0 and jc == 1.0 and hd == 0.0 and asd == 0.0


def test_s2arb_150_arm_is_a_separate_config_that_differs_only_in_epochs(tmp_path):
    out, lock_path = ssl._synthetic_prepared(tmp_path)
    ckpt = tmp_path / "unet_best_model.pth"
    ckpt.write_bytes(b"toy")

    def flat(x, p=""):
        if isinstance(x, dict):
            r = {}
            for k, v in x.items():
                r.update(flat(v, f"{p}/{k}"))
            return r
        return {p: json.dumps(x)}

    a = flat(mk.build("S2arbM3", out, lock_path, 0.99, ckpt))
    b = flat(mk.build("S2arbM3E150", out, lock_path, 0.99, ckpt, epochs=150))
    assert sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)) == ["/training/epochs", "/training/updates"]
    assert mk.ENGINE_ARM["S2arbM3E150"] == "A2"
