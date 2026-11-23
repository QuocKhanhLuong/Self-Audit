"""SliceSSM S/M/L width ablation: only the widths (and the parameter cap) change."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_w2s_ssl4mis as ssl  # noqa: E402
import w2s_make_configs_ssl4mis as mk  # noqa: E402

from self_audit.candidate_pg.slicessm_student import SliceSSMStudent, build_slicessm_student, export_slicessm_student, load_slicessm_export  # noqa: E402


def _flat(x, p=""):
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            r.update(_flat(v, f"{p}/{k}"))
        return r
    return {p: json.dumps(x)}


def test_s_and_l_configs_differ_from_m_only_in_widths_and_cap(tmp_path):
    out, lock = ssl._synthetic_prepared(tmp_path)
    base = _flat(mk.build("S0slicessm", out, lock, 0.99, epochs=150))
    for arm, widths in mk.SLICESSM_WIDTHS.items():
        cfg = _flat(mk.build(arm, out, lock, 0.99, epochs=150))
        diff = sorted(k for k in set(cfg) | set(base) if cfg.get(k) != base.get(k))
        assert diff == ["/student/slicessm/channels", "/student/slicessm/parameter_cap"], (arm, diff)
        assert json.loads(cfg["/student/slicessm/channels"]) == list(widths) and cfg["/student/slicessm/parameter_cap"] == "null"


def test_widths_scale_parameters_and_the_default_keeps_the_cap(tmp_path):
    out, lock = ssl._synthetic_prepared(tmp_path)
    counts = {}
    for arm in ("S0slicessmS", "S0slicessm", "S0slicessmL"):
        model = build_slicessm_student(mk.build(arm, out, lock, 0.99, epochs=150))
        counts[arm] = sum(p.numel() for p in model.parameters())
    assert counts["S0slicessm"] == 2_553_001
    assert counts["S0slicessmS"] < counts["S0slicessm"] < counts["S0slicessmL"] and counts["S0slicessmL"] > 3_000_000
    cfg = mk.build("S0slicessmL", out, lock, 0.99, epochs=150)
    cfg["student"]["slicessm"]["parameter_cap"] = 3_000_000
    with pytest.raises(ValueError, match="cap"):
        build_slicessm_student(cfg)
    cfg["student"]["slicessm"]["channels"] = [10, 20, 40, 80, 100]
    with pytest.raises(ValueError, match="multiples of 8"):
        build_slicessm_student(cfg)


@pytest.mark.parametrize("arm", ["S0slicessmS", "S0slicessmL"])
def test_larger_widths_forward_export_and_cold_load(arm, tmp_path):
    out, lock = ssl._synthetic_prepared(tmp_path)
    cfg = mk.build(arm, out, lock, 0.99, epochs=150)
    model = build_slicessm_student(cfg).eval()
    x = torch.rand(1, 1, 5, 32, 32)
    geo = {"shape": (5, 32, 32), "spacing_zyx": [10.0, 1.4, 1.4]}
    with torch.no_grad():
        y = model(x, geo)["logits"]
    assert y.shape == (1, 4, 5, 32, 32) and torch.isfinite(y).all()
    export_slicessm_student(model, tmp_path / "m.pt", epoch=1)
    dep = load_slicessm_export(tmp_path / "m.pt")
    assert list(dep.channels) == list(model.channels)
    with torch.no_grad():
        assert torch.allclose(dep(x, geo)["logits"], y, atol=1e-4)
