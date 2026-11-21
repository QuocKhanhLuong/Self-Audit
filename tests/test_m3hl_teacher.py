"""M3HL teacher wrapper and SSL4MIS h5 builder (CPU, no real data)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from self_audit.candidate_pg.teacher_m3hl import M3HLTeacher, minmax_per_volume

ROOT = Path(__file__).resolve().parents[1]


class ToyNet(nn.Module):
    """Returns (logits, high, low) like UNet_HL; logits depend on the input so normalisation is observable."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return self.conv(x), x, x


def test_minmax_matches_hand_value():
    x = torch.tensor([[[[[2.0, 4.0], [6.0, 10.0]]]]])
    out = minmax_per_volume(x)
    assert torch.allclose(out, torch.tensor([[[[[0.0, 0.25], [0.5, 1.0]]]]]))


def test_wrapper_shapes_probabilities_no_grad_and_resize_roundtrip():
    net = ToyNet()
    teacher = M3HLTeacher(net, slice_batch=3)
    image = torch.rand(2, 1, 5, 24, 40, requires_grad=True)
    out = teacher(image, {"shape": (5, 24, 40)}, return_features=True)
    p = out["probabilities"]
    assert p.shape == (2, 4, 5, 24, 40)
    assert torch.allclose(p.sum(1), torch.ones(2, 5, 24, 40), atol=1e-5)
    assert torch.allclose(out["region_logits"].softmax(1), p, atol=1e-4)
    assert not p.requires_grad and out["features"] is None
    assert all(not q.requires_grad for q in teacher.parameters())
    assert net.seen[0].shape[-2:] == (256, 256)
    teacher.train()
    assert not teacher.net.training  # always eval


def test_wrapper_applies_per_volume_minmax_before_resize():
    net = ToyNet()
    teacher = M3HLTeacher(net)
    base = torch.rand(1, 1, 3, 16, 16)
    teacher(base * 7.0 + 3.0)
    first = torch.cat(net.seen)
    net.seen.clear()
    teacher(base)
    second = torch.cat(net.seen)
    assert torch.allclose(first, second, atol=1e-5)  # affine intensity changes are removed by min-max
    assert float(first.min()) == pytest.approx(0.0, abs=1e-6) and float(first.max()) == pytest.approx(1.0, abs=1e-6)


def test_wrapper_rejects_bad_rank():
    with pytest.raises(ValueError):
        M3HLTeacher(ToyNet()).probabilities(torch.rand(1, 5, 8, 8))


@pytest.mark.skipif(not (ROOT / "external/M3HL/code/networks/unet.py").exists(), reason="external/M3HL not cloned")
def test_real_unet_hl_loads_and_runs(tmp_path):
    from self_audit.candidate_pg.teacher_m3hl import _import_unet_hl, build_m3hl_teacher

    net = _import_unet_hl(ROOT / "external/M3HL")(in_chns=1, class_num=4)
    ckpt = tmp_path / "unet_best_model.pth"
    torch.save(net.state_dict(), ckpt)
    teacher = build_m3hl_teacher(ckpt, ROOT / "external/M3HL")
    p = teacher.probabilities(torch.rand(1, 1, 2, 32, 48))
    assert p.shape == (1, 4, 2, 32, 48) and torch.isfinite(p).all()


def _lock_and_lists(tmp_path, labeled_first=True):
    pats = [f"patient{i:03d}" for i in range(1, 11)]
    lock = {"labeled": {"patients": pats[:2]}, "unlabeled": {"patients": pats[2:6]}, "val": {"patients": pats[6:8]},
            "test": {"patients": pats[8:]}}
    d = tmp_path / "lists"
    d.mkdir()
    cases = lambda ps: [f"{p}_frame0{k}" for p in ps for k in (1, 2)]
    train = cases(pats[:6])
    slices = [f"{c}_slice_{i}" for c in train[:4] for i in range(1, 69)] if False else None
    return pats, lock, d, train, cases


def test_h5_builder_reproduces_counts_and_l7_head(tmp_path, monkeypatch):
    import scripts.m3hl_build_h5 as mod

    monkeypatch.setattr(mod, "LABELED_SLICES_7", 8)  # toy: 2 labeled patients x 2 frames x 2 slices
    pats, lock, d, train, cases = _lock_and_lists(tmp_path)
    depth = {c: 2 for c in train}
    slices = [f"{c}_slice_{i}" for c in train[:4] for i in (1, 2)] + [f"{c}_slice_{i}" for c in train[4:] for i in (1, 2)]
    (d / "train.list").write_text("\n".join(train) + "\n")
    (d / "train_slices.list").write_text("\n".join(slices) + "\n")
    (d / "val.list").write_text("\n".join(cases(lock["val"]["patients"])) + "\n")
    (d / "test.list").write_text("\n".join(cases(lock["test"]["patients"])) + "\n")

    def loader(raw_root, case, want_label):
        rng = np.random.default_rng(abs(hash(case)) % 1000)
        image = rng.random((2, 6, 6)).astype(np.float32) * 100 + 5
        label = rng.integers(0, 4, (2, 6, 6)).astype(np.uint8) if want_label else None
        f = tmp_path / "src.bin"
        f.write_bytes(b"x")
        return image, label, 1, [f]

    receipt = mod.build(d, lock, tmp_path, tmp_path / "out", loader=loader)
    assert receipt["status"] == "READY" and receipt["slices_written"] == len(slices)
    import h5py
    with h5py.File(tmp_path / "out/data/patient003_frame01.h5") as f:  # unlabeled TRAIN patient: zero labels
        assert f["label"][:].max() == 0 and 0.0 <= f["image"][:].min() and f["image"][:].max() <= 1.0
    with h5py.File(tmp_path / "out/data/patient001_frame01.h5") as f:
        assert f["label"][:].max() > 0 and f["image"][:].min() == 0.0 and f["image"][:].max() == 1.0

    # a list whose head crosses into an unlabeled patient must be rejected
    bad = slices[2:] + slices[:2]
    (d / "train_slices.list").write_text("\n".join(bad) + "\n")
    with pytest.raises(SystemExit):
        mod.build(d, lock, tmp_path, tmp_path / "out2", loader=loader)


def test_real_lists_pass_lock_check():
    import scripts.m3hl_build_h5 as mod

    lock = json.loads((ROOT / "configs/candidate_pg/ssl4mis_split_lock.json").read_text())
    info = mod.check_lists(ROOT / "configs/candidate_pg/ssl4mis_lists", lock)
    assert info["problems"] == [] and info["labeled_slice_count"] == 136 and len(info["slices"]) == 1312
