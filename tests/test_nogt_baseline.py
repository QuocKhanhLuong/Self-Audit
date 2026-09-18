"""GT firewalls and geometry tests; synthetic success is not MRI evidence."""
import inspect
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_audit_nogt.core import SmallAnnotator, cache_loss, canonicalize, cluster_image, direct_loss, name_partition
from self_audit_nogt.data import ReadFirewall, native_volume, prepare_images
from self_audit_nogt.runner import source_hashes

torch.set_num_threads(2)


def ring_fixture(gap=False):
    yy, xx = np.mgrid[:96, :96]
    r2 = (yy - 48)**2 + (xx - 48)**2
    p = np.zeros((96, 96), np.uint8)
    p[r2 < 16**2] = 1
    p[(r2 >= 16**2) & (r2 <= 22**2)] = 2
    if gap:
        p[48, 64:72] = 0  # explicit four-neighbour cavity-to-exterior channel
    x = np.take(np.array([.1, .8, .4], np.float32), p)
    return x, p


def test_actual_resolver_ring_gap():
    x, p = ring_fixture()
    closed = name_partition(x, p)
    _, p2 = ring_fixture(True)
    opened = name_partition(x, p2)  # same image, only anonymous topology changes
    assert "R3_enclosure" in closed["trace"]["rules_fired"]
    assert "R3_enclosure" not in opened["trace"]["rules_fired"]
    assert closed["valid"][48, 48]
    assert not opened["valid"][48, 48]
    assert opened["named"][48, 48] == 255
    assert (p == p2).mean() > .99


def test_permutation_anonymous_mapping():
    x, p = ring_fixture()
    perm = np.take(np.array([5, 7, 3]), p)
    a, b = name_partition(x, p), name_partition(x, perm)
    assert np.array_equal(a["named"], b["named"])
    assert np.array_equal(canonicalize(p, x), canonicalize(perm, x))


def test_absent_class_and_all_empty():
    x, p = ring_fixture()
    r = name_partition(x, p)
    assert 1 not in np.unique(r["named"])  # RV need not be invented
    empty = name_partition(x, np.zeros_like(p))
    assert np.all(empty["named"] == 255)
    assert not empty["valid"].any()


@pytest.mark.parametrize("mode", ["cache", "direct"])
def test_shape_grad_finite_and_no_history(mode):
    torch.manual_seed(17)
    model = SmallAnnotator()
    x = torch.rand(2, 1, 48, 48)
    mask = torch.ones(2, 48, 48, dtype=torch.bool)
    logits = model(x)
    loss = cache_loss(logits, (x[:, 0] * 4).long().clamp_max(3), mask) if mode == "cache" else direct_loss(logits, x, mask)
    loss.backward()
    assert logits.shape == (2, 4, 48, 48) and torch.isfinite(loss)
    assert model.enc1[0].weight.grad.norm() > 0
    assert model.head[-1].weight.grad.norm() > 0
    with torch.no_grad():
        assert torch.allclose(model(x)[0], model(x[:1])[0], atol=1e-6)


def test_reference_open_denied(tmp_path):
    p = tmp_path / "patient001_frame01_gt.nii.gz"
    p.write_bytes(b"reference")
    guard = ReadFirewall()
    with guard, pytest.raises(PermissionError):
        p.read_bytes()
    assert len(guard.denied) == 1


def test_missing_and_swapped_references_do_not_change_training(tmp_path):
    """Run actual scratch loss/optimizer with references present, missing, replaced."""
    x = torch.linspace(0, 1, 32 * 32).reshape(1, 1, 32, 32)
    target = torch.from_numpy(cluster_image(x[0, 0].numpy()))[None]
    ref = tmp_path / "reference.npy"
    states, predictions = [], []
    for condition in ("original", "missing", "swapped"):
        if condition == "missing":
            ref.unlink()
        else:
            np.save(ref, np.full((32, 32), int(condition == "swapped")))
        guard = ReadFirewall()
        with guard:
            torch.manual_seed(17)
            m = SmallAnnotator()
            opt = torch.optim.Adam(m.parameters(), lr=.001)
            for _ in range(2):
                opt.zero_grad()
                cache_loss(m(x), target, torch.ones_like(target, dtype=torch.bool)).backward()
                opt.step()
            states.append({k: v.clone() for k, v in m.state_dict().items()})
            predictions.append(m(x).detach())
        assert not guard.denied
    for state in states[1:]:
        assert all(torch.equal(v, state[k]) for k, v in states[0].items())
    assert all(torch.equal(predictions[0], p) for p in predictions[1:])


def test_raw_prepare_no_mask_reads_and_geometry(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    affine = np.diag([1.2, 1.8, 8.0, 1])
    image = np.arange(30 * 40 * 2, dtype=np.float32).reshape(30, 40, 2)
    nib.save(nib.Nifti1Image(image, affine), raw / "patient001_frame01.nii.gz")
    (raw / "patient001_frame01_gt.nii.gz").write_bytes(b"unreadable-mask")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_patients": ["patient001"], "val_patients": []}))
    guard = ReadFirewall()
    with guard:
        m = prepare_images(raw, split, tmp_path / "cache", 48)
    assert not guard.denied
    assert all("_gt" not in p for p in guard.reads)
    assert np.array_equal(m["volumes"][0]["affine"], nib.load(raw / "patient001_frame01.nii.gz").affine)
    n = native_volume(np.full((2, 48, 48), 255, np.uint8), m["volumes"][0])
    assert n.shape == image.shape and (n == 255).all()


def test_patient_overlap_rejected(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_patients": ["patient001"], "val_patients": ["patient001"]}))
    with pytest.raises(ValueError, match="overlap"):
        prepare_images(tmp_path, split, tmp_path / "cache", 48)


def test_method_interfaces_and_frozen_sources():
    for fn in (SmallAnnotator.forward, cluster_image, name_partition, direct_loss, cache_loss):
        assert not set(inspect.signature(fn).parameters) & {"gt", "reference", "label_true", "y"}
    assert "self_audit_maskfree/ontology.py" in source_hashes()


def test_deterministic_no_equal_area_constraint():
    x = np.zeros((48, 48), np.float32)
    x[20:24, 20:24] = 1
    a = cluster_image(x)
    assert np.array_equal(a, cluster_image(x))
    areas = np.bincount(a.ravel(), minlength=4)
    assert areas.max() > 2 * areas.min()


def test_evaluator_rejects_changed_prediction_before_gt_open(tmp_path, monkeypatch):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_nogt_frozen.py"
    spec = importlib.util.spec_from_file_location("independent_evaluator", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    assert evaluator.adjusted_rand(np.array([[2, 0], [0, 2]])) == 1
    assert evaluator.adjusted_rand(np.array([[0, 2], [2, 0]])) == 1
    assert evaluator.adjusted_rand(np.ones((2, 2))) == pytest.approx(-.5)
    run = tmp_path / "run"
    run.mkdir()
    (run / "prediction.npz").write_bytes(b"tampered")
    (run / "FROZEN.json").write_text(json.dumps({"predictions": {"prediction.npz": "wrong_hash"}}))
    calls = []
    monkeypatch.setattr(evaluator.nib, "load", lambda p: calls.append(p))
    monkeypatch.setattr(sys, "argv", [str(path), "--run", str(run), "--references", str(tmp_path),
                                     "--out", str(tmp_path / "result.json")])
    with pytest.raises(RuntimeError, match="changed after freeze"):
        evaluator.main()
    assert not calls
    # Evaluation does not even construct torch graphs; no caller in the method imports it.
    assert "torch" not in evaluator.__dict__
