"""TPC-Gate signals on toy volumes with known answers."""
from __future__ import annotations

import numpy as np
import pytest

from self_audit.candidate_pg import tpc_signals as t


def _cylinder(d=7, h=40, w=40, drift=0.0):
    lab = np.zeros((d, h, w), dtype=np.int64)
    for z in range(d):
        cy, cx = 20, int(round(12 + drift * z))
        yy, xx = np.ogrid[:h, :w]
        lab[z][(yy - cy) ** 2 + (xx - cx) ** 2 <= 36] = 3  # LV disc
        lab[z][((yy - cy) ** 2 + (xx - cx) ** 2 <= 100) & ((yy - cy) ** 2 + (xx - cx) ** 2 > 36)] = 2  # MYO ring
    return lab


def test_alignment_removes_a_linear_drift_and_is_identity_without_one():
    lab = _cylinder(drift=2.0)
    shifts = t.slice_shifts(lab, (3,))
    assert np.ptp(shifts[:, 1]) == 12 and np.all(shifts[:, 0] == 0)  # 6 slices x 2 px of drift
    aligned, valid, _ = t.align(lab, cardiac=True)
    centroids = [np.nonzero(aligned[z] == 3)[1].mean() for z in range(7)]
    assert max(centroids) - min(centroids) < 1.0 and valid.any() and not valid.all()
    straight = _cylinder(drift=0.0)
    assert np.all(t.slice_shifts(straight, (3,)) == 0)


def test_s_flip_known_values():
    pred = np.zeros((5, 4, 4), dtype=np.int64)
    pred[2, 1, 1] = 1  # one voxel flips against both neighbours
    valid = np.ones_like(pred, dtype=bool)
    f = t.s_flip(pred, valid)
    assert f[2, 1, 1] == 1.0 and f[1, 1, 1] == 0.5 and f[3, 1, 1] == 0.5 and f[0, 1, 1] == 0.0 and f[2, 0, 0] == 0.0
    valid[1] = False  # an invalid neighbour is not counted
    assert t.s_flip(pred, valid)[2, 1, 1] == 1.0 and t.s_flip(pred, valid)[1, 1, 1] == 0.0


def test_s_area_known_values_and_end_slices():
    pred = np.zeros((3, 10, 10), dtype=np.int64)
    pred[0, :2, :5] = 3  # area 10
    pred[1, :4, :5] = 3  # area 20
    pred[2, :2, :5] = 3  # area 10
    a = t.s_area(pred)
    assert a[1, 0, 0] == pytest.approx(abs(20 - 10) / 10)  # |20 - mean(10, 10)| / 10 = 1
    assert a[0, 0, 0] == pytest.approx(abs(10 - 20) / 20)  # end slice: one neighbour
    assert a[0, 9, 9] == 0.0  # background voxels carry no area signal


def test_s_topo_generic_and_cardiac():
    pred = _cylinder()
    assert t.s_topo(pred, cardiac=False).max() == 0.0 and t.s_topo(pred, cardiac=True).max() == 0.0
    split = pred.copy()
    split[3, 2:5, 2:5] = 3  # second LV component in slice 3
    topo = t.s_topo(split, cardiac=False)
    assert topo[3].min() == 1.0 and topo[2].max() == 0.0
    hole = pred.copy()
    hole[4, 20, 12] = 0  # hole inside the LV
    assert t.s_topo(hole, cardiac=False)[4].min() == 1.0
    opened = pred.copy()
    opened[5][(opened[5] == 2) & (np.arange(40)[None, :] > 12)] = 0  # MYO ring cut open on one side
    assert t.s_topo(opened, cardiac=True)[5].min() == 1.0 and t.s_topo(opened, cardiac=False)[5].max() == 0.0
    touch = pred.copy()
    touch[1][(touch[1] == 2) & (np.arange(40)[None, :] > 12) & (np.arange(40)[:, None] > 20)] = 1  # RV label where MYO should separate
    assert t.s_topo(touch, cardiac=True)[1].min() == 1.0


def test_rank_roi_and_tpc_flags_the_inconsistent_slice():
    pred = _cylinder(d=9)
    bad = pred.copy()
    bad[4] = 0  # slice 4 lost entirely
    aligned, valid, _ = t.align(bad, cardiac=False)
    roi = t.region_of_interest(aligned, valid)
    sig = t.tpc_score(aligned, valid, roi, cardiac=False)
    assert sig["tpc"][3:6][roi[3:6]].mean() > sig["tpc"][0:2][roi[0:2]].mean()
    assert np.all(t.rank_pct(np.zeros(pred.shape), roi) == 0)  # constant signal ranks 0 (ties take the lowest rank)


def test_auroc_and_baselines():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    w = np.array([False, False, True, True])
    assert t.auroc(s, w) == 1.0 and t.auroc(-s, w) == 0.0 and np.isnan(t.auroc(s, np.zeros(4, bool)))
    prob = np.zeros((4, 3, 2, 2), dtype=np.float32)
    prob[0] = 1.0
    b = t.baselines(prob)
    assert np.allclose(b["one_minus_conf"], 0) and np.allclose(b["entropy"], 0, atol=1e-6) and b["slice_position"][1].max() == 0.0 and b["slice_position"][0].max() == 1.0
