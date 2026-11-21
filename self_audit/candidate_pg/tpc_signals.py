"""TPC-Gate signals: through-plane coherence of a slice-wise segmentation (offline analysis, numpy only).

Everything is computed on label volumes ``[D,H,W]`` (classes 0 bg, 1 RV, 2 MYO, 3 LV) after a per-slice translation that puts the foreground
(generic) or LV (cardiac) centroid path on a straight line (quadratic fit in z).  Alignment shifts are integer pixels; voxels shifted in from
outside are marked invalid and never counted.  Signals: ``s_flip`` (adjacent-slice label disagreement), ``s_area`` (area jump), ``s_topo``
(per-slice topology flag), their within-volume rank combination (``tpc_score``), and baselines.  Pre-registered choices (not tuned): shift by
rounded polynomial of degree <= 2; rank method "min" over the region of interest; cardiac MYO-ring rule: the 1-px dilation shell of the LV is
"not closed" when less than 90 % of it is MYO; RV touches LV when the 1-px dilation of the LV overlaps RV.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata

RV, MYO, LV = 1, 2, 3


def slice_shifts(label: np.ndarray, classes: tuple[int, ...] | None = None, degree: int = 2) -> np.ndarray:
    """Integer ``(dy, dx)`` per slice that moves the centroid path onto a constant (its mean); zeros if < 2 non-empty slices."""
    d = label.shape[0]
    mask = label > 0 if classes is None else np.isin(label, classes)
    zs, cy, cx = [], [], []
    for z in range(d):
        ys, xs = np.nonzero(mask[z])
        if ys.size:
            zs.append(z), cy.append(ys.mean()), cx.append(xs.mean())
    if len(zs) < 2:
        return np.zeros((d, 2), dtype=int)
    deg = min(degree, len(zs) - 1)
    z_all = np.arange(d)
    path = np.stack([np.polyval(np.polyfit(zs, cy, deg), z_all), np.polyval(np.polyfit(zs, cx, deg), z_all)], axis=1)
    return np.rint(path.mean(axis=0, keepdims=True) - path).astype(int)


def apply_shifts(volume: np.ndarray, shifts: np.ndarray, fill: float = 0) -> np.ndarray:
    """Translate each slice by ``shifts[z]`` (rows, cols), filling uncovered pixels with ``fill``."""
    out = np.full_like(volume, fill)
    h, w = volume.shape[1:]
    for z, (dy, dx) in enumerate(shifts):
        ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
        xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        if abs(dy) < h and abs(dx) < w:
            out[z, yd, xd] = volume[z, ys, xs]
    return out


def align(pred: np.ndarray, *, cardiac: bool, extra: tuple[np.ndarray, ...] = ()) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Aligned prediction, valid mask, and the same translation applied to ``extra`` arrays (e.g. GT, probabilities per slice)."""
    shifts = slice_shifts(pred, (LV,) if cardiac else None)
    valid = apply_shifts(np.ones(pred.shape, dtype=bool), shifts, fill=False)
    return apply_shifts(pred, shifts, fill=0), valid, tuple(apply_shifts(a, shifts, fill=0) for a in extra)


def s_flip(pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fraction of the valid adjacent slices (z-1, z+1) whose label at the same position differs; 0 where no neighbour is valid."""
    d = pred.shape[0]
    diff, count = np.zeros(pred.shape, dtype=np.float32), np.zeros(pred.shape, dtype=np.float32)
    for off in (-1, 1):
        lo, hi = max(0, -off), d - max(0, off)
        nb = slice(lo + off, hi + off)
        ok = valid[lo:hi] & valid[nb]
        diff[lo:hi] += ((pred[lo:hi] != pred[nb]) & ok)
        count[lo:hi] += ok
    return np.where(count > 0, diff / np.maximum(count, 1), 0.0).astype(np.float32)


def s_area(pred: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Per voxel of class c in slice z: |A_c(z) - mean(A_c(z-1), A_c(z+1))| / max(that mean, eps); one neighbour at the end slices; bg -> 0."""
    d = pred.shape[0]
    out = np.zeros(pred.shape, dtype=np.float32)
    for c in (RV, MYO, LV):
        area = (pred == c).sum(axis=(1, 2)).astype(np.float64)
        for z in range(d):
            nbs = [area[k] for k in (z - 1, z + 1) if 0 <= k < d]
            mean = float(np.mean(nbs)) if nbs else 0.0
            out[z][pred[z] == c] = abs(area[z] - mean) / max(mean, 1e-6) if mean > 0 else (0.0 if area[z] == 0 else 1.0)
    return out


def _has_hole_or_split(mask: np.ndarray, *, holes: bool = True) -> bool:
    if not mask.any():
        return False
    _, n = ndi.label(mask)
    return n > 1 or (holes and int(ndi.binary_fill_holes(mask).sum()) != int(mask.sum()))


def s_topo(pred: np.ndarray, *, cardiac: bool, myo_closed: float = 0.9) -> np.ndarray:
    """Per-slice flag (0/1) broadcast over the slice; generic: > 1 component or a hole in any class; cardiac adds an open MYO ring or RV touching LV."""
    flags = np.zeros(pred.shape[0], dtype=np.float32)
    for z in range(pred.shape[0]):
        sl = pred[z]
        bad = any(_has_hole_or_split(sl == c, holes=c != MYO) for c in (RV, MYO, LV))  # the MYO ring has an anatomical hole (the LV)
        if cardiac and not bad:
            lv = sl == LV
            if lv.any():
                shell = ndi.binary_dilation(lv) & ~lv
                if (sl == MYO).any() and (shell & (sl == MYO)).sum() < myo_closed * shell.sum():
                    bad = True
                if (shell & (sl == RV)).any():
                    bad = True
        flags[z] = float(bad)
    return np.broadcast_to(flags[:, None, None], pred.shape).astype(np.float32)


def region_of_interest(pred: np.ndarray, valid: np.ndarray, radius: int = 5) -> np.ndarray:
    """Voxels within ``radius`` px (in plane) of the predicted foreground; independent of the ground truth."""
    fg = pred > 0
    roi = np.stack([ndi.binary_dilation(s, iterations=radius) if s.any() else s for s in fg])
    return roi & valid


def rank_pct(x: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Within-volume rank in [0,1] over the ROI (ties take the lowest rank, so a constant signal ranks 0); 0 outside the ROI."""
    out = np.zeros(x.shape, dtype=np.float32)
    v = x[roi]
    if v.size > 1:
        out[roi] = ((rankdata(v, method="min") - 1) / (v.size - 1)).astype(np.float32)
    return out


def tpc_score(pred: np.ndarray, valid: np.ndarray, roi: np.ndarray, *, cardiac: bool) -> dict[str, np.ndarray]:
    flip, area, topo = s_flip(pred, valid), s_area(pred, valid), s_topo(pred, cardiac=cardiac)
    score = np.maximum(np.maximum(rank_pct(flip, roi), rank_pct(area, roi)), topo * roi)
    return {"s_flip": flip, "s_area": area, "s_topo": topo, "tpc": score.astype(np.float32)}


def baselines(prob: np.ndarray) -> dict[str, np.ndarray]:
    """``prob`` ``[C,D,H,W]`` softmax: 1 - c_T, entropy, 1 - margin, normalised slice distance from the middle."""
    srt = np.sort(prob, axis=0)
    ent = -(prob * np.log(np.clip(prob, 1e-8, 1.0))).sum(0)
    d = prob.shape[1]
    zpos = np.abs(np.arange(d) - (d - 1) / 2) / max((d - 1) / 2, 1e-6)
    return {"one_minus_conf": 1 - srt[-1], "entropy": ent, "one_minus_margin": 1 - (srt[-1] - srt[-2]),
            "slice_position": np.broadcast_to(zpos[:, None, None], prob.shape[1:]).astype(np.float32)}


def auroc(score: np.ndarray, wrong: np.ndarray) -> float:
    """AUROC for ``wrong`` (bool) from ``score`` via ranks; nan if one class is absent."""
    pos = int(wrong.sum())
    neg = wrong.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[wrong].sum() - pos * (pos + 1) / 2) / (pos * neg))


__all__ = ["align", "apply_shifts", "auroc", "baselines", "rank_pct", "region_of_interest", "s_area", "s_flip", "s_topo", "slice_shifts", "tpc_score"]
