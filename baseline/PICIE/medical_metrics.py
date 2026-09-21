import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def dice_score(pred, gt, num_classes):
    scores = np.zeros(num_classes)
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        intersection = np.sum(p & g)
        union = np.sum(p) + np.sum(g)
        if union == 0:
            scores[c] = 1.0 if np.sum(g) == 0 else 0.0
        else:
            scores[c] = 2.0 * intersection / union
    return scores


def _extract_boundary(mask, connectivity=1):
    struct = ndimage.generate_binary_structure(mask.ndim, connectivity)
    eroded = ndimage.binary_erosion(mask, structure=struct)
    return mask & ~eroded


def _surface_distances(pred_boundary, gt_boundary, spacing):
    pred_pts = np.argwhere(pred_boundary) * np.array(spacing)
    gt_pts = np.argwhere(gt_boundary) * np.array(spacing)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return np.array([0.0]), np.array([0.0])
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.array([np.inf]), np.array([np.inf])

    gt_tree = cKDTree(gt_pts)
    pred_tree = cKDTree(pred_pts)

    d_pred_to_gt, _ = gt_tree.query(pred_pts)
    d_gt_to_pred, _ = pred_tree.query(gt_pts)

    return d_pred_to_gt, d_gt_to_pred


def hausdorff_95(pred, gt, spacing, num_classes):
    scores = np.full(num_classes, np.nan)
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        if not np.any(g):
            continue
        pb = _extract_boundary(p)
        gb = _extract_boundary(g)
        d_p2g, d_g2p = _surface_distances(pb, gb, spacing)
        all_d = np.concatenate([d_p2g, d_g2p])
        if np.any(np.isinf(all_d)):
            scores[c] = np.inf
        else:
            scores[c] = np.percentile(all_d, 95)
    return scores


def assd(pred, gt, spacing, num_classes):
    scores = np.full(num_classes, np.nan)
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        if not np.any(g):
            continue
        pb = _extract_boundary(p)
        gb = _extract_boundary(g)
        d_p2g, d_g2p = _surface_distances(pb, gb, spacing)
        if np.any(np.isinf(d_p2g)) or np.any(np.isinf(d_g2p)):
            scores[c] = np.inf
        else:
            scores[c] = (d_p2g.mean() + d_g2p.mean()) / 2.0
    return scores


def compute_medical_metrics(pred_volume, gt_volume, spacing, num_classes=4):
    dice = dice_score(pred_volume, gt_volume, num_classes)
    hd95 = hausdorff_95(pred_volume, gt_volume, spacing, num_classes)
    assd_val = assd(pred_volume, gt_volume, spacing, num_classes)

    return {
        'dice': dice,
        'hd95': hd95,
        'assd': assd_val,
    }
