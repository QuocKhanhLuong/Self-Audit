"""Per-image segment-to-class matching and Dice, evaluator-only.

Independent implementation of the behaviour specified in ``spec.json``
(``per_image_remapped_dice``), which records the evidence from the pinned
DSS-US evaluation path. No upstream source is copied or imported.

Per image:
1. Raw segment IDs and GT labels are each ranked in ascending order (consecutive
   indices); IoU is computed for every (segment, class) pair, IoU threshold 0.
2. Equal counts: one-to-one assignment maximising IoU (Hungarian; cost
   ``pixel_count - IoU``, rows = segments, columns = classes).
   Unequal counts: exclusive majority vote. GT classes are visited in ascending
   order; each takes its highest-IoU segment (lowest index on ties). If that
   segment already serves an earlier class, the later class wins only with a
   strictly greater IoU; the losing class stays unmatched and is never revisited.
3. Matched segments take their class label; every other pixel becomes 0.
4. Dice is computed per foreground label 1..n_classes-1 against the original GT;
   a label absent from both GT and the remapped prediction scores 0. The image
   score is the mean over those labels.
Across images: arithmetic mean and population standard deviation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_matrix(segments: np.ndarray, classes: np.ndarray, partition: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU[segment_index, class_index]."""
    result = np.empty((len(segments), len(classes)))
    for i, segment in enumerate(segments):
        in_segment = partition == segment
        for j, label in enumerate(classes):
            in_class = gt == label
            union = np.count_nonzero(in_segment | in_class)
            result[i, j] = np.count_nonzero(in_segment & in_class) / max(float(union), 1e-8)
    return result


def _exclusive_majority(iou: np.ndarray) -> dict[int, int]:
    """Class index -> segment index (unmatched classes omitted)."""
    owner: dict[int, int] = {}       # segment index -> class index currently holding it
    assigned: dict[int, int] = {}    # class index -> segment index
    for class_index in range(iou.shape[1]):
        best_segment = int(np.argmax(iou[:, class_index]))
        holder = owner.get(best_segment)
        if holder is None:
            owner[best_segment] = class_index
            assigned[class_index] = best_segment
        elif iou[best_segment, class_index] > iou[best_segment, holder]:
            owner[best_segment] = class_index
            assigned[class_index] = best_segment
            del assigned[holder]
    return assigned


def match_segments(partition: np.ndarray, gt: np.ndarray) -> tuple[dict[int, int], str]:
    """Return {raw segment ID: GT label} and the branch used ("hungarian" or "exclusive_majority")."""
    partition, gt = np.asarray(partition), np.asarray(gt)
    if partition.shape != gt.shape:
        raise ValueError("native evaluation grid mismatch")
    segments, classes = np.unique(partition), np.unique(gt)
    iou = _iou_matrix(segments, classes, partition, gt)
    if len(segments) == len(classes):
        rows, columns = linear_sum_assignment(float(partition.size) - iou)
        pairs = {int(r): int(c) for r, c in zip(rows, columns)}
        return {int(segments[r]): int(classes[c]) for r, c in pairs.items()}, "hungarian"
    assigned = _exclusive_majority(iou)
    return {int(segments[s]): int(classes[c]) for c, s in assigned.items()}, "exclusive_majority"


def remap(partition: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    result = np.zeros(np.asarray(partition).shape, dtype=np.int64)
    for segment, label in mapping.items():
        result[np.asarray(partition) == segment] = label
    return result


def image_dice(remapped: np.ndarray, gt: np.ndarray, n_classes: int) -> list[float]:
    scores = []
    for label in range(1, int(n_classes)):
        in_gt, in_prediction = gt == label, remapped == label
        tp = np.count_nonzero(in_gt & in_prediction)
        fp = np.count_nonzero(~in_gt & in_prediction)
        fn = np.count_nonzero(in_gt & ~in_prediction)
        scores.append(2.0 * tp / max(float(2 * tp + fp + fn), 1e-8))
    return scores


def per_image_remapped_dice(pairs, n_classes: int) -> dict:
    """pairs: iterable of (sample_id, raw partition, integer GT). Returns per-image and aggregate Dice."""
    if int(n_classes) < 2:
        raise ValueError("n_classes must include background and at least one foreground label")
    rows = []
    for sample_id, partition, gt in pairs:
        gt = np.asarray(gt)
        if not np.issubdtype(gt.dtype, np.integer):
            raise ValueError("GT must be an integer label map")
        mapping, branch = match_segments(partition, gt)
        per_class = image_dice(remap(partition, mapping), gt, n_classes)
        rows.append({"sample_id": sample_id, "branch": branch, "dice": float(np.mean(per_class)),
                     "dice_per_foreground_label": per_class,
                     "mapping": {str(k): v for k, v in sorted(mapping.items())}})
    if not rows:
        raise ValueError("no samples to evaluate")
    values = np.array([row["dice"] for row in rows])
    return {"n_images": len(rows), "dice_mean": float(values.mean()), "dice_std_population": float(values.std()),
            "images": rows}
