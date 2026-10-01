"""Independent source-evidenced semantic matching branch, evaluator-only.

Equal-count Hungarian IoU only. Unreachable for native runs while run.py keeps the gate.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from shared_benchmark.native_protocol import ProtocolBlocked


def semantic_match(partition, gt):
    if partition.shape != gt.shape:
        raise ValueError("native evaluation grid mismatch")
    raw_ids, classes = np.unique(partition), np.unique(gt)
    if len(raw_ids) != len(classes):
        raise ProtocolBlocked(["Unequal-count branch is the pinned eval_utils.majority_vote_exclusive; its "
                               "paper-row applicability, remapping and aggregation are unresolved; not reimplemented"])
    scores = np.empty((len(raw_ids), len(classes)))
    for i, raw_id in enumerate(raw_ids):
        a = partition == raw_id
        for j, semantic_class in enumerate(classes):
            b = gt == semantic_class
            scores[i, j] = np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1e-8)
    rows, columns = linear_sum_assignment(-scores)
    mapping = {int(raw_ids[i]): int(classes[j]) for i, j in zip(rows, columns)}
    reordered = np.zeros_like(gt)
    for raw_id, semantic_class in mapping.items():
        reordered[partition == raw_id] = semantic_class
    return reordered, mapping
