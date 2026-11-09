"""Shared dense segmentation loss and aligned spatial augmentation."""
import torch
from torch.nn import functional as F


def supervised_loss(logits, labels):
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be tensors")
    if logits.ndim < 3 or labels.ndim != logits.ndim - 1:
        raise ValueError("logits must be [B,C,*spatial] and labels [B,*spatial]")
    if labels.shape != logits.shape[:1] + logits.shape[2:] or logits.shape[1] < 2:
        raise ValueError("logits and labels must have identical batch/spatial shape and at least two classes")
    if labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise TypeError("labels must be an integer class-index tensor")
    classes = int(logits.shape[1])
    valid = labels != 255
    if not valid.any():
        raise ValueError("empty labeled support")
    if bool((valid & ((labels < 0) | (labels >= classes))).any()):
        raise ValueError("labels contain an unknown class ID")
    labels = labels.long()
    ce = F.cross_entropy(logits, labels, ignore_index=255)
    y = F.one_hot(labels.clamp_min(0).clamp_max(classes - 1), classes)
    y = y.movedim(-1, 1).to(logits.dtype) * valid[:, None]
    p = logits.softmax(1) * valid[:, None]
    dims = (0, *range(2, logits.ndim))
    den = p.sum(dims) + y.sum(dims)
    dice = (2 * (p * y).sum(dims) + 1e-5) / (den + 1e-5)
    present = y[:, 1:].sum(dims) > 0
    dl = (1 - dice[1:][present]).mean() if present.any() else logits.sum() * 0
    return ce + dl


def aligned_flip(x, y, rng):
    if rng.random() < .5:
        return x.flip(-1), y.flip(-1)
    return x, y
