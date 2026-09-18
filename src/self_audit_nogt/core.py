"""Minimal anonymous partition baseline with explicit, abstaining anatomy naming.

The ontology is reused unchanged. Anonymous cluster identities are NOT anatomy.
No training or method function accepts a manual segmentation reference.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torch import nn
from torch.nn import functional as F

from self_audit_maskfree.contracts import FittingView
from self_audit_maskfree.ontology import resolve_roles

UNKNOWN = 255


def canonicalize(partition: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Image-only cluster alignment; missing IDs are legal, tie by original ID."""
    groups = np.unique(partition)
    order = sorted(groups, key=lambda k: (float(image[partition == k].mean()), int(k)))
    result = np.zeros_like(partition, dtype=np.uint8)
    for new, old in enumerate(order):
        result[partition == old] = new
    return result


def cluster_image(image: np.ndarray, iterations: int = 15) -> np.ndarray:
    """Deterministic Lloyd iterations, four anonymous appearance groups."""
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    features = np.stack((image, uniform_filter(image, 5, mode="reflect"),
                         0.10 * yy, 0.10 * xx), axis=-1).reshape(-1, 4).astype(np.float32)
    quantiles = np.quantile(image, [0.125, 0.375, 0.625, 0.875])
    centers = np.stack((quantiles, quantiles, np.zeros(4), np.zeros(4)), axis=1).astype(np.float32)
    labels = np.zeros(h * w, dtype=np.int64)
    for _ in range(iterations):
        distance = ((features[:, None] - centers[None]) ** 2).sum(-1)
        labels = distance.argmin(1)
        for k in range(4):
            keep = labels == k
            if keep.any():
                centers[k] = features[keep].mean(0)
    return canonicalize(labels.reshape(h, w), image)


def name_partition(image: np.ndarray, partition: np.ndarray, unit: str = "unit") -> dict:
    """Run the production resolver on the actual FOV, never on padding."""
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None]
    view = FittingView(x, torch.ones_like(x[0], dtype=torch.bool), x.expand(3, -1, -1),
                       unit.split("_")[0], unit, metadata={})
    h = resolve_roles(torch.from_numpy(np.ascontiguousarray(partition)).long(), view)
    draft = h.labels.numpy().astype(np.uint8)
    valid = h.validity.numpy() > 0
    named = np.where(valid, draft, UNKNOWN).astype(np.uint8)
    return {"named": named, "draft": draft, "valid": valid,
            "unresolved": bool(h.semantic_unresolved), "trace": h.metadata["ontology_trace"]}


class SmallAnnotator(nn.Module):
    """Shared one-pass CNN; scratch initialization, no external weights or state."""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(12, 12, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(60, 12, 1), nn.ReLU(), nn.Conv2d(12, 4, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        local = self.enc1(image)
        deep = self.enc3(self.enc2(local))
        deep = F.interpolate(deep, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat((local, deep), dim=1))


def direct_loss(logits: torch.Tensor, image: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Soft piecewise-constant fit + TV + conditional entropy. No balanced-area prior."""
    p = logits.softmax(1)
    m = support[:, None].to(p.dtype)
    mass = (p * m).sum((2, 3), keepdim=True).clamp_min(1e-6)
    means = (p * m * image).sum((2, 3), keepdim=True) / mass
    distortion = (p * (image - means).square() * m).sum() / m.sum().clamp_min(1)
    dh = (p[:, :, 1:] - p[:, :, :-1]).abs()
    dw = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs()
    mh = m[:, :, 1:] * m[:, :, :-1]
    mw = m[:, :, :, 1:] * m[:, :, :, :-1]
    tv = (dh * mh).sum() / (4 * mh.sum()).clamp_min(1)
    tv = tv + (dw * mw).sum() / (4 * mw.sum()).clamp_min(1)
    entropy = (-(p * p.clamp_min(1e-8).log()).sum(1) * support).sum() / support.sum().clamp_min(1)
    return distortion + 0.01 * tv + 0.01 * entropy


def cache_loss(logits: torch.Tensor, partition: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    per_pixel = F.cross_entropy(logits, partition.long(), reduction="none")
    return (per_pixel * support).sum() / support.sum().clamp_min(1)
