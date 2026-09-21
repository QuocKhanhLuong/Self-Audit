"""Integrated research scaffold for Self-Audit v3.

Two deliberately separated paths:
1) CinePseudoTeacher: offline, image-only pseudo-label generation.
2) AdaptiveAnnotationStudent: deployment model trained from frozen named pseudo-labels.
   It reuses the canonical Dynamic Window AnnotationExpert in feature-only mode.

This is a research scaffold, not a validated ACDC model.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from self_audit.models.annotation_expert import AnnotationExpert

NUM_CLASSES = 4
UNKNOWN = 255


def _gn(c: int) -> nn.GroupNorm:
    for g in (8, 4, 2):
        if c % g == 0 and c // g >= 2:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class Block(nn.Module):
    def __init__(self, ci: int, co: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False), _gn(co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1, bias=False), _gn(co), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class AppearanceEncoder(nn.Module):
    """2.5-D appearance encoder: z-1/z/z+1 are channels."""
    def __init__(self, width: int = 24, dim: int = 48):
        super().__init__()
        self.e0 = Block(3, width)
        self.e1 = Block(width, width * 2, 2)
        self.e2 = Block(width * 2, width * 4, 2)
        self.out = nn.Conv2d(width * 4, dim, 1)
        self.recon = nn.Conv2d(dim, 1, 1)
    def forward(self, x):
        h = self.e2(self.e1(self.e0(x)))
        f = self.out(h)
        return f, self.recon(F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False))


class MotionBranch(nn.Module):
    """Cheap cine motion descriptor from t-1/t/t+1 center slices.

    The full experiment can replace this branch with unsupervised registration;
    the interface intentionally stays the same.
    """
    def __init__(self, dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, dim, 3, padding=1, bias=False), _gn(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=4, padding=1),
        )
    def forward(self, prev, cur, nxt):
        p, c, n = prev[:, 1:2], cur[:, 1:2], nxt[:, 1:2]
        raw = torch.cat([c - p, n - c, (c - p).abs(), (n - c).abs()], 1)
        return self.net(raw)


class RegionPrototypeHead(nn.Module):
    def __init__(self, dim: int, k: int = 12, temperature: float = 0.1):
        super().__init__()
        self.p = nn.Parameter(torch.randn(k, dim) * 0.02)
        self.temperature = float(temperature)
    def forward(self, f):
        f = F.normalize(f, dim=1)
        p = F.normalize(self.p, dim=1)
        logits = torch.einsum("bchw,kc->bkhw", f, p) / self.temperature
        return logits.softmax(1)


def pool_regions(assign, feat, image, motion):
    b, k, h, w = assign.shape
    mass = assign.sum((-2, -1)).clamp_min(1e-6)
    f = torch.einsum("bkhw,bchw->bkc", assign, feat) / mass[..., None]
    img = F.interpolate(image, (h, w), mode="bilinear", align_corners=False)
    mot = F.interpolate(motion.mean(1, keepdim=True), (h, w), mode="bilinear", align_corners=False)
    inten = torch.einsum("bkhw,bchw->bkc", assign, img) / mass[..., None]
    mov = torch.einsum("bkhw,bchw->bkc", assign, mot) / mass[..., None]
    area = (mass / float(h * w))[..., None]
    return torch.cat([f, inten, mov, area], -1)


class CinePseudoTeacher(nn.Module):
    """Offline pseudo-label teacher. It may abstain; UNKNOWN is never trainable."""
    def __init__(self, width=24, appearance_dim=48, motion_dim=16, fused_dim=64, k=12):
        super().__init__()
        self.appearance = AppearanceEncoder(width, appearance_dim)
        self.motion = MotionBranch(motion_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(appearance_dim + motion_dim, fused_dim, 1, bias=False), _gn(fused_dim), nn.GELU()
        )
        self.regions = RegionPrototypeHead(fused_dim, k)
        region_dim = fused_dim + 3
        self.semantic = nn.Sequential(nn.Linear(region_dim, 96), nn.GELU(), nn.Linear(96, NUM_CLASSES))

    def forward(self, prev, cur, nxt, *, evidence_logits=None, min_prob=0.70, min_margin=0.20):
        app, recon = self.appearance(cur)
        motion = self.motion(prev, cur, nxt)
        fused = self.fuse(torch.cat([app, motion], 1))
        q = self.regions(fused)
        r = pool_regions(q, fused, cur[:, 1:2], motion)
        logits = self.semantic(r)
        if evidence_logits is not None:
            if evidence_logits.shape != logits.shape:
                raise ValueError("evidence_logits must be [B,K,4]")
            logits = logits + evidence_logits
        prob = logits.softmax(-1)
        top2 = prob.topk(2, -1).values
        valid_region = (top2[..., 0] >= min_prob) & ((top2[..., 0] - top2[..., 1]) >= min_margin)
        dense_prob_low = torch.einsum("bkhw,bkc->bchw", q, prob)
        dense_valid_low = torch.einsum("bkhw,bk->bhw", q, valid_region.to(q.dtype)) >= 0.5
        dense_prob = F.interpolate(dense_prob_low, cur.shape[-2:], mode="bilinear", align_corners=False)
        dense_valid = F.interpolate(dense_valid_low[:, None].float(), cur.shape[-2:], mode="nearest")[:, 0].bool()
        # Confident regions may disagree on their names. Apply the acceptance
        # contract to the final pixel mixture, including interpolation boundaries.
        pixel_top2 = dense_prob.topk(2, dim=1).values
        dense_valid = dense_valid & (pixel_top2[:, 0] >= min_prob) & (
            (pixel_top2[:, 0] - pixel_top2[:, 1]) >= min_margin
        )
        label = dense_prob.argmax(1)
        pseudo = torch.where(dense_valid, label, torch.full_like(label, UNKNOWN))
        return {"pseudo_label": pseudo, "valid": dense_valid, "soft_label": dense_prob,
                "region_prob": q, "semantic_prob": prob, "reconstruction": recon,
                "appearance_features": app, "motion_features": motion}


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    turns: int


PROFILES = {
    "compact": ResourceProfile("compact", 0),
    "balanced": ResourceProfile("balanced", 1),
    "accurate": ResourceProfile("accurate", 2),
}


class DeploymentEncoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(Block(3, width), Block(width, width, 2), Block(width, width, 2))
    def forward(self, x):
        return self.net(x)


class AdaptiveAnnotationStudent(nn.Module):
    """Final annotation model: A0 plus optional Dynamic Window refinement."""
    def __init__(self, width=32, window_k=8):
        super().__init__()
        self.encoder = DeploymentEncoder(width)
        self.a0_head = nn.Conv2d(width, NUM_CLASSES, 1)
        self.refiner = AnnotationExpert(
            feature_channels=width, num_classes=NUM_CLASSES, audit_channels=3,
            window_k=window_k, max_turns=2, audit_conditioning="feature_only",
            offset_mode="structured",
        )

    def forward(self, x, *, profile="balanced", return_metadata=False):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {sorted(PROFILES)}")
        feat = self.encoder(x)
        a0 = F.interpolate(self.a0_head(feat), x.shape[-2:], mode="bilinear", align_corners=False)
        logits = a0
        stages = [a0]
        metadata = []
        for turn in range(PROFILES[profile].turns):
            out = self.refiner(feat, logits, previous_audit_evidence=None,
                               turn_index=turn, return_metadata=return_metadata)
            logits = out.candidate_logits
            stages.append(logits)
            if return_metadata:
                metadata.append(out.window_metadata)
        return {"a0_logits": a0, "final_logits": logits, "stages": tuple(stages),
                "profile": profile, "window_metadata": tuple(metadata)}


def pseudo_supervision_loss(outputs, target, valid, a0_weight=0.25):
    """Train the deployment annotator only on frozen accepted pseudo-label pixels."""
    if valid.dtype != torch.bool or valid.shape != target.shape:
        raise ValueError("target/valid must be [B,H,W] and valid must be bool")
    final = outputs["final_logits"]
    a0 = outputs["a0_logits"]
    accepted = valid & (target != UNKNOWN)
    if not bool(accepted.any()):
        return final.sum() * 0.0
    def ce(x):
        # Select before CE: UNKNOWN must never be passed as a class index,
        # even if a stale validity mask incorrectly marks that pixel accepted.
        return F.cross_entropy(x.permute(0, 2, 3, 1)[accepted], target[accepted])
    return ce(final) + float(a0_weight) * ce(a0)
