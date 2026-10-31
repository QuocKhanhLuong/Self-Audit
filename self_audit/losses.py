"""Dense labeled native loss and reproducible, aligned intensity augmentation."""
import copy
import numpy as np
import torch
from torch.nn import functional as F


FOCAL_DICE_DEFAULT = {
    "name": "focal_dice",
    "gamma": 2.0,
    "focal_weight": 1.0,
    "dice_weight": 1.0,
}


def validate_supervised_loss_config(config):
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("supervised_loss must be a dict or None")
    c = copy.deepcopy(config)
    name = c.get("name")
    if name == "ce_dice":
        if set(c.keys()) != {"name"}:
            raise ValueError("ce_dice configuration only allows 'name' key")
        return c
    if name == "focal_dice":
        required = {"name", "gamma", "focal_weight", "dice_weight"}
        if set(c.keys()) != required:
            raise ValueError(f"focal_dice requires exact keys: {sorted(required)}")
        for k in ("gamma", "focal_weight", "dice_weight"):
            v = c[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise ValueError(f"supervised_loss {k} must be a finite non-negative number")
        if c["focal_weight"] == 0 and c["dice_weight"] == 0:
            raise ValueError("focal_weight and dice_weight cannot both be 0")
        return c
    raise ValueError(f"unknown supervised_loss name: {name}")


def focal_loss(logits, labels, gamma=2.0):
    """Multi-class focal loss across all 4 classes, valid-pixel denominator ignore255, no alpha."""
    if labels.shape != logits.shape[:1] + logits.shape[2:] or logits.shape[1] != 4:
        raise ValueError("native logits and labels must have identical spatial shape")
    valid = labels != 255
    if not valid.any():
        raise ValueError("empty labeled support")
    if (labels[valid] < 0).any() or (labels[valid] > 3).any():
        raise ValueError("labels must have values in {0, 1, 2, 3} or 255 (ignore)")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite non-negative number")
    if gamma == 0:
        return F.cross_entropy(logits, labels, ignore_index=255)
    log_p = F.log_softmax(logits, dim=1)
    target = labels.clone()
    target[~valid] = 0
    target = target.unsqueeze(1)
    log_pt = log_p.gather(1, target).squeeze(1)
    pt = log_pt.exp().clamp(0.0, 1.0)
    mod = (1.0 - pt).pow(gamma)
    pixel_loss = mod * (-log_pt)
    return (pixel_loss * valid).sum() / valid.sum()


def supervised_loss(logits, labels, config=None):
    if labels.shape != logits.shape[:1] + logits.shape[2:] or logits.shape[1] != 4:
        raise ValueError("native logits and labels must have identical spatial shape")
    valid = labels != 255
    if not valid.any():
        raise ValueError("empty labeled support")
    if (labels[valid] < 0).any() or (labels[valid] > 3).any():
        raise ValueError("labels must have values in {0, 1, 2, 3} or 255 (ignore)")

    if config is not None:
        config = validate_supervised_loss_config(config)

    # 1. Base loss: CE or Focal
    if config is None or config.get("name") == "ce_dice":
        base_loss = F.cross_entropy(logits, labels, ignore_index=255)
        dice_weight = 1.0
        focal_weight = 1.0
    elif config.get("name") == "focal_dice":
        gamma = float(config["gamma"])
        focal_weight = float(config["focal_weight"])
        dice_weight = float(config["dice_weight"])
        base_loss = focal_loss(logits, labels, gamma=gamma)
    else:
        raise ValueError(f"unsupported supervised loss: {config.get('name')}")

    # 2. Foreground Dice loss (identical to canonical v2 formula)
    y = F.one_hot(labels.clamp_max(3), 4).permute(0, 3, 1, 2).to(logits.dtype) * valid[:, None]
    p = logits.softmax(1) * valid[:, None]
    dims = (0, 2, 3)
    den = p.sum(dims) + y.sum(dims)
    dice = (2 * (p * y).sum(dims) + 1e-5) / (den + 1e-5)
    present = y[:, 1:].sum(dims) > 0
    dl = (1 - dice[1:][present]).mean() if present.any() else logits.sum() * 0

    if config is not None and config.get("name") == "focal_dice":
        return focal_weight * base_loss + dice_weight * dl
    return base_loss + dl


def strong_intensity(x, rng):
    """No spatial misregistration: output remains in teacher coordinates."""
    gamma = float(rng.uniform(.8, 1.2))
    noise = torch.from_numpy(rng.normal(0, .03, tuple(x.shape))).to(device=x.device, dtype=x.dtype)
    return (x.pow(gamma) + noise).clamp(0, 1)


def aligned_flip(x, y, rng, valid_mask=None):
    if rng.random() < .5:
        x, y = x.flip(-1), y.flip(-1)
        if valid_mask is not None:
            valid_mask = valid_mask.flip(-1)
    return (x, y, valid_mask) if valid_mask is not None else (x, y)
