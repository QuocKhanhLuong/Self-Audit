"""Dense labeled native loss and reproducible, aligned intensity augmentation."""
import torch
from torch.nn import functional as F


def supervised_loss(logits, labels, *, return_components=False, deterministic_ce=False):
    if labels.shape != logits.shape[:1] + logits.shape[2:] or logits.shape[1] != 4:
        raise ValueError("native logits and labels must have identical spatial shape")
    valid = labels != 255
    if not valid.any():
        raise ValueError("empty labeled support")
    if deterministic_ce:
        # Same valid-pixel mean CE; CUDA's spatial NLL reduction uses atomics.
        # The flattened NLL kernel supports strict deterministic execution.
        ce = F.cross_entropy(logits.movedim(1, -1).reshape(-1, 4), labels.reshape(-1), ignore_index=255)
    else:
        ce = F.cross_entropy(logits, labels, ignore_index=255)
    y = F.one_hot(labels.clamp_max(3), 4).permute(0, 3, 1, 2).to(logits.dtype) * valid[:, None]
    p = logits.softmax(1) * valid[:, None]
    dims = (0, 2, 3)
    den = p.sum(dims) + y.sum(dims)
    dice = (2 * (p * y).sum(dims) + 1e-5) / (den + 1e-5)
    present = y[:, 1:].sum(dims) > 0
    dl = (1 - dice[1:][present]).mean() if present.any() else logits.sum() * 0
    total = ce + dl
    if return_components:
        return total, {"ce": float(ce.detach()), "dice": float(dl.detach())}
    return total


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
