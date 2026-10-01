# SPDX-License-Identifier: GPL-3.0
# Official-code-derived context arithmetic. No softmax/absolute-density correction.
import numpy as np
import torch
import torch.nn.functional as F


def coordinate_tensors(height, width, device):
    x = np.tile(np.arange(height), (width, 1)) / height * 2 - 1.0
    y = np.tile(np.arange(width), (height, 1)).T / width * 2 - 1.0
    return (torch.from_numpy(x.astype(np.float32)).to(device).transpose(1, 0),
            torch.from_numpy(y.astype(np.float32)).to(device).transpose(1, 0))


def centers(logits, epsilon):
    _, height, width = logits.shape
    x, y = coordinate_tensors(height, width, logits.device)
    result = []
    for plane in logits:
        values = plane + epsilon
        density = values / values.sum()
        result.append(torch.stack(((density * x).sum(), (density * y).sum())))
    return torch.stack(result)


def context_loss(logits, epsilon):
    locations = centers(logits, epsilon)
    _, height, width = logits.shape
    x, y = coordinate_tensors(height, width, logits.device)
    total = 0
    for channel, plane in enumerate(logits):
        values = plane + epsilon
        density = values / values.sum()
        cx, cy = locations[channel]
        vx = (density * ((x - cx) * (x - cx))).sum()
        vy = (density * ((y - cy) * (y - cy))).sum()
        total = total + (vx + vy)
    return total


def loss_terms(logits, settings):
    channels, height, width = logits.shape
    flat = logits.permute(1, 2, 0).contiguous().view(-1, channels)
    target = flat.argmax(dim=1)
    spatial_grid = flat.reshape(height, width, channels)
    vertical = spatial_grid[1:] - spatial_grid[:-1]
    horizontal = spatial_grid[:, 1:] - spatial_grid[:, :-1]
    ce = F.cross_entropy(flat, target)
    spatial = F.l1_loss(vertical, torch.zeros_like(vertical)) + F.l1_loss(horizontal, torch.zeros_like(horizontal))
    context = context_loss(logits, settings["context_epsilon"]) if settings["context_loss"] else logits.new_zeros(())
    total = settings["ce_weight"] * ce + settings["spatial_weight"] * spatial + context
    return total, target, {"ce": ce, "spatial": spatial, "context": context}
