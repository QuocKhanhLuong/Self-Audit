"""Inspectable tensor operators; proofs and boundaries are in THEORY.md."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def block_mean(x: Tensor, factor: int = 4) -> Tensor:
    """D: average disjoint cells. No hidden resize/padding is performed."""
    if x.ndim != 4 or type(factor) is not int or factor < 1:
        raise ValueError('expected NCHW and a positive integer factor')
    if x.shape[-2] % factor or x.shape[-1] % factor:
        raise ValueError('pad explicitly to a multiple of factor before projection')
    return F.avg_pool2d(x, factor, factor)


def block_lift(x: Tensor, factor: int = 4) -> Tensor:
    """U: replicate a coarse value over its cell, so D U = I."""
    if x.ndim != 4 or type(factor) is not int or factor < 1:
        raise ValueError('expected NCHW and positive integer factor')
    return x.repeat_interleave(factor, -2).repeat_interleave(factor, -1)


def zero_mean_detail(x: Tensor, factor: int = 4) -> Tensor:
    """(I-U D)x. Cell-local high-pass, not an ideal Fourier frequency split."""
    return x - block_lift(block_mean(x, factor), factor)


def compose_logits(coarse: Tensor, detail: Tensor, factor: int = 4) -> Tensor:
    up = block_lift(coarse, factor)
    if up.shape != detail.shape:
        raise ValueError('coarse/detail resolution or channels do not match')
    return up + zero_mean_detail(detail, factor)


class CenterDifferenceInput(nn.Module):
    """MRI [previous-z, center-z, next-z] -> pretrained three-channel stem input.

    Input must already be image-only normalized to [0,1]. The pretrained path
    sees the center repeated and then channel-normalized. A zero-initialized
    bias-free difference convolution learns slice context. No pretrained
    accuracy claim follows from the initialization equivalence.
    """
    def __init__(self, mean=(.485, .456, .406), std=(.229, .224, .225)):
        super().__init__()
        if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
            raise ValueError('three finite channel means/stds are required')
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        self.context = nn.Conv2d(2, 3, 3, padding=1, bias=False)
        nn.init.zeros_(self.context.weight)

    def forward(self, slices: Tensor) -> Tensor:
        if slices.ndim != 4 or slices.shape[1] != 3:
            raise ValueError('input must be [B,3,H,W] with center channel 1')
        if not torch.isfinite(slices).all() or slices.min() < 0 or slices.max() > 1:
            raise ValueError('expected finite image-only normalized values in [0,1]')
        center = slices[:, 1:2]
        base = (center.expand(-1, 3, -1, -1) - self.mean) / self.std
        diff = torch.cat((slices[:, :1]-center, slices[:, 2:3]-center), dim=1)
        return base + self.context(diff)
