"""Frozen M3HL 2D U-Net as a 3D teacher for the W2S-Match student step (``pg.teacher_source = "m3hl"``).

M3HL (MICCAI 2025, ``PHPJava666/M3HL``, pinned commit 19151b3) is a 2D U-Net trained on SSL4MIS ACDC slices.  The
student step needs class probabilities on the student's own 3D grid, so this wrapper applies the official slice-wise
test path to a weak view ``[B,1,D,H,W]``:

  1. per sample, min-max normalise the whole 3D volume to [0,1] (SSL4MIS preprocessing is per 3D volume; our views are
     percentile-clipped cine intensities, so min-max of the view equals the raw min-max up to the clipped outliers);
  2. per slice, resize to 256x256 (``zoom`` order 0 in the official code; here nearest, identical for integer factors
     and for the same index rounding as ``scipy.ndimage.zoom`` up to boundary pixels);
  3. run the frozen U-Net (the first output if the network returns a tuple), softmax;
  4. resize the probabilities back to the native slice size with **bilinear** interpolation (the official code resizes
     labels with order 0; probabilities need a smooth resize for the arbiter and for KD), renormalise over classes.

The wrapper is an ``nn.Module`` with the interface ``_teacher_output`` expects: ``forward(image, geometry,
return_features=True)`` returns ``{"region_logits": log p, "features": None}`` so ``softmax(region_logits) == p``.  It
has no gradients and no feature levels (the M3HL U-Net has no MedNeXt-like level-3 map on the student grid).
"""
from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

M3HL_COMMIT = "19151b3"
M3HL_INPUT = 256
M3HL_CLASSES = 4


def minmax_per_volume(image: Tensor, eps: float = 1e-8) -> Tensor:
    """Min-max normalise each sample of ``[B,1,D,H,W]`` over its whole volume (SSL4MIS per-volume preprocessing)."""
    flat = image.flatten(1)
    low, high = flat.min(1).values, flat.max(1).values
    shape = (-1,) + (1,) * (image.ndim - 1)
    return (image - low.view(shape)) / (high - low).clamp_min(eps).view(shape)


class M3HLTeacher(nn.Module):
    """Frozen slice-wise M3HL network exposed as a 3D probability teacher."""

    feature_phases = None
    teacher_phases = None
    output_space = True  # CBKD-SA reads the output probabilities (cbkd_sa_out), not a feature level

    def __init__(self, net: nn.Module, *, input_size: int = M3HL_INPUT, num_classes: int = M3HL_CLASSES,
                 slice_batch: int = 32):
        super().__init__()
        self.net = net
        self.input_size, self.num_classes, self.slice_batch = int(input_size), int(num_classes), int(slice_batch)
        self.net.eval()
        self.net.requires_grad_(False)

    def train(self, mode: bool = True) -> "M3HLTeacher":  # the teacher is always in eval mode (BN statistics, dropout)
        super().train(False)
        return self

    @torch.no_grad()
    def probabilities(self, image: Tensor) -> Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("M3HL teacher expects [B,1,D,H,W]")
        batch, _, depth, height, width = image.shape
        volume = minmax_per_volume(image.float())
        slices = volume.permute(0, 2, 1, 3, 4).reshape(batch * depth, 1, height, width)
        resized = F.interpolate(slices, size=(self.input_size, self.input_size), mode="nearest")
        chunks = []
        for start in range(0, resized.shape[0], self.slice_batch):
            out = self.net(resized[start:start + self.slice_batch])
            logits = out[0] if isinstance(out, (tuple, list)) else out
            chunks.append(logits.float().softmax(1))
        prob = torch.cat(chunks)
        if (height, width) != (self.input_size, self.input_size):
            prob = F.interpolate(prob, size=(height, width), mode="bilinear", align_corners=False)
            prob = prob / prob.sum(1, keepdim=True).clamp_min(1e-8)
        return prob.reshape(batch, depth, self.num_classes, height, width).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, image: Tensor, geometry: Mapping[str, Any] | None = None, *, return_features: bool = True,
                **_: Any) -> Mapping[str, Any]:
        del geometry, return_features
        prob = self.probabilities(image)
        return {"region_logits": prob.clamp_min(1e-8).log(), "features": None, "probabilities": prob}


def _import_unet_hl(source_root: Path):
    code = source_root / "code"
    if not (code / "networks" / "unet.py").exists():
        raise FileNotFoundError(f"M3HL source not found at {code}")
    sys.path.insert(0, str(code))
    try:
        from networks.unet import UNet_HL  # type: ignore
    finally:
        sys.path.remove(str(code))
    return UNet_HL


def build_m3hl_teacher(checkpoint: Path, source_root: Path, *, device: str = "cpu") -> M3HLTeacher:
    """Load ``unet_best_model.pth`` (the official val-selected ``model`` weights) into a frozen teacher."""
    net = _import_unet_hl(Path(source_root))(in_chns=1, class_num=M3HL_CLASSES)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    return M3HLTeacher(net).to(device)


__all__ = ["M3HLTeacher", "build_m3hl_teacher", "minmax_per_volume", "M3HL_COMMIT"]
