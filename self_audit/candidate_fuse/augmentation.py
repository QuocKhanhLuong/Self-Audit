"""Locked, reproducible in-plane training augmentations for Candidate FUSE."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.nn import functional as F


_PROBABILITIES = {
    "rotate": .5, "scale": .5, "elastic": .2, "flip_y": .5,
    "flip_x": .5, "rot90": .5, "gamma": .3, "brightness": .15,
    "contrast": .15, "noise": .1, "blur": .2, "low_resolution": .25,
}


class StrongInplaneAugment:
    """Frame-wise augmentation with one private CPU RNG and no z mixing."""

    def __init__(self, seed: int = 42, *, probabilities: Mapping[str, float] | None = None):
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) + 2003)
        self.probabilities = dict(_PROBABILITIES)
        if probabilities is not None:
            unknown = set(probabilities) - set(self.probabilities)
            if unknown:
                raise ValueError(f"unknown augmentation probabilities: {sorted(unknown)}")
            for key, value in probabilities.items():
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ValueError(f"probability {key} must be in [0,1]")
                self.probabilities[key] = float(value)

    def _rand(self) -> float:
        return float(torch.rand((), generator=self.generator))

    def _uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self._rand()

    def _enabled(self, name: str) -> bool:
        return self._rand() < self.probabilities[name]

    def state_dict(self) -> dict:
        return {"generator_state": self.generator.get_state().clone(), "probabilities": dict(self.probabilities)}

    def load_state_dict(self, state: Mapping) -> None:
        if set(state) != {"generator_state", "probabilities"} or dict(state["probabilities"]) != self.probabilities:
            raise ValueError("augmentation RNG state or probability contract mismatch")
        self.generator.set_state(state["generator_state"].cpu())

    @staticmethod
    def _validate(image: Tensor, target: Tensor | None, geometry: dict) -> None:
        if image.dtype != torch.float32 or image.ndim != 5 or tuple(image.shape[:2]) != (1, 1):
            raise ValueError("image must be float32 [1,1,D,H,W]")
        if not torch.isfinite(image).all() or bool(((image < 0) | (image > 1)).any()):
            raise ValueError("image must be finite and normalized to [0,1]")
        if target is not None:
            if target.dtype != torch.long or target.shape != (1, image.shape[2], image.shape[3], image.shape[4]):
                raise ValueError("target must be long [1,D,H,W]")
            if bool(((target < 0) | (target > 3)).any()):
                raise ValueError("target labels must be in 0..3")
        if tuple(geometry.get("shape", ())) != tuple(image.shape[2:]):
            raise ValueError("geometry shape must match image native shape")

    @staticmethod
    def _warp(image: Tensor, target: Tensor | None, matrix: Tensor, displacement: Tensor | None = None):
        # matrix maps output pixel coordinates to input coordinates; normalized
        # grid uses align_corners=False so the original canvas remains centered.
        d, h, w = image.shape[2:]
        device = image.device
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device), torch.arange(w, dtype=torch.float32, device=device), indexing="ij")
        center_y, center_x = (h - 1) / 2, (w - 1) / 2
        matrix = matrix.to(device=device, dtype=torch.float32)
        coords = torch.stack((xx - center_x, yy - center_y), dim=-1) @ matrix.T
        x, y = coords[..., 0] + center_x, coords[..., 1] + center_y
        if displacement is not None:
            displacement = displacement.to(device=device, dtype=torch.float32)
            y, x = y + displacement[0], x + displacement[1]
        gx = (2 * x + 1) / w - 1
        gy = (2 * y + 1) / h - 1
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(d, -1, -1, -1)
        slices = image[0, 0].unsqueeze(1)
        warped = F.grid_sample(slices, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        out_image = warped[:, 0].unsqueeze(0).unsqueeze(0)
        out_target = None
        if target is not None:
            masks = target[0].to(torch.float32).unsqueeze(1)
            out_target = F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].round().long().unsqueeze(0)
        return out_image, out_target

    def spatial(self, image: Tensor, target: Tensor | None, geometry: dict):
        # J runs inside model autocast. Keep coordinate construction and mask
        # sampling in fp32: bf16 pixel coordinates lose native subpixel detail.
        with torch.autocast(device_type=image.device.type, enabled=False):
            return self._spatial_impl(image, target, geometry)

    def _spatial_impl(self, image: Tensor, target: Tensor | None, geometry: dict):
        self._validate(image, target, geometry)
        if not any(self.probabilities[name] for name in ("rotate", "scale", "elastic", "flip_y", "flip_x", "rot90")):
            return image, target, geometry
        out, lab = image.clone(), target.clone() if target is not None else None
        # Apply each operation in specification order. This deliberately keeps
        # elastic offsets in the rotated/scaled pixel coordinate system.
        h, w = image.shape[-2:]
        if self._enabled("rotate"):
            angle = np.deg2rad(self._uniform(-180, 180))
            c, s = float(np.cos(angle)), float(np.sin(angle))
            out, lab = self._warp(out, lab, torch.tensor([[c, -s], [s, c]], dtype=torch.float32))
        if self._enabled("scale"):
            scale = self._uniform(.7, 1.4)
            out, lab = self._warp(out, lab, torch.eye(2) / scale)
        if self._enabled("elastic"):
            alpha, sigma = self._uniform(0, 200), self._uniform(9, 13)
            raw = torch.rand((2, h, w), generator=self.generator).numpy() * 2 - 1
            field = np.stack([gaussian_filter(raw[i], sigma=sigma, mode="constant") for i in range(2)])
            # scipy's normalized Gaussian kernel has unit sum; batchgenerators
            # smooths uniform noise with the same sigma then applies alpha.
            displacement = torch.from_numpy(field).to(torch.float32) * alpha
            out, lab = self._warp(out, lab, torch.eye(2), displacement)
        if self._enabled("flip_y"):
            out = torch.flip(out, dims=(-2,))
            if lab is not None:
                lab = torch.flip(lab, dims=(-2,))
        if self._enabled("flip_x"):
            out = torch.flip(out, dims=(-1,))
            if lab is not None:
                lab = torch.flip(lab, dims=(-1,))
        spacing = geometry.get("spacing_zyx")
        safe_turn = spacing is not None and len(spacing) == 3 and abs(float(spacing[1]) - float(spacing[2])) <= 1e-3
        if safe_turn and self._enabled("rot90"):
            k = int(torch.randint(1, 4, (), generator=self.generator))
            angle = k * np.pi / 2
            c, s = float(np.cos(angle)), float(np.sin(angle))
            out, lab = self._warp(out, lab, torch.tensor([[c, -s], [s, c]], dtype=torch.float32))
        return out, lab, geometry

    def intensity(self, image: Tensor) -> Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            return self._intensity_impl(image)

    def _intensity_impl(self, image: Tensor) -> Tensor:
        if image.dtype != torch.float32 or image.ndim != 5 or tuple(image.shape[:2]) != (1, 1):
            raise ValueError("intensity image must be float32 [1,1,D,H,W]")
        if not torch.isfinite(image).all() or bool(((image < 0) | (image > 1)).any()):
            raise ValueError("intensity image must be finite and normalized to [0,1]")
        if not any(self.probabilities[k] for k in ("gamma", "brightness", "contrast", "noise", "blur", "low_resolution")):
            return image
        out = image.clone()
        if self._enabled("gamma"):
            out = out.clamp_min(0).pow(self._uniform(.7, 1.5))
        if self._enabled("brightness"):
            out = out * self._uniform(.75, 1.25)
        if self._enabled("contrast"):
            mean = out.mean(dim=(-2, -1), keepdim=True)
            out = (out - mean) * self._uniform(.75, 1.25) + mean
        if self._enabled("noise"):
            std = self._uniform(0, .1)
            noise = torch.randn(out.shape, generator=self.generator, dtype=torch.float32)
            out = out + noise.to(out.device) * std
        if self._enabled("blur"):
            sigma = self._uniform(.5, 1)
            radius = max(1, int(np.ceil(3 * sigma)))
            coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-coords.square() / (2 * sigma * sigma)); kernel /= kernel.sum()
            k2 = torch.outer(kernel, kernel).to(out.device)
            flat = out.reshape(-1, 1, *out.shape[-2:])
            flat = F.conv2d(F.pad(flat, (radius, radius, radius, radius), mode="replicate"), k2[None, None])
            out = flat.reshape_as(out)
        if self._enabled("low_resolution"):
            scale = self._uniform(.5, 1)
            h, w = out.shape[-2:]
            sh, sw = max(1, round(h * scale)), max(1, round(w * scale))
            low = F.interpolate(out.reshape(-1, 1, h, w), size=(sh, sw), mode="nearest")
            out = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False).reshape_as(out)
        return out.clamp(0, 1)

    def teacher_view(self, image: Tensor, target: Tensor | None, geometry: dict):
        image, target, geometry = self.spatial(image, target, geometry)
        return self.intensity(image), target, geometry

    def joint_views(self, image: Tensor, target: Tensor | None, geometry: dict):
        spatial_image, target, geometry = self.spatial(image, target, geometry)
        return spatial_image, self.intensity(spatial_image), target, geometry
