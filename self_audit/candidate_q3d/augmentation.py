"""Native-geometry-preserving Q-3D spatial and intensity augmentations."""

from __future__ import annotations

import math
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from ..candidate_a3d.geometry import canonical_geometry


def _image_shape(image: Tensor) -> tuple[int, int, int]:
    if not isinstance(image, Tensor) or image.ndim != 5 or tuple(image.shape[:2]) != (1, 1):
        raise ValueError("Q-3D native image must have shape [1,1,D,H,W]")
    return tuple(int(value) for value in image.shape[-3:])


def _updated_geometry(geometry: Mapping[str, Any], shape, spacing, affine) -> dict[str, Any]:
    updated = dict(geometry)
    updated["shape"] = tuple(int(v) for v in shape)
    updated["spacing_zyx"] = tuple(float(v) for v in spacing)
    updated["affine_zyx"] = np.asarray(affine, dtype=np.float64).tolist()
    return canonical_geometry(updated)


def _flip_geometry(geometry: dict[str, Any], axis: int) -> dict[str, Any]:
    """Map transformed indices back into the original voxel/world coordinates."""
    affine = np.asarray(geometry["affine_zyx"], dtype=np.float64)
    linear = affine[:3, :3].copy()
    translation = affine[:3, 3].copy()
    size = int(geometry["shape"][axis])
    translation += linear[:, axis] * (size - 1)
    linear[:, axis] *= -1.0
    affine[:3, :3] = linear
    affine[:3, 3] = translation
    return _updated_geometry(geometry, geometry["shape"], geometry["spacing_zyx"], affine)


def _rotate90_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    """Update geometry for torch.rot90(..., k=1, dims=(-2,-1))."""
    affine = np.asarray(geometry["affine_zyx"], dtype=np.float64)
    old_linear = affine[:3, :3].copy()
    old_width = int(geometry["shape"][2])
    affine[:3, 3] += old_linear[:, 2] * (old_width - 1)
    affine[:3, :3] = old_linear
    affine[:3, 1] = -old_linear[:, 2]
    affine[:3, 2] = old_linear[:, 1]
    depth, height, width = geometry["shape"]
    dz, dy, dx = geometry["spacing_zyx"]
    return _updated_geometry(geometry, (depth, width, height), (dz, dx, dy), affine)


def apply_spatial_transform(
    image: Tensor,
    mask: Tensor | None,
    geometry: Mapping[str, Any],
    *,
    flip_y: bool = False,
    flip_x: bool = False,
    rot90_k: int = 0,
) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
    """Apply exact native flips/rotations to image and optional label mask.

    SDF targets are intentionally not accepted or returned; callers recompute
    them from the transformed labeled mask and returned geometry.
    """

    if type(flip_y) is not bool or type(flip_x) is not bool:
        raise ValueError("flip_y and flip_x must be booleans")
    if type(rot90_k) is not int or not 0 <= rot90_k <= 3:
        raise ValueError("rot90_k must be an integer from zero through three")
    current = canonical_geometry(geometry)
    if _image_shape(image) != tuple(current["shape"]):
        raise ValueError("image shape differs from native geometry")
    if not torch.isfinite(image).all() or bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("Q-3D spatial image must be finite and normalized to [0,1]")
    if mask is not None:
        if (not isinstance(mask, Tensor) or mask.ndim != 4 or int(mask.shape[0]) != 1
                or tuple(int(value) for value in mask.shape[-3:]) != tuple(current["shape"])):
            raise ValueError("Q-3D labeled mask must have shape [1,D,H,W] matching geometry")
        if mask.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64} \
                or bool(((mask < 0) | (mask > 3)).any()):
            raise ValueError("Q-3D labeled mask must contain only integer classes 0 through 3")
    if rot90_k and abs(float(current["spacing_zyx"][1]) - float(current["spacing_zyx"][2])) > 1e-3:
        raise ValueError("in-plane rot90 requires spacing_y and spacing_x to agree within 1e-3 mm")

    transformed_image, transformed_mask = image, mask
    if flip_y:
        transformed_image = torch.flip(transformed_image, dims=(-2,))
        if transformed_mask is not None:
            transformed_mask = torch.flip(transformed_mask, dims=(-2,))
        current = _flip_geometry(current, 1)
    if flip_x:
        transformed_image = torch.flip(transformed_image, dims=(-1,))
        if transformed_mask is not None:
            transformed_mask = torch.flip(transformed_mask, dims=(-1,))
        current = _flip_geometry(current, 2)
    for _ in range(rot90_k):
        transformed_image = torch.rot90(transformed_image, k=1, dims=(-2, -1))
        if transformed_mask is not None:
            transformed_mask = torch.rot90(transformed_mask, k=1, dims=(-2, -1))
        current = _rotate90_geometry(current)
    if _image_shape(transformed_image) != tuple(current["shape"]):
        raise RuntimeError("Q-3D spatial transform produced inconsistent native geometry")
    return transformed_image, transformed_mask, current


def sample_spatial_transform(
    geometry: Mapping[str, Any],
    *,
    rng: random.Random,
    flip_probability: float = 0.5,
    flip_y_probability: float | None = None,
    flip_x_probability: float | None = None,
    rotation_probability: float = 0.5,
) -> dict[str, Any]:
    """Sample the declared y/x flips and spacing-gated quarter-turn."""

    if not isinstance(rng, random.Random):
        raise TypeError("spatial augmentation requires a random.Random generator")
    flip_y_probability = flip_probability if flip_y_probability is None else flip_y_probability
    flip_x_probability = flip_probability if flip_x_probability is None else flip_x_probability
    for name, value in (("flip_y_probability", flip_y_probability),
                        ("flip_x_probability", flip_x_probability),
                        ("rotation_probability", rotation_probability)):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    current = canonical_geometry(geometry)
    flip_y = rng.random() < flip_y_probability
    flip_x = rng.random() < flip_x_probability
    equal_spacing = abs(current["spacing_zyx"][1] - current["spacing_zyx"][2]) <= 1e-3
    rotate = equal_spacing and rng.random() < rotation_probability
    rot90_k = rng.randint(1, 3) if rotate else 0
    return {"flip_y": flip_y, "flip_x": flip_x, "rot90_k": rot90_k}


def strong_intensity(
    image: Tensor,
    config: Mapping[str, Any] | None = None,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply per-volume gamma [0.7,1.5] and voxel noise std 0.03 by default."""

    if not isinstance(image, Tensor) or image.ndim != 5:
        raise ValueError("Q-3D intensity input must be [B,1,D,H,W]")
    if not torch.isfinite(image).all() or bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("Q-3D intensity input must be finite and normalized to [0,1]")
    section: Mapping[str, Any]
    if config is None:
        section = {}
    elif not isinstance(config, Mapping):
        raise TypeError("Q-3D augmentation config must be a mapping")
    else:
        raw = config.get("augmentation", config)
        if not isinstance(raw, Mapping):
            raise ValueError("config.augmentation must be a mapping")
        section = raw
    gamma_range = section.get("intensity_gamma", section.get("gamma_range", (0.7, 1.5)))
    if not isinstance(gamma_range, (tuple, list)) or len(gamma_range) != 2:
        raise ValueError("intensity_gamma must contain [low, high]")
    low, high = float(gamma_range[0]), float(gamma_range[1])
    noise_std = float(section.get("intensity_noise_std", section.get("voxel_noise_std", 0.03)))
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or low > high:
        raise ValueError("intensity_gamma must be finite, positive and ordered")
    if not math.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("intensity_noise_std must be finite and nonnegative")
    random_device = (torch.device(getattr(generator, "device", "cpu"))
                     if generator is not None else torch.device("cpu"))
    factors = torch.rand((int(image.shape[0]), 1, 1, 1, 1), generator=generator,
                         device=random_device, dtype=torch.float32)
    gamma = (low + (high - low) * factors).to(device=image.device, dtype=image.dtype)
    result = image.pow(gamma)
    if noise_std:
        noise = torch.randn(image.shape, generator=generator, device=random_device, dtype=torch.float32)
        result = result + noise.to(device=image.device, dtype=image.dtype) * noise_std
    return result.clamp(0.0, 1.0)


def maybe_strong_intensity(
    image: Tensor,
    config: Mapping[str, Any] | None = None,
    *,
    probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply the strong view with a deterministic optional probability gate."""

    if not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
        raise ValueError("intensity augmentation probability must be in [0,1]")
    random_device = (torch.device(getattr(generator, "device", "cpu"))
                     if generator is not None else torch.device("cpu"))
    draw = torch.rand((), generator=generator, device=random_device).item()
    return strong_intensity(image, config, generator=generator) if draw < probability else image


def augment_labeled(
    image: Tensor,
    mask: Tensor,
    geometry: Mapping[str, Any],
    *,
    rng: random.Random,
    intensity_config: Mapping[str, Any] | None = None,
    intensity_generator: torch.Generator | None = None,
    flip_probability: float | None = None,
    rotation_probability: float | None = None,
    intensity_probability: float | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Transform labeled image/mask, then optionally add strong intensity."""

    options = _augmentation_options(intensity_config)
    fallback_flip = 0.5 if flip_probability is None else flip_probability
    transform = sample_spatial_transform(
        geometry, rng=rng,
        flip_y_probability=float(options.get("flip_y_probability", fallback_flip)),
        flip_x_probability=float(options.get("flip_x_probability", fallback_flip)),
        rotation_probability=(float(options.get("rotate90_probability", 0.5))
                              if rotation_probability is None else rotation_probability),
        flip_probability=fallback_flip,
    )
    spatial_image, spatial_mask, result_geometry = apply_spatial_transform(
        image, mask, geometry, **transform,
    )
    assert spatial_mask is not None
    intensity_probability = (float(options.get("intensity_probability_labeled", 0.5))
                             if intensity_probability is None else intensity_probability)
    return (maybe_strong_intensity(spatial_image, intensity_config,
                                   probability=intensity_probability, generator=intensity_generator),
            spatial_mask, result_geometry)


def augment_unlabeled(
    image: Tensor,
    geometry: Mapping[str, Any],
    *,
    rng: random.Random,
    intensity_config: Mapping[str, Any] | None = None,
    intensity_generator: torch.Generator | None = None,
    flip_probability: float | None = None,
    rotation_probability: float | None = None,
    intensity_probability: float | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Return shared-spatial weak teacher and strong student U views."""

    options = _augmentation_options(intensity_config)
    fallback_flip = 0.5 if flip_probability is None else flip_probability
    transform = sample_spatial_transform(
        geometry, rng=rng,
        flip_y_probability=float(options.get("flip_y_probability", fallback_flip)),
        flip_x_probability=float(options.get("flip_x_probability", fallback_flip)),
        rotation_probability=(float(options.get("rotate90_probability", 0.5))
                              if rotation_probability is None else rotation_probability),
        flip_probability=fallback_flip,
    )
    weak, _, result_geometry = apply_spatial_transform(image, None, geometry, **transform)
    intensity_probability = (float(options.get("intensity_probability_unlabeled_student", 1.0))
                             if intensity_probability is None else intensity_probability)
    strong = maybe_strong_intensity(
        weak, intensity_config, probability=intensity_probability,
        generator=intensity_generator,
    )
    return weak, strong, result_geometry


def _augmentation_options(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("Q-3D augmentation config must be a mapping")
    section = config.get("augmentation", config)
    if not isinstance(section, Mapping):
        raise ValueError("config.augmentation must be a mapping")
    return section


__all__ = [
    "apply_spatial_transform", "sample_spatial_transform", "strong_intensity",
    "maybe_strong_intensity", "augment_labeled", "augment_unlabeled",
]
