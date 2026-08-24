"""Geometry-preserving sample transforms for 2.5-D Self-Audit inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


def _transform_pair(sample: dict[str, Any], operation: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, Any]:
    image = sample["image"]
    mask = sample["mask"]
    if not torch.is_tensor(image) or image.ndim != 3:
        raise ValueError("sample['image'] must be [3,H,W]")
    if image.shape[0] != 3:
        raise ValueError(f"sample['image'] must have exactly 3 neighboring slices, got {tuple(image.shape)}")
    if not torch.is_tensor(mask) or mask.ndim != 2:
        raise ValueError("sample['mask'] must be [H,W]")
    transformed = dict(sample)
    transformed["image"] = operation(image)
    transformed["mask"] = operation(mask)
    return transformed


class Compose:
    def __init__(self, transforms: Iterable[Callable[[dict[str, Any]], dict[str, Any]]]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = float(probability)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if torch.rand(()) < self.probability:
            return _transform_pair(sample, lambda tensor: torch.flip(tensor, dims=(-1,)))
        return sample


class RandomVerticalFlip:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = float(probability)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if torch.rand(()) < self.probability:
            return _transform_pair(sample, lambda tensor: torch.flip(tensor, dims=(-2,)))
        return sample


class RandomRotate90:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = float(probability)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if torch.rand(()) >= self.probability:
            return sample
        quarter_turns = int(torch.randint(1, 4, ()).item())
        return _transform_pair(sample, lambda tensor: torch.rot90(tensor, quarter_turns, dims=(-2, -1)))


class RandomGeometricTransform:
    """Default lightweight augmentation shared by all three image channels."""

    def __init__(self, probability: float = 0.5) -> None:
        self.transform = Compose(
            (
                RandomHorizontalFlip(probability),
                RandomVerticalFlip(probability),
                RandomRotate90(probability),
            )
        )

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self.transform(sample)


def validate_geometric_transform(sample_before: dict[str, Any], sample_after: dict[str, Any]) -> None:
    """Check the shared contract after a custom transform in tests."""

    image_before = sample_before["image"]
    image_after = sample_after["image"]
    mask_after = sample_after["mask"]
    if image_before.shape[0] != image_after.shape[0] or image_after.shape[0] != 3:
        raise ValueError("A geometric transform must preserve the three-slice channel axis")
    if tuple(image_after.shape[-2:]) != tuple(mask_after.shape[-2:]):
        raise ValueError("Image and center mask must retain matching spatial geometry")
