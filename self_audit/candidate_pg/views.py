"""Pure view and CutMix helpers for Candidate PG.

The PG track keeps the existing FUSE augmentation implementation as its only
spatial/intensity primitive.  This module adds mask-aware bookkeeping around
that primitive and a small checkpointable RNG wrapper.  All operations are
native-volume operations: CutMix is in-plane and spans the full depth.
"""

from __future__ import annotations

import math
from itertools import combinations
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from self_audit.candidate_fuse.augmentation import StrongInplaneAugment


CutMixBox = tuple[int, int, int, int]


class MixPacket(dict):
    """Dictionary packet with attribute access for engine readability."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - normal AttributeError semantics
            raise AttributeError(name) from exc


def _mask4(value: Tensor, *, name: str) -> Tensor:
    """Normalize a dense mask to ``[B,D,H,W]`` without changing its device."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim == 5 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [B,D,H,W] or [B,1,D,H,W]")
    if value.dtype.is_complex:
        raise TypeError(f"{name} must be real")
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return value


def _box_tuple(box: Sequence[int], shape: Sequence[int]) -> CutMixBox:
    if isinstance(box, Tensor):
        box = box.detach().cpu().tolist()
    if not isinstance(box, Sequence) or len(box) != 4:
        raise ValueError("CutMix box must be (y0, y1, x0, x1)")
    y0, y1, x0, x1 = (int(v) for v in box)
    h, w = int(shape[-2]), int(shape[-1])
    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise ValueError("CutMix box is outside the image canvas")
    return y0, y1, x0, x1


def _generator(generator: torch.Generator | None, *, seed: int = 0) -> torch.Generator:
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        return generator
    # Never consume torch's process-global RNG for a pure helper.
    result = torch.Generator(device="cpu")
    result.manual_seed(int(seed))
    return result


def _valid_intersection(valid: Tensor | Sequence[int]) -> tuple[Tensor | None, tuple[int, int]]:
    if isinstance(valid, Tensor):
        mask = _mask4(valid, name="valid").bool()
        if mask.shape[0] <= 0 or mask.shape[1] <= 0:
            raise ValueError("valid must have non-empty batch/depth")
        # Z padding contributes no acquired slice.  Choose the in-plane box
        # inside every acquired slice; mixed padded slices retain valid=0.
        slices = mask.flatten(0, 1)
        acquired = slices.flatten(1).any(1)
        intersection = slices[acquired].all(0) if acquired.any() else torch.zeros_like(slices[0])
        return intersection, (int(mask.shape[-2]), int(mask.shape[-1]))
    if not isinstance(valid, Sequence) or len(valid) != 2:
        raise TypeError("valid must be a tensor or a (height, width) shape")
    h, w = (int(v) for v in valid)
    if h <= 0 or w <= 0:
        raise ValueError("canvas shape must be positive")
    return None, (h, w)


def cutmix_box(
    valid: Tensor | Sequence[int],
    generator: torch.Generator | None = None,
    *,
    partner_valid: Tensor | None = None,
    area_fraction: float | Sequence[float] = (0.25, 0.5),
    align_px: int = 16,
    full_depth: bool = True,
) -> CutMixBox:
    """Sample an aligned in-plane CutMix box within the valid intersection.

    Coordinates and side lengths are multiples of ``align_px``.  Candidate
    boxes lie inside the common in-plane support of acquired slices. Entirely
    padded z slices keep zero validity after mixing. No unaligned fallback is
    allowed when a canvas cannot contain a box.
    """

    if isinstance(full_depth, bool) is False:
        raise TypeError("full_depth must be a bool")
    if not full_depth:
        raise ValueError("PG CutMix is defined as full-depth in-plane mixing")
    if isinstance(align_px, bool) or int(align_px) <= 0:
        raise ValueError("align_px must be a positive integer")
    align_px = int(align_px)
    support, (height, width) = _valid_intersection(valid)
    if partner_valid is not None:
        partner_support, partner_shape = _valid_intersection(partner_valid)
        if partner_shape != (height, width):
            raise ValueError("partner_valid must share the anchor canvas")
        if support is None:
            support = partner_support
        elif partner_support is not None:
            support = support & partner_support
    if support is not None and not bool(support.any()):
        raise ValueError("CutMix valid intersection is empty")
    if isinstance(area_fraction, (int, float)):
        low = high = float(area_fraction)
    elif isinstance(area_fraction, Sequence) and len(area_fraction) == 2:
        low, high = (float(v) for v in area_fraction)
    else:
        raise ValueError("area_fraction must be a number or [low, high]")
    if not math.isfinite(low) or not math.isfinite(high) or not (0 < low <= high <= 1):
        raise ValueError("area_fraction must satisfy 0 < low <= high <= 1")

    # Include only aligned endpoints.
    y_ends = list(range(0, height + 1, align_px))
    x_ends = list(range(0, width + 1, align_px))
    candidates = [(y0, y1, x0, x1) for y0, y1 in combinations(y_ends, 2)
                  for x0, x1 in combinations(x_ends, 2)]
    if support is not None and candidates:
        # Check all rectangles at once, avoiding thousands of CUDA .item()
        # synchronizations per strong view on a native 256px canvas.
        boxes = torch.tensor(candidates, device=support.device)
        y0, y1, x0, x1 = boxes.unbind(1)
        integral = torch.nn.functional.pad(support.to(torch.int64).cumsum(0).cumsum(1), (1, 0, 1, 0))
        totals = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        candidates = [tuple(box) for box in boxes[totals == (y1 - y0) * (x1 - x0)].cpu().tolist()]

    target = low + (high - low) * float(torch.rand((), generator=_generator(generator)))
    in_range = [box for box in candidates if low <= ((box[1] - box[0]) * (box[3] - box[2])) / (height * width) <= high]
    pool = in_range or candidates
    if pool:
        if in_range:
            index = int(torch.randint(len(pool), (), generator=_generator(generator)))
            return pool[index]
        # Choose the valid aligned box closest to the requested area.  Ties
        # are resolved through the private generator for deterministic replay.
        ranked = sorted(
            pool,
            key=lambda b: abs(((b[1] - b[0]) * (b[3] - b[2])) / (height * width) - target),
        )
        best_distance = abs(((ranked[0][1] - ranked[0][0]) * (ranked[0][3] - ranked[0][2])) / (height * width) - target)
        tied = [b for b in ranked if abs(((b[1] - b[0]) * (b[3] - b[2])) / (height * width) - target) == best_distance]
        return tied[int(torch.randint(len(tied), (), generator=_generator(generator)))]

    raise ValueError("CutMix valid intersection contains no aligned box")


def _replace_box(anchor: Tensor, partner: Tensor, box: CutMixBox) -> Tensor:
    if not isinstance(anchor, Tensor) or not isinstance(partner, Tensor):
        raise TypeError("CutMix values must be tensors")
    if anchor.ndim not in (4, 5) or partner.ndim != anchor.ndim:
        raise ValueError("CutMix values must both be [B,D,H,W] or [B,C,D,H,W]")
    if anchor.shape[0] != partner.shape[0]:
        if partner.shape[0] == 1:
            partner = partner.expand(anchor.shape[0], *partner.shape[1:])
        else:
            raise ValueError("CutMix anchor/partner batch sizes differ")
    if anchor.shape[2:] != partner.shape[2:] if anchor.ndim == 5 else anchor.shape[1:] != partner.shape[1:]:
        raise ValueError("CutMix anchor/partner native shapes differ")
    y0, y1, x0, x1 = _box_tuple(box, anchor.shape)
    result = anchor.clone()
    if anchor.ndim == 5:
        result[:, :, :, y0:y1, x0:x1] = partner[:, :, :, y0:y1, x0:x1]
    else:
        result[:, :, y0:y1, x0:x1] = partner[:, :, y0:y1, x0:x1]
    return result


def mix_packet(
    p_W: Tensor,
    y: Tensor,
    w: Tensor,
    valid: Tensor,
    box: CutMixBox,
    *,
    partner_p_W: Tensor | None = None,
    partner_y: Tensor | None = None,
    partner_w: Tensor | None = None,
    partner_valid: Tensor | None = None,
    partner: Mapping[str, Tensor] | None = None,
    extra: Mapping[str, Tensor] | None = None,
    partner_extra: Mapping[str, Tensor] | None = None,
) -> MixPacket:
    """Mix weak probabilities, labels, weights, and validity with one box.

    All packet tensors are detached targets.  ``partner_valid`` is copied into
    the mixed region (and therefore preserves center-crop/pad invalid cells);
    when omitted, the anchor validity is retained in that region.
    """

    if partner is not None:
        if not isinstance(partner, Mapping):
            raise TypeError("partner must be a mapping")
        partner_p_W = partner.get("p_W", partner_p_W)
        partner_y = partner.get("y", partner_y)
        partner_w = partner.get("w", partner_w)
        partner_valid = partner.get("valid", partner_valid)
        if extra is None:
            extra = {key: value for key, value in partner.items() if key not in {"p_W", "y", "w", "valid"}}
    if partner_p_W is None or partner_y is None or partner_w is None or partner_valid is None:
        raise ValueError("mix_packet requires a distinct partner packet")
    values = {"p_W": p_W, "y": y, "w": w, "valid": valid}
    for name, value in values.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
    valid4 = _mask4(valid, name="valid")
    if p_W.ndim != 5 or p_W.shape[0] != valid4.shape[0] or tuple(p_W.shape[-3:]) != tuple(valid4.shape[-3:]):
        raise ValueError("p_W must be [B,C,D,H,W] aligned with valid")
    y4, w4 = _mask4(y, name="y"), _mask4(w, name="w")
    if tuple(y4.shape) != tuple(valid4.shape) or tuple(w4.shape) != tuple(valid4.shape):
        raise ValueError("y, w, and valid must share [B,D,H,W]")
    partner_valid4 = _mask4(partner_valid, name="partner_valid")
    partner_y4, partner_w4 = _mask4(partner_y, name="partner_y"), _mask4(partner_w, name="partner_w")
    if partner_p_W.ndim != 5 or partner_p_W.shape[0] not in (1, p_W.shape[0]):
        raise ValueError("partner_p_W must be [B,C,D,H,W]")
    target_spatial = tuple(int(v) for v in p_W.shape[-3:])
    # Different-patient partners are center-crop/padded to the anchor canvas.
    # Weights are already computed before this call, so the same crop/pad is
    # applied to every packet field and padded validity/weight cells are zero.
    if tuple(partner_p_W.shape[-3:]) != target_spatial:
        partner_p_W = center_crop_pad(partner_p_W, target_spatial, pad_value=0)
    if tuple(partner_valid4.shape[-3:]) != tuple(valid4.shape[-3:]):
        partner_valid4 = center_crop_pad(partner_valid4, target_spatial, pad_value=0).bool()
    if tuple(partner_y4.shape[-3:]) != tuple(valid4.shape[-3:]):
        partner_y4 = center_crop_pad(partner_y4, target_spatial, pad_value=0).long()
    if tuple(partner_w4.shape[-3:]) != tuple(valid4.shape[-3:]):
        partner_w4 = center_crop_pad(partner_w4, target_spatial, pad_value=0).to(dtype=partner_w4.dtype)
    y0, y1, x0, x1 = _box_tuple(box, p_W.shape)
    anchor_support, _ = _valid_intersection(valid4)
    partner_support, _ = _valid_intersection(partner_valid4)
    if not bool(anchor_support[y0:y1, x0:x1].all()) or not bool(partner_support[y0:y1, x0:x1].all()):
        raise ValueError("CutMix box must lie inside the combined anchor/partner valid intersection")
    mixed = MixPacket({
        "p_W": _replace_box(p_W.detach(), partner_p_W.detach(), box),
        "y": _replace_box(y4.detach(), partner_y4.detach(), box),
        "w": _replace_box(w4.detach(), partner_w4.detach(), box),
        "valid": _replace_box(valid4.detach(), partner_valid4.detach(), box).bool(),
        "box": _box_tuple(box, p_W.shape),
    })
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise TypeError("extra must be a mapping of tensors")
        if partner_extra is None:
            if partner is not None:
                partner_extra = {key: value for key, value in partner.items() if key not in {"p_W", "y", "w", "valid"}}
            else:
                raise ValueError("partner_extra is required when extra is supplied")
        for key, anchor_extra in extra.items():
            if not isinstance(key, str) or key == "box" or not isinstance(anchor_extra, Tensor):
                raise TypeError("extra packet values must be named tensors")
            if key not in partner_extra or not isinstance(partner_extra[key], Tensor):
                raise ValueError(f"partner_extra is missing {key!r}")
            mixed[key] = _replace_box(anchor_extra.detach(), partner_extra[key].detach(), box)
    return mixed


def mix_tensor(anchor: Tensor, partner: Tensor, box: CutMixBox) -> Tensor:
    """Apply one full-depth CutMix box to an arbitrary tensor field."""

    return _replace_box(anchor, partner, box)


def center_crop_pad(value: Tensor, target_shape: Sequence[int], *, pad_value: float | int = 0) -> Tensor:
    """Center-crop or pad a native volume to ``(D,H,W)``."""

    if not isinstance(value, Tensor) or value.ndim not in (4, 5):
        raise ValueError("value must be [B,D,H,W] or [B,C,D,H,W]")
    if not isinstance(target_shape, Sequence) or len(target_shape) != 3:
        raise ValueError("target_shape must be (D,H,W)")
    td, th, tw = (int(v) for v in target_shape)
    if min(td, th, tw) <= 0:
        raise ValueError("target_shape must be positive")
    result_shape = (*value.shape[:2], td, th, tw) if value.ndim == 5 else (value.shape[0], td, th, tw)
    result = torch.full(result_shape, pad_value, dtype=value.dtype, device=value.device)
    source = value
    src_spatial = value.shape[-3:]
    src_slices, dst_slices = [], []
    for source_size, target_size in zip(src_spatial, (td, th, tw)):
        take = min(int(source_size), target_size)
        src_slices.append(slice((int(source_size) - take) // 2, (int(source_size) - take) // 2 + take))
        dst_slices.append(slice((target_size - take) // 2, (target_size - take) // 2 + take))
    if value.ndim == 5:
        result[:, :, dst_slices[0], dst_slices[1], dst_slices[2]] = source[:, :, src_slices[0], src_slices[1], src_slices[2]]
    else:
        result[:, dst_slices[0], dst_slices[1], dst_slices[2]] = source[:, src_slices[0], src_slices[1], src_slices[2]]
    return result


class StatefulViewGenerator:
    """Checkpointable weak/strong view generator built on FUSE augmentation."""

    def __init__(self, seed: int = 42, *, probabilities: Mapping[str, float] | None = None):
        self.seed = int(seed)
        self.augment = StrongInplaneAugment(seed=self.seed, probabilities=probabilities)
        self.cutmix_generator = torch.Generator(device="cpu")
        self.cutmix_generator.manual_seed(self.seed + 3001)

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "augmentation": self.augment.state_dict(),
            "cutmix_generator_state": self.cutmix_generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {"seed", "augmentation", "cutmix_generator_state"}:
            raise ValueError("invalid view-generator state")
        if int(state["seed"]) != self.seed:
            raise ValueError("view-generator seed mismatch")
        self.augment.load_state_dict(state["augmentation"])
        self.cutmix_generator.set_state(state["cutmix_generator_state"].cpu())

    @staticmethod
    def _validate_image(image: Tensor) -> None:
        if not isinstance(image, Tensor) or image.ndim != 5 or image.shape[1] != 1 or image.dtype != torch.float32:
            raise ValueError("image must be float32 [B,1,D,H,W]")
        if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
            raise ValueError("image must be finite and normalized to [0,1]")

    def weak(
        self,
        image: Tensor,
        valid: Tensor,
        geometry: Mapping[str, Any],
        target: Tensor | None = None,
    ):
        """Apply one spatial draw to image and valid mask; geometry is unchanged."""

        self._validate_image(image)
        valid4 = _mask4(valid, name="valid")
        if tuple(valid4.shape) != tuple(image.shape[:1]) + tuple(image.shape[2:]):
            raise ValueError("valid must align with image [B,D,H,W]")
        target4 = None
        if target is not None:
            target4 = _mask4(target, name="target").long()
            if tuple(target4.shape) != tuple(valid4.shape) or bool(((target4 < 0) | (target4 > 3)).any()):
                raise ValueError("target must be long labels 0..3 aligned with valid")
        if image.shape[0] > 1:
            images, masks = [], []
            targets = []
            geometry_out = geometry
            for index in range(image.shape[0]):
                current = self.weak(
                    image[index:index + 1], valid4[index:index + 1], geometry_out,
                    None if target4 is None else target4[index:index + 1],
                )
                images.append(current[0])
                if target4 is None:
                    masks.append(current[1])
                else:
                    targets.append(current[1])
                    masks.append(current[2])
                geometry_out = current[-1]
            if target4 is None:
                return torch.cat(images, dim=0), torch.cat(masks, dim=0), geometry_out
            return torch.cat(images, dim=0), torch.cat(targets, dim=0), torch.cat(masks, dim=0), geometry_out
        # StrongInplaneAugment is frame-wise ([1,1,D,H,W]).  Replaying its
        # private state gives the optional target exactly the same spatial draw
        # without changing augmentation.py or creating a second RNG contract.
        replay_state = self.augment.state_dict()
        image_w, valid_w, geometry_w = self.augment.spatial(image, valid4.long(), geometry)
        if target4 is None:
            return image_w, valid_w.bool(), geometry_w
        self.augment.load_state_dict(replay_state)
        _, target_w, geometry_w = self.augment.spatial(image, target4, geometry)
        return image_w, target_w, valid_w.bool(), geometry_w

    def strong(self, weak_image: Tensor) -> Tensor:
        self._validate_image(weak_image)
        if weak_image.shape[0] > 1:
            return torch.cat([self.strong(weak_image[index:index + 1]) for index in range(weak_image.shape[0])], dim=0)
        return self.augment.intensity(weak_image).clamp(0, 1)

    def joint_views(self, image: Tensor, valid: Tensor, geometry: Mapping[str, Any], target: Tensor | None = None):
        if target is None:
            weak_image, weak_valid, geometry_w = self.weak(image, valid, geometry)
            return weak_image, self.strong(weak_image), weak_valid, geometry_w
        weak_image, weak_target, weak_valid, geometry_w = self.weak(image, valid, geometry, target)
        return weak_image, self.strong(weak_image), weak_target, weak_valid, geometry_w

    def weak_with_target(
        self, image: Tensor, target: Tensor, valid: Tensor, geometry: Mapping[str, Any],
    ):
        """Explicit labeled-view spelling (target before valid)."""

        return self.weak(image, valid, geometry, target)

    def joint_views_with_target(
        self, image: Tensor, target: Tensor, valid: Tensor, geometry: Mapping[str, Any],
    ):
        return self.joint_views(image, valid, geometry, target)

    def sample_box(self, valid: Tensor, **kwargs: Any) -> CutMixBox:
        return cutmix_box(valid, self.cutmix_generator, **kwargs)


__all__ = [
    "CutMixBox",
    "MixPacket",
    "StatefulViewGenerator",
    "ViewGenerator",
    "center_crop_pad",
    "cutmix_box",
    "mix_tensor",
    "mix_packet",
]
