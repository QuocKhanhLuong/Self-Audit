"""Image-only normalization and positive transport on declared pixel-edge grids."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from layer_upgrade.layers import block_lift
from layer_upgrade.losses import evidence_distribution, masked_feature_loss, soft_ce


def require_digest(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("a reviewed SHA256 is required")
    return value


def normalize_cine(cine: Tensor, image_sha256: str) -> tuple[Tensor, dict]:
    """One [0,1] transform shared by all T/Z slices of this individual image."""
    require_digest(image_sha256)
    if cine.ndim != 4 or min(cine.shape) < 1 or not torch.isfinite(cine).all():
        raise ValueError("finite nonempty TZYX cine required")
    x = cine.detach().float()
    lo, hi = float(x.min()), float(x.max())
    normalized = (x - lo) / (hi - lo) if hi > lo else torch.zeros_like(x)
    return normalized, {
        "schema": "layer_guided.per_cine_minmax.v1",
        "image_sha256": image_sha256,
        "shape_tzyx": list(x.shape),
        "minimum": lo,
        "maximum": hi,
        "constant_image": hi == lo,
        "scope": "one_image_all_frames_and_spatial_slices_no_cohort_fitting",
        "input_domain": "native_intensity_or_historical_zscore",
    }


def slice_contexts(cine: Tensor, indices: list[tuple[int, int]]) -> Tensor:
    if cine.ndim != 4 or not indices or not torch.isfinite(cine).all():
        raise ValueError("finite TZYX cine and nonempty indices required")
    if cine.min() < 0 or cine.max() > 1:
        raise ValueError("normalize the whole cine before selecting slices")
    for t, z in indices:
        if (
            type(t) is not int
            or type(z) is not int
            or not (0 <= t < cine.shape[0] and 0 <= z < cine.shape[1])
        ):
            raise ValueError("invalid temporal/spatial slice index")
    return torch.stack(
        [
            torch.stack(
                [
                    cine[t, max(0, z - 1)],
                    cine[t, z],
                    cine[t, min(cine.shape[1] - 1, z + 1)],
                ]
            )
            for t, z in indices
        ]
    )


@dataclass(frozen=True)
class TeacherRegistration:
    """Source-side geometry bound to a frozen cache and its declared producer."""

    native_hw: tuple[int, int]
    teacher_hw: tuple[int, int]
    image_sha256: str
    affine_xyzt: tuple[tuple[float, ...], ...]
    source_manifest_sha256: str
    producer_sha256: tuple[tuple[str, str], ...]
    source_axis_order: str = "TZYXC"
    mapping: str = "full_native_fov_pixel_edges"

    def __post_init__(self):
        require_digest(self.image_sha256)
        require_digest(self.source_manifest_sha256)
        if not self.producer_sha256:
            raise ValueError("registered source producer hashes required")
        for _, digest in self.producer_sha256:
            require_digest(digest)
        if (
            self.source_axis_order != "TZYXC"
            or self.mapping != "full_native_fov_pixel_edges"
        ):
            raise ValueError("unsupported source axes or coordinate registration")

    @property
    def sha256(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


@dataclass(frozen=True)
class GridGeometry:
    native_hw: tuple[int, int]
    teacher_hw: tuple[int, int]
    image_sha256: str
    affine_xyzt: tuple[tuple[float, ...], ...]
    spatial_unit: str
    registration: TeacherRegistration
    teacher_mapping: str = "full_native_fov_pixel_edges"

    def __post_init__(self):
        require_digest(self.image_sha256)
        if any(
            len(v) != 2 or any(type(n) is not int or n < 1 for n in v)
            for v in (self.native_hw, self.teacher_hw)
        ):
            raise ValueError("positive native and teacher grid dimensions required")
        a = torch.tensor(self.affine_xyzt, dtype=torch.double)
        if (
            a.shape != (4, 4)
            or not torch.isfinite(a).all()
            or abs(float(torch.linalg.det(a[:3, :3]))) < 1e-12
        ):
            raise ValueError("finite nonsingular native affine required")
        if self.spatial_unit not in {"unknown", "meter", "mm", "micron"}:
            raise ValueError("record native units; do not invent spacing")
        if self.teacher_mapping != "full_native_fov_pixel_edges":
            raise ValueError("unregistered teacher coordinate transform")
        r = self.registration
        if not isinstance(r, TeacherRegistration) or (
            r.native_hw != self.native_hw
            or r.teacher_hw != self.teacher_hw
            or r.image_sha256 != self.image_sha256
            or r.affine_xyzt != self.affine_xyzt
            or r.mapping != self.teacher_mapping
        ):
            raise ValueError(
                "source/target image geometry registration mismatch; no implicit flip or resampling"
            )

    @property
    def padded_hw(self):
        return tuple(math.ceil(n / 32) * 32 for n in self.native_hw)

    def receipt(self):
        return {
            **asdict(self),
            "registration_sha256": self.registration.sha256,
            "padded_hw": self.padded_hw,
            "padding": "bottom_right_replicate_image_zero_evidence",
            "coarse_stride": 4,
            "feature_stride": 16,
            "transport": "positive_pixel_cell_overlap_density_average",
        }


def _overlap(native: int, source: int, padded: int, stride: int, like: Tensor):
    src = torch.arange(source + 1, device=like.device, dtype=like.dtype) * (
        native / source
    )
    dst = torch.arange(padded // stride, device=like.device, dtype=like.dtype) * stride
    return (
        torch.minimum(dst[:, None] + stride, src[None, 1:])
        - torch.maximum(dst[:, None], src[None, :-1])
    ).clamp_min(0) / stride


def transport_density(value: Tensor, geometry: GridGeometry, stride: int = 4) -> Tensor:
    """Same positive linear map for mass and support; padding contributes zero."""
    if (
        stride not in (4, 16)
        or value.ndim != 4
        or tuple(value.shape[-2:]) != geometry.teacher_hw
    ):
        raise ValueError("declared source NCHW grid and stride 4/16 required")
    if (
        value.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(value).all()
    ):
        raise ValueError(
            "finite floating point densities required, never integer labels"
        )
    value = value.detach()
    y, x = (
        _overlap(n, s, p, stride, value)
        for n, s, p in zip(geometry.native_hw, geometry.teacher_hw, geometry.padded_hw)
    )
    return torch.einsum("yh,bchw,xw->bcyx", y, value, x)


def coarse_evidence(
    mass: Tensor, support: Tensor, geometry: GridGeometry, *, source: dict
):
    """Accept explicit source mass/support; never reconstruct conditional Z from Q."""
    if source.get("kind") not in {"fixed_anatomical_mass", "conditional_z_mass"}:
        raise ValueError("named evidence needs an explicit source mass contract")
    if (
        require_digest(source.get("manifest_sha256"))
        != geometry.registration.source_manifest_sha256
    ):
        raise ValueError("semantic source manifest differs from its registration")
    if source.get("registration_sha256") != geometry.registration.sha256:
        raise ValueError("semantic source registration mismatch")
    if source.get("orientation_verified") is not True:
        raise ValueError("named RV/LV guidance requires verified orientation")
    if source.get("ontology") != ["BG", "RV", "MYO", "LV"] or mass.shape[1] != 4:
        raise ValueError("explicit named class ontology required")
    evidence_distribution(mass, support)  # Validate consistency BEFORE transport.
    transported_mass = transport_density(mass, geometry)
    transported_support = transport_density(support[:, None], geometry)[:, 0]
    target, valid = evidence_distribution(transported_mass, transported_support)
    return (
        target,
        transported_support * valid,
        {"source": dict(source), "geometry": geometry.receipt()},
    )


def coarse_semantic_loss(outputs: dict, target: Tensor, weights: Tensor):
    """Only the declared coarse tensor is supervised: no native-logit resizing."""
    return soft_ce(outputs["coarse_logits"], target, weights)


def masked_feature_objective(
    model, images: Tensor, teacher: Tensor, geometry: GridGeometry, mask: Tensor
):
    """Run the student on a truly masked image; no teacher tensor enters forward."""
    if (
        images.ndim != 4
        or images.shape[1] != 3
        or tuple(images.shape[-2:]) != geometry.native_hw
    ):
        raise ValueError("aligned native NCHW image contexts required")
    if teacher.ndim != 4 or teacher.shape[0] != images.shape[0]:
        raise ValueError("aligned teacher batch required")
    aligned = transport_density(teacher, geometry, 16)
    coverage = transport_density(torch.ones_like(teacher[:, :1]), geometry, 16)[:, 0]
    # Partial border cells are excluded from the feature mask, not labeled BG.
    valid = coverage >= 1 - 1e-6
    if mask.dtype != torch.bool or mask.shape != valid.shape or (mask & ~valid).any():
        raise ValueError("mask must select only fully native feature cells")
    aligned = aligned / coverage[:, None].clamp_min(1e-8)
    native_mask = block_lift(mask[:, None], 16)[
        ..., : images.shape[-2], : images.shape[-1]
    ]
    masked_view = images.masked_fill(native_mask, 0)
    outputs = model(masked_view)
    loss = masked_feature_loss(outputs["features"], aligned, mask)
    return loss, {
        "masked_pixels": int(native_mask.sum()),
        "masked_cells": int(mask.sum()),
        "teacher_in_forward": False,
        "geometry": geometry.receipt(),
    }
