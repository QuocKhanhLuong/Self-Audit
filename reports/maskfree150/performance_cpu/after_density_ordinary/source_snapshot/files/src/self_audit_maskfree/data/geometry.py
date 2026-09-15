"""Native geometry, masked resampling and the exact inverse-transform record.

Two responsibilities:

1. Read native geometry from whatever container a root actually holds, and say
   honestly whether a native-grid export is possible. A ``.npy`` volume carries
   no affine, so anything derived from it is a *stored-grid* result; claiming a
   native-grid NIfTI export from it would be a fabricated geometry claim.
2. Resample while never mixing two observation roles. Plain bilinear resizing
   of a role-masked image averages across the role boundary. Here the image and
   its role mask are resampled as a normalized convolution -- ``sum(values in
   role) / sum(mask in role)`` -- so a resized pixel is a weighted mean of
   same-role native pixels only, and is marked unsupported when less than half
   its native footprint carried that role.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .firewall import assert_image_only

#: A resampled pixel keeps a role only when at least this fraction of its
#: native footprint carried that role. Prevents one-pixel role bleed.
SUPPORT_FLOOR = 0.5

# NIfTI-1 permits a small set of physical units.  We retain the declared unit
# and expose a converted millimetre spacing only when the header declares a
# recognised spatial unit.  In particular, ``unknown`` is never silently
# treated as millimetres.
_SPATIAL_UNIT_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    # nibabel exposes the NIfTI-1 unit-code labels as ``meter`` and
    # ``micron`` (rather than the abbreviated ``m``/``um`` spellings).
    "meter": 1000.0,
    "um": 0.001,
    "micron": 0.001,
    "microns": 0.001,
    "micrometer": 0.001,
    "micrometers": 0.001,
}


class GeometryError(ValueError):
    """Raised for unusable or self-inconsistent native geometry."""


def _nifti_geometry(path: Path) -> dict[str, Any]:
    import nibabel as nib

    img = nib.load(str(path))
    header = img.header
    affine = np.asarray(img.affine, dtype=np.float64)
    zooms = [float(z) for z in header.get_zooms()]
    affine_valid = bool(
        affine.shape == (4, 4)
        and np.all(np.isfinite(affine))
        and float(abs(np.linalg.det(affine[:3, :3]))) > 0.0
    )
    try:
        orientation = [str(code) for code in nib.aff2axcodes(affine)]
    except Exception:  # pragma: no cover - degenerate affine
        orientation = None
    units = None
    try:
        raw_units = header.get_xyzt_units()
        units = [None if value is None else str(value).lower() for value in raw_units]
    except Exception:  # pragma: no cover - exotic header
        units = None
    spatial_unit = str(units[0]).lower() if units and units[0] else "unknown"
    unit_factor = _SPATIAL_UNIT_TO_MM.get(spatial_unit)
    positive_zooms = len(zooms) >= 3 and all(
        np.isfinite(z) and z > 0.0 for z in zooms[:3]
    )
    spacing_valid = bool(affine_valid and positive_zooms and unit_factor is not None)
    spacing_reason = None
    if not affine_valid:
        spacing_reason = "nifti affine is absent, non-finite or singular"
    elif not positive_zooms:
        spacing_reason = "nifti spatial zooms are missing or non-positive"
    elif unit_factor is None:
        spacing_reason = (
            "nifti header does not declare a recognised spatial unit; "
            "millimetre spacing is unavailable"
        )
    native_geometry_reason = None if affine_valid else "nifti affine is absent, non-finite or singular"
    return {
        "source_format": "nifti",
        "shape": [int(s) for s in img.shape],
        "dtype": str(np.dtype(header.get_data_dtype())),
        "native_affine": affine.tolist(),
        "orientation": orientation,
        "zooms": zooms,
        "xyzt_units": units,
        "spatial_unit": spatial_unit,
        "spacing_mm": [float(z * unit_factor) for z in zooms[:3]]
        if unit_factor is not None and positive_zooms
        else None,
        "spacing_valid": spacing_valid,
        # Geometry and physical spacing are separate claims.  A valid affine
        # still permits a native-grid export even when its physical units are
        # unknown; only millimetre metrics are suppressed in that case.
        "spacing_reason": spacing_reason,
        "native_geometry": "available" if affine_valid else "unavailable",
        "native_geometry_reason": native_geometry_reason,
        "export_grid": "native" if affine_valid else "stored",
        "native_grid_export": affine_valid,
    }


def _npy_geometry(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r")
    return {
        "source_format": "npy",
        "shape": [int(s) for s in array.shape],
        "dtype": str(array.dtype),
        "native_affine": None,
        "orientation": None,
        "zooms": None,
        "xyzt_units": None,
        "spatial_unit": "unknown",
        "spacing_mm": None,
        "spacing_valid": False,
        "spacing_reason": "npy container carries no affine or physical unit header",
        "native_geometry": "unavailable",
        "native_geometry_reason": "npy container carries no affine or header",
        "export_grid": "stored",
        "native_grid_export": False,
    }


def read_geometry(path: str | Path) -> dict[str, Any]:
    """Header-only geometry probe. Never loads full voxel data for NIfTI."""
    p = assert_image_only(path)
    name = p.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return _nifti_geometry(p)
    if name.endswith(".npy"):
        return _npy_geometry(p)
    raise GeometryError(f"unsupported image container: {p.name}")


def _open_source_selector(
    path: str | Path,
    *,
    depth_axis: int,
    frame_index: int | None,
    frame_axis: int | None = None,
) -> tuple[Any, list[slice], int | None]:
    """Open one source and validate its frame/depth selectors exactly once."""
    p = assert_image_only(path)
    name = p.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        data = nib.load(str(p)).dataobj
    else:
        data = np.load(p, mmap_mode="r")

    ndim = len(data.shape)
    if depth_axis < 0 or depth_axis >= min(ndim, 3):
        raise GeometryError(f"depth_axis {depth_axis} invalid for rank-{ndim} source")
    actual_frame_axis: int | None
    if ndim == 4:
        actual_frame_axis = 3 if frame_axis is None else int(frame_axis)
        if actual_frame_axis < 0 or actual_frame_axis >= ndim or actual_frame_axis == depth_axis:
            raise GeometryError(
                f"frame_axis {actual_frame_axis} invalid for depth_axis {depth_axis}"
            )
        if frame_index is None:
            raise GeometryError("4-D acquisition requires an explicit frame index")
        if not 0 <= int(frame_index) < int(data.shape[actual_frame_axis]):
            raise GeometryError(
                f"frame_index {frame_index} outside frame extent "
                f"{data.shape[actual_frame_axis]}"
            )
        selector = [slice(None)] * 4
        selector[actual_frame_axis] = int(frame_index)
    elif ndim == 3:
        # Legacy rank-3 callers may pass ``frame_index=None`` or an ignored
        # frame-axis hint.  Preserve the original reader's no-op semantics.
        actual_frame_axis = None
        selector = [slice(None)] * 3
    else:
        raise GeometryError(f"expected a 3-D or 4-D volume, got shape {tuple(data.shape)}")

    return data, selector, actual_frame_axis


def read_source_frame(
    path: str | Path,
    *,
    depth_axis: int,
    frame_index: int | None,
    frame_axis: int | None = None,
    _return_frame_axis: bool = False,
) -> np.ndarray | tuple[np.ndarray, int | None]:
    """Return one native rank-3 frame using the canonical source decoder.

    The cache layer uses this helper so there is exactly one implementation of
    the NIfTI/NPY selector, dtype conversion, and finite-value handling.  It is
    deliberately image-only and returns source axis order with any frame axis
    removed; neighbouring-slice extraction remains in :func:`read_slice_stack`.
    """
    data, selector, actual_frame_axis = _open_source_selector(
        path,
        depth_axis=depth_axis,
        frame_index=frame_index,
        frame_axis=frame_axis,
    )

    raw = np.asarray(data[tuple(selector)])
    if not np.issubdtype(raw.dtype, np.number):
        raise GeometryError(f"image frame has non-numeric dtype {raw.dtype}")
    frame = np.asarray(raw, dtype=np.float32)
    if not np.all(np.isfinite(frame)):
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    frame = np.ascontiguousarray(frame)
    return (frame, actual_frame_axis) if _return_frame_axis else frame


def read_slice_stack(
    path: str | Path,
    *,
    depth_axis: int,
    slice_index: int,
    frame_index: int | None,
    frame_axis: int | None = None,
) -> np.ndarray:
    """Return the ``[3, H, W]`` neighbour stack ``(z-1, z, z+1)`` as float32.

    Border slices replicate the centre slice, so the stack always has three
    channels and the centre is always channel 1.
    """
    data, selector, actual_frame_axis = _open_source_selector(
        path,
        depth_axis=depth_axis,
        frame_index=frame_index,
        frame_axis=frame_axis,
    )
    # ``selector`` indexes the source proxy before any frame axis is removed,
    # so the declared depth axis remains the source axis for all three reads.
    # The selected planes themselves are rank-2; no full frame is materialised.
    depth = int(data.shape[depth_axis])
    if depth <= 0 or not 0 <= int(slice_index) < depth:
        raise GeometryError(f"slice_index {slice_index} outside depth extent {depth}")
    planes = []
    for offset in (-1, 0, 1):
        z = int(np.clip(slice_index + offset, 0, depth - 1))
        local = list(selector)
        local[depth_axis] = z
        planes.append(np.asarray(data[tuple(local)], dtype=np.float32))
    stack = np.stack(planes, axis=0)
    if stack.ndim != 3:
        raise GeometryError(f"slice extraction produced shape {stack.shape}, expected [3,H,W]")
    if not np.all(np.isfinite(stack)):
        stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    return stack


def _pool(values: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Area resampling for any direction, via adaptive average pooling."""
    return F.adaptive_avg_pool2d(values.unsqueeze(0), out_hw).squeeze(0)


def masked_resize(
    values: torch.Tensor,
    support: torch.Tensor,
    out_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample ``values`` [C,H,W] using only pixels inside ``support`` [H,W].

    Returns ``(resized_values, resized_support)``. Values outside the resized
    support are exactly zero, and no output pixel mixes a supported native
    pixel with an unsupported one.
    """
    if values.ndim != 3:
        raise GeometryError("values must be [C,H,W]")
    if support.shape != values.shape[1:]:
        raise GeometryError("support must match the spatial shape of values")
    mask = support.to(values.dtype).unsqueeze(0)
    numerator = _pool(values * mask, out_hw)
    denominator = _pool(mask, out_hw)
    out_support = denominator[0] >= SUPPORT_FLOOR
    resized = numerator / denominator.clamp_min(1e-6)
    resized = torch.where(out_support.unsqueeze(0), resized, torch.zeros_like(resized))
    return resized, out_support


def nearest_resize_mask(mask: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbour resize of a boolean role mask (never interpolated)."""
    resized = F.interpolate(
        mask.to(torch.float32).unsqueeze(0).unsqueeze(0),
        size=out_hw,
        mode="nearest-exact",
    )
    return resized[0, 0] > 0.5


def inverse_transform_record(
    *,
    native_shape: list[int],
    depth_axis: int,
    slice_index: int,
    frame_index: int | None,
    native_hw: tuple[int, int],
    unit_hw: tuple[int, int],
    native_grid_export: bool,
    frame_axis: int | None = None,
    full_native_shape: list[int] | None = None,
) -> dict[str, Any]:
    """Exactly how to put a unit-grid prediction back on the source grid.

    Consumed by the exporter (W7). It is a description, not an operation: no
    resampling happens here, and ``native_grid_export=False`` means the only
    honest destination is the stored grid.
    """
    if len(native_shape) != 3:
        raise GeometryError(
            "inverse_transform_record expects the three spatial native axes; "
            "pass full_native_shape separately for a 4-D source"
        )
    axis_order = {0: "ZHW", 1: "HZW", 2: "HWZ"}.get(int(depth_axis))
    if axis_order is None:
        raise GeometryError(f"depth_axis {depth_axis} outside the three spatial axes")
    return {
        "version": 1,
        "native_shape": [int(s) for s in native_shape],
        "native_shape_full": (
            [int(s) for s in full_native_shape] if full_native_shape is not None else None
        ),
        "native_axis_order": axis_order,
        "depth_axis": int(depth_axis),
        "frame_axis": None if frame_axis is None else int(frame_axis),
        "slice_index": int(slice_index),
        "frame_index": None if frame_index is None else int(frame_index),
        "crop": None,
        "crop_rule": "whole_field_of_view_no_image_informed_crop",
        "forward_resize": {
            "input_hw": [int(native_hw[0]), int(native_hw[1])],
            "output_hw": [int(unit_hw[0]), int(unit_hw[1])],
            "mode_values": "masked_area_normalized_convolution",
            "mode_masks": "nearest_exact",
            "support_floor": SUPPORT_FLOOR,
        },
        "inverse_resize": {
            "input_hw": [int(unit_hw[0]), int(unit_hw[1])],
            "output_hw": [int(native_hw[0]), int(native_hw[1])],
            "mode_labels": "nearest_exact",
            "mode_probabilities": "bilinear_then_renormalize",
            "mode_validity": "bilinear_support_weighted",
        },
        "native_grid_export": bool(native_grid_export),
        "export_grid": "native" if native_grid_export else "stored",
    }
