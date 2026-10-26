"""Physical geometry and reversible-transform helpers for native volumes.

The v2 training manifest deliberately contains only image/mask payload paths.
This module keeps physical metadata in a separately hash-pinned sidecar.  The
sidecar is intentionally small and JSON-only so preparation can be inspected
without opening any U/VAL/TEST labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "self_audit.volume.geometry.v1"
AXIS_ORDER = "T,Z,Y,X"
UNKNOWN_LABEL = 255


def forbidden_mask_reference(value: Any, *, _path: str = "") -> str | None:
    """Return the first nested key denoting a mask reference.

    U geometry is image-only.  Checking just the top-level ``mask_sources``
    field is insufficient because a forbidden path can be nested below
    ``source_image`` or another provenance object.  The structural check does
    not inspect values or open files and intentionally errs on the safe side:
    any key containing ``mask`` is a forbidden reference for U.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{_path}.{key_text}" if _path else key_text
            if "mask" in key_text.casefold():
                return path
            found = forbidden_mask_reference(item, _path=path)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = forbidden_mask_reference(item, _path=f"{_path}[{index}]")
            if found is not None:
                return found
    return None


def sha256_file(path: str | Path) -> str:
    """Return a stable SHA-256 digest for a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _finite_affine(affine: Any) -> np.ndarray:
    a = np.asarray(affine, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError("affine_xyz_to_world_mm must be a finite 4x4 matrix")
    if not np.allclose(a[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("affine last row must be [0,0,0,1]")
    if abs(float(np.linalg.det(a[:3, :3]))) < 1e-12:
        raise ValueError("affine voxel matrix must be invertible")
    return a


def spacing_from_affine(affine: Any) -> np.ndarray:
    """Return physical spacing in the affine's XYZ column order."""
    a = _finite_affine(affine)
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("affine spacing must be finite and positive")
    return spacing


def _record_shape(record: Mapping[str, Any]) -> tuple[int, int, int, int]:
    # ``shape_tzyx`` is retained as a read-only compatibility alias used by
    # existing native manifests; new sidecars emit both names.
    raw = record.get("native_shape_tzyx", record.get("shape_tzyx", ()))
    shape = tuple(int(x) for x in raw)
    if len(shape) != 4 or any(x < 1 for x in shape):
        raise ValueError("native_shape_tzyx must contain four positive dimensions")
    return shape  # type: ignore[return-value]


def validate_geometry_record(record: Mapping[str, Any], *, expected_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Validate and normalize one patient geometry record.

    Missing physical fields are rejected.  In particular, shape alone is never
    used to invent spacing or an affine.
    """
    if not isinstance(record, Mapping):
        raise ValueError("geometry record must be an object")
    shape = _record_shape(record)
    if expected_shape is not None and tuple(expected_shape) != shape:
        raise ValueError(f"geometry shape {shape} does not match image shape {tuple(expected_shape)}")
    if record.get("axis_order") not in (AXIS_ORDER, None):
        raise ValueError("only T,Z,Y,X image axis order is supported")
    affine = _finite_affine(record.get("affine_xyz_to_world_mm"))
    spacing = spacing_from_affine(affine)
    declared = record.get("spacing_xyz_mm")
    if declared is not None and not np.allclose(np.asarray(declared, dtype=float), spacing, atol=1e-6, rtol=1e-6):
        raise ValueError("declared spacing does not agree with affine")
    units = record.get("units")
    if units != "mm":
        raise ValueError("physical geometry units must be mm")
    # These fields are intentionally mandatory.  A shape and an affine are not
    # enough to identify a physical cine grid: orientation, slice ordering and
    # the explicit frame map are needed to prove that an image/mask bridge is
    # not being guessed from filenames or array dimensions.
    orientation = record.get("orientation")
    if not isinstance(orientation, str) or not orientation.strip():
        raise ValueError("orientation is required for physical geometry")
    slice_order = record.get("slice_order")
    if not isinstance(slice_order, str) or not slice_order.strip():
        raise ValueError("slice_order is required for physical geometry")
    frame_map = record.get("temporal_frame_mapping")
    if not isinstance(frame_map, list) or len(frame_map) != shape[0]:
        raise ValueError("temporal_frame_mapping must have one entry per frame")
    if any(type(x) is not int or x < 0 for x in frame_map) or len(set(frame_map)) != len(frame_map):
        raise ValueError("temporal_frame_mapping must contain unique non-negative frame IDs")
    out = dict(record)
    out["axis_order"] = AXIS_ORDER
    out["native_shape_tzyx"] = list(shape)
    out["shape_tzyx"] = list(shape)
    out["native_shape_zyx"] = list(shape[1:])
    out["affine_xyz_to_world_mm"] = affine.tolist()
    out["spacing_xyz_mm"] = spacing.tolist()
    out["native_spacing_xyz_mm"] = spacing.tolist()
    out["units"] = "mm"
    out["orientation"] = orientation.strip()
    out["slice_order"] = slice_order.strip()
    out["temporal_frame_mapping"] = list(frame_map)
    return out


def validate_sidecar(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    if sidecar.get("schema") != SCHEMA:
        raise ValueError(f"geometry sidecar schema must be {SCHEMA}")
    if sidecar.get("axis_order") != AXIS_ORDER:
        raise ValueError("geometry sidecar axis_order must be T,Z,Y,X")
    if sidecar.get("units") != "mm":
        raise ValueError("geometry sidecar units must be mm")
    if type(sidecar.get("synthetic")) is not bool:
        raise ValueError("geometry sidecar requires an explicit synthetic identity")
    manifest_sha = sidecar.get("source_manifest_sha256")
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise ValueError("geometry sidecar requires a hash-pinned source manifest")
    patients = sidecar.get("patients")
    if not isinstance(patients, Mapping) or not patients:
        raise ValueError("geometry sidecar requires non-empty patients object")
    out = dict(sidecar)
    out["patients"] = {
        str(pid): validate_geometry_record(rec)
        for pid, rec in patients.items()
    }
    return out


def load_sidecar(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    p = Path(path)
    if expected_sha256 is not None and sha256_file(p) != expected_sha256:
        raise ValueError("geometry sidecar hash mismatch")
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read geometry sidecar: {p}") from exc
    return validate_sidecar(data)


def voxel_to_world(affine: Any, voxel_xyz: Any) -> np.ndarray:
    a = _finite_affine(affine)
    v = np.asarray(voxel_xyz, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError("voxel coordinates must end in an XYZ dimension")
    flat = v.reshape(-1, 3)
    out = np.concatenate([flat, np.ones((len(flat), 1))], axis=1) @ a.T
    return out[:, :3].reshape(v.shape)


def world_to_voxel(affine: Any, world_xyz_mm: Any) -> np.ndarray:
    a = _finite_affine(affine)
    w = np.asarray(world_xyz_mm, dtype=float)
    if w.shape[-1] != 3:
        raise ValueError("world coordinates must end in an XYZ dimension")
    flat = w.reshape(-1, 3)
    out = np.concatenate([flat, np.ones((len(flat), 1))], axis=1) @ np.linalg.inv(a).T
    return out[:, :3].reshape(w.shape)


def _source_path(base_dir: str | Path, source: Mapping[str, Any]) -> Path:
    raw = source.get("path", source.get("source_path"))
    if not isinstance(raw, str) or not raw:
        raise ValueError("physical source metadata requires a path")
    p = Path(raw)
    return p if p.is_absolute() else Path(base_dir) / p


def validate_nifti_correspondence(
    image_path: str | Path,
    *,
    image_sha256: str,
    expected_shape_tzyx: tuple[int, int, int, int],
    expected_affine_xyz_to_world_mm: Any,
    expected_orientation: str,
    expected_frame_mapping: list[int],
    masks: Mapping[str, Any] | None = None,
    base_dir: str | Path | None = None,
    prepared_image_tzyx: Any | None = None,
    prepared_masks_tzyx: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate optional raw NIfTI provenance for a prepared TRAIN record.

    Prepared arrays remain the training payload; this helper is an explicit
    physical-space receipt for real data.  It checks source hash, NIfTI units,
    orientation, affine/shape and frame IDs.  When frame/mask sidecars are
    supplied, each mask is checked against both its same-frame NIfTI image and
    the corresponding cine frame voxel-for-voxel.  No source discovery is
    attempted: missing metadata or files fails closed.
    """
    try:
        import nibabel as nib
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ValueError("nibabel is required to validate physical NIfTI metadata") from exc
    source = {"path": str(image_path), "sha256": image_sha256}
    root = Path(base_dir) if base_dir is not None else Path(image_path).resolve().parent
    image_p = _source_path(root, source)
    if not image_p.is_file() or sha256_file(image_p) != image_sha256:
        raise ValueError("raw cine source hash/path mismatch")
    image_obj = nib.load(str(image_p))
    if tuple(int(x) for x in image_obj.shape) != tuple(reversed(expected_shape_tzyx)):
        raise ValueError("raw cine NIfTI shape does not match prepared TZYX geometry")
    if not np.allclose(np.asarray(image_obj.affine, dtype=float), np.asarray(expected_affine_xyz_to_world_mm, dtype=float), atol=1e-5, rtol=1e-6):
        raise ValueError("raw cine affine does not match geometry sidecar")
    units = image_obj.header.get_xyzt_units()[0]
    if units not in ("mm", "unknown"):
        raise ValueError("raw cine spatial units must be millimetres")
    if units != "mm":
        raise ValueError("raw cine NIfTI is missing explicit millimetre units")
    axcodes = "".join(str(x) for x in nib.aff2axcodes(image_obj.affine))
    if expected_orientation and expected_orientation not in {axcodes, "affine_direction_cosines"}:
        raise ValueError("raw cine orientation does not match geometry sidecar")
    if len(expected_frame_mapping) != expected_shape_tzyx[0] or len(set(expected_frame_mapping)) != len(expected_frame_mapping):
        raise ValueError("temporal frame mapping does not cover the cine")
    if prepared_image_tzyx is not None:
        prepared = np.asarray(prepared_image_tzyx)
        if tuple(int(x) for x in prepared.shape) != tuple(int(x) for x in expected_shape_tzyx):
            raise ValueError("prepared image shape does not match expected TZYX geometry")
        if prepared.dtype.kind not in "fiu" or not np.isfinite(prepared).all():
            raise ValueError("prepared image must be finite numeric TZYX data")
        # ``prepare_acdc.py`` is the canonical image-only bridge: raw NIfTI is
        # XYZT, each cine is percentile-normalized and clipped to [0,1], then
        # transposed to TZYX.  Recomputing this transform binds the prepared
        # payload to the hash-pinned raw source, rather than trusting shape,
        # affine and units alone.
        raw = np.asarray(image_obj.dataobj)
        if raw.ndim != 4 or not np.isfinite(raw).all():
            raise ValueError("raw cine must be finite XYZT data")
        lo, hi = np.percentile(raw, [0.5, 99.5])
        normalized = np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0, 1)
        expected_prepared = normalized.transpose(3, 2, 1, 0).astype(np.float32)
        if not np.allclose(prepared.astype(np.float32, copy=False), expected_prepared,
                           atol=2e-5, rtol=2e-5):
            raise ValueError("prepared image/raw cine voxel correspondence failed")
    # Validate the optional per-frame references.  The prepared NPY mask hash
    # is checked by VolumeTrainingData; this receipt validates the original
    # physical source and exact frame bridge.
    for raw_t, item in (masks or {}).items():
        if not isinstance(item, Mapping):
            raise ValueError("mask physical metadata must be objects")
        frame_id = int(raw_t)
        if 0 <= frame_id < expected_shape_tzyx[0]:
            frame_t = frame_id
        elif frame_id in expected_frame_mapping:
            frame_t = expected_frame_mapping.index(frame_id)
        else:
            raise ValueError("mask frame ID is absent from temporal mapping")
        mask_raw = item.get("source", item)
        if not isinstance(mask_raw, Mapping):
            raise ValueError("mask source metadata must be an object")
        mask_path = _source_path(root, mask_raw)
        mask_sha = mask_raw.get("sha256", mask_raw.get("source_sha256"))
        if not isinstance(mask_sha, str) or sha256_file(mask_path) != mask_sha:
            raise ValueError("mask source hash/path mismatch")
        mask_obj = nib.load(str(mask_path))
        if tuple(int(x) for x in mask_obj.shape) != tuple(reversed(expected_shape_tzyx[1:])):
            raise ValueError("mask NIfTI shape does not match cine frame")
        if not np.allclose(np.asarray(mask_obj.affine, dtype=float), np.asarray(image_obj.affine, dtype=float), atol=1e-5, rtol=1e-6):
            raise ValueError("mask affine does not match cine affine")
        if mask_obj.header.get_xyzt_units()[0] != "mm":
            raise ValueError("mask NIfTI is missing explicit millimetre units")
        if "orientation" in mask_raw and str(mask_raw["orientation"]) != axcodes:
            raise ValueError("mask orientation does not match cine orientation")
        mask_values = np.asarray(mask_obj.dataobj)
        if not np.isin(mask_values, [0, 1, 2, 3, UNKNOWN_LABEL]).all():
            raise ValueError("mask NIfTI contains an unknown class outside 255")
        if prepared_masks_tzyx is not None:
            prepared_mask = prepared_masks_tzyx.get(str(raw_t))
            if prepared_mask is None:
                prepared_mask = prepared_masks_tzyx.get(str(frame_id))
            if prepared_mask is None:
                raise ValueError("prepared mask coverage does not match raw mask_sources")
            expected_mask = mask_values.transpose(2, 1, 0)
            prepared_mask_array = np.asarray(prepared_mask)
            if prepared_mask_array.shape != expected_mask.shape or not np.array_equal(
                    prepared_mask_array, expected_mask):
                raise ValueError("prepared mask/raw mask voxel correspondence failed")
        frame_path_raw = item.get("frame_path", item.get("cine_frame_path"))
        if frame_path_raw is not None:
            frame_source = {"path": frame_path_raw, "sha256": item.get("frame_sha256")}
            if not isinstance(frame_source["sha256"], str):
                raise ValueError("cine frame source hash is required")
            frame_path = _source_path(root, frame_source)
            if sha256_file(frame_path) != frame_source["sha256"]:
                raise ValueError("cine frame source hash/path mismatch")
            frame_obj = nib.load(str(frame_path))
            if tuple(int(x) for x in frame_obj.shape) != tuple(reversed(expected_shape_tzyx[1:])):
                raise ValueError("cine frame NIfTI shape mismatch")
            if not np.allclose(np.asarray(frame_obj.affine), np.asarray(image_obj.affine), atol=1e-5, rtol=1e-6):
                raise ValueError("cine frame affine mismatch")
            if not np.array_equal(np.asarray(frame_obj.dataobj), np.asarray(image_obj.dataobj)[..., frame_t]):
                raise ValueError("cine frame/cine voxel correspondence failed")
    return {
        "source_sha256": image_sha256,
        "shape_xyzt": list(reversed(expected_shape_tzyx)),
        "orientation": "".join(str(x) for x in nib.aff2axcodes(image_obj.affine)),
        "units": "mm",
        "frame_mapping": list(expected_frame_mapping),
    }


def _transform_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    transform = metadata.get("reversible_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("metadata lacks reversible_transform")
    return dict(transform)


def invert_probabilities(probabilities_czyx: Any, metadata: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Map a cropped slab's C,Z,Y,X probabilities to native C,Z,Y,X.

    The returned support is conservative.  If slab indices repeat at a native
    edge, values are averaged rather than silently selecting one copy.  The
    operation is continuous (linear interpolation for an optional resample)
    and never performs an argmax; callers choose the declared post-processing
    rule after this native-grid export.
    """
    # Work on a contiguous copy: the coupled horizontal flip below creates a
    # negative-stride view which scipy/numpy do not consistently accept.
    if hasattr(probabilities_czyx, "detach"):
        probabilities_czyx = probabilities_czyx.detach().cpu().numpy()
    p = np.array(probabilities_czyx, dtype=np.float32, copy=True)
    if p.ndim != 4 or not np.isfinite(p).all() or np.any(p < -1e-6):
        raise ValueError("probabilities must be finite and non-negative C,Z,Y,X")
    native_shape = tuple(int(x) for x in metadata.get("native_shape_zyx", ()))
    if len(native_shape) != 3 or any(x < 1 for x in native_shape):
        raise ValueError("metadata native_shape_zyx is required")
    tr = _transform_from_metadata(metadata)
    z_indices = [int(x) for x in tr.get("z_indices", metadata.get("z_indices", []))]
    if len(z_indices) != p.shape[1]:
        raise ValueError("probability slab depth does not match metadata z_indices")
    crop = tr.get("crop", {})
    full_shape = tuple(int(x) for x in tr.get("current_shape_zyx", native_shape))
    if len(full_shape) != 3:
        raise ValueError("current_shape_zyx must be Z,Y,X")
    top, left = int(crop.get("top", 0)), int(crop.get("left", 0))
    crop_h, crop_w = int(crop.get("height", p.shape[-2])), int(crop.get("width", p.shape[-1]))
    if (crop_h, crop_w) != p.shape[-2:]:
        raise ValueError("crop metadata does not match probability shape")
    # Undo coupled flip before inserting the crop.  Geometric validity is
    # carried with the prediction rather than inferred from array shape.
    if bool(tr.get("flip", False)):
        p = p[..., ::-1]
    if full_shape[0] < 1:
        raise ValueError("current_shape_zyx depth must be positive")
    resample = tr.get("resample")
    work_z = full_shape[0] if resample else native_shape[0]
    current = np.zeros((p.shape[0], work_z, full_shape[1], full_shape[2]), dtype=np.float32)
    support = np.zeros((work_z, full_shape[1], full_shape[2]), dtype=bool)
    sums = np.zeros_like(current)
    counts = np.zeros_like(support, dtype=np.float32)
    # Crops may have been taken after edge padding.  Remove that padding from
    # coordinates and intersect the emitted crop with the native current grid.
    pad = crop.get("pad_lrtb", [0, 0, 0, 0])
    if len(pad) != 4:
        raise ValueError("crop pad_lrtb must contain left,right,top,bottom")
    top -= int(pad[2]); left -= int(pad[0])
    y1, x1 = top + crop_h, left + crop_w
    sy0, sx0 = max(0, -top), max(0, -left)
    sy1, sx1 = min(crop_h, full_shape[1] - top), min(crop_w, full_shape[2] - left)
    if sy1 <= sy0 or sx1 <= sx0:
        raise ValueError("crop does not overlap current grid")
    # A slab is a set of actual Z locations.  Repeated locations indicate
    # clipped edge padding from an older sampler and must not manufacture
    # support by averaging duplicate predictions.
    actual_z = [z for z in z_indices if 0 <= z < work_z]
    if len(actual_z) != len(set(actual_z)):
        raise ValueError("reversible transform contains duplicate Z indices")
    raw_valid = tr.get("geometric_valid")
    if raw_valid is None:
        if resample:
            raise ValueError("resampled inversion requires carried geometric_valid support")
        if any(not 0 <= z < work_z for z in z_indices) or sy0 != 0 or sx0 != 0 or sy1 != crop_h or sx1 != crop_w:
            raise ValueError("cropped/slab inversion requires carried geometric_valid support")
        valid_crop = np.ones((len(z_indices), crop_h, crop_w), dtype=bool)
    else:
        valid_crop = np.asarray(raw_valid, dtype=bool)
        if valid_crop.shape != (len(z_indices), crop_h, crop_w):
            raise ValueError("geometric_valid metadata does not match emitted crop")
        if bool(tr.get("flip", False)):
            valid_crop = valid_crop[..., ::-1]
    for iz, native_z in enumerate(z_indices):
        if not 0 <= native_z < work_z:
            # Invalid Z padding is represented by zeros and never contributes
            # to support, even when its image plane was edge-filled.
            continue
        local_valid = valid_crop[iz, sy0:sy1, sx0:sx1]
        if not local_valid.any():
            continue
        block = p[:, iz, sy0:sy1, sx0:sx1]
        dst = (slice(None), native_z, slice(top + sy0, top + sy1), slice(left + sx0, left + sx1))
        sums[dst] += np.where(local_valid[None], block, 0.0)
        counts[native_z, top + sy0:top + sy1, left + sx0:left + sx1] += local_valid.astype(np.float32)
    np.divide(sums, counts[None], out=current, where=counts[None] > 0)
    support = counts > 0
    # Undo an optional physical resample.  The transform stores a mapping from
    # current XYZ to native XYZ and output current_shape_zyx is before crop.
    if resample:
        from scipy import ndimage

        matrix = np.asarray(resample.get("native_to_current_matrix_zyx"), dtype=float)
        offset = np.asarray(resample.get("native_to_current_offset_zyx"), dtype=float)
        if matrix.shape != (3, 3) or offset.shape != (3,):
            raise ValueError("invalid resample transform")
        # affine_transform maps output native coordinates to input current.
        inv = np.linalg.inv(matrix)
        inv_offset = -inv @ offset
        out = np.empty((p.shape[0], *native_shape), dtype=np.float32)
        for c in range(p.shape[0]):
            out[c] = ndimage.affine_transform(current[c], inv, offset=inv_offset, output_shape=native_shape, order=1, mode="constant", cval=0.0)
        # ``order=0`` support interpolation is unsafe: it marks a native voxel
        # valid when the nearest current voxel is valid even though part of the
        # linear probability stencil falls outside support.  Interpolate the
        # boolean support with the *same* order as probabilities and require a
        # full stencil (value exactly one).  This gives conservative edge
        # support for asymmetric affine/resampling grids.
        support_interp = ndimage.affine_transform(
            support.astype(np.float32), inv, offset=inv_offset,
            output_shape=native_shape, order=1, mode="constant", cval=0.0,
            prefilter=False,
        )
        support = support_interp >= (1.0 - 1e-6)
        current = out
    # Interpolation can perturb a probability simplex.  Normalize only on
    # conservative support; never turn unsupported/zero voxels into a class.
    denom = np.sum(current, axis=0)
    supported = support & np.isfinite(denom) & (denom > 1e-8)
    normalized = np.zeros_like(current, dtype=np.float32)
    np.divide(current, denom[None], out=normalized, where=supported[None])
    return normalized, supported


__all__ = [
    "SCHEMA", "AXIS_ORDER", "UNKNOWN_LABEL", "sha256_file", "spacing_from_affine",
    "forbidden_mask_reference",
    "validate_geometry_record", "validate_sidecar", "load_sidecar", "voxel_to_world",
    "world_to_voxel", "validate_nifti_correspondence", "invert_probabilities",
]
