"""Frozen native-volume inference, scoring, and teacher-free deployment.

The training data bridge deliberately samples one slab at a time. Evaluation
uses the same native ``T,Z,Y,X`` image convention but enumerates a complete
frame and merges overlapping spatial/depth tiles before one native-grid
argmax. Image-only inference never opens a mask; labels are read only after
all predictions are frozen and only when the caller explicitly enables the
separate frozen scoring protocol.
"""

from __future__ import annotations

import copy
import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from self_audit.evaluation import class_metrics, confusion_matrix
from self_audit.model import digest

from .config import config_sha256, validate_volume_config
from .geometry import (
    AXIS_ORDER,
    UNKNOWN_LABEL,
    invert_probabilities,
    load_sidecar,
    validate_geometry_record,
    validate_nifti_correspondence,
)
from .data import _resample_zyx, _target_shape_from_fov


NUM_CLASSES = 4
_EVAL_SCHEMAS = {"self_audit.data.v2", "self_audit.volume.evaluation.v1"}
_EVAL_ROLES = {"development", "external_frozen", "evaluation"}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Evaluation masks are cache/native arrays.  Geometry metadata is intentionally
# a small, explicit contract rather than an open-ended bag of values: an
# affine or source/frame claim that the scorer does not understand must fail
# closed instead of being silently ignored.  Physical raw-source metadata lives
# in the hash-pinned geometry sidecar and is validated after frozen predictions.
_MASK_METADATA_KEYS = {
    "path", "sha256", "affine_xyz_to_world_mm", "native_affine_xyz_to_world_mm",
    "reference_affine_xyz_to_world_mm", "mask_affine_xyz_to_world_mm",
    "axis_order", "units", "orientation", "slice_order", "native_shape_zyx",
    "frame", "frame_index", "native_frame", "cache_frame", "source_frame",
    "raw_frame", "temporal_frame_id", "source", "source_path", "source_sha256",
    "source_image_sha256", "voxel_axis_order", "array_axis_order",
}


def _sha(path: Path) -> str:
    return digest(path)


def _resolve(base: Path, raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else base / p


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("evaluation manifest must be a JSON object")
    return value


def _model_forward(model: nn.Module, image: Tensor, valid: Tensor | None) -> Tensor:
    """Call a volume model with an explicit support mask."""
    try:
        logits = model(image, valid_mask=valid)
    except TypeError as exc:
        # Legacy test doubles accept only ``forward(image)``. Real native
        # models accept the mask; this compatibility seam never fabricates
        # support because valid is still carried through the merge metadata.
        if valid is not None and "valid_mask" not in str(exc):
            raise
        logits = model(image)
    if not isinstance(logits, Tensor) or logits.ndim != 5 or logits.shape[:2] != (image.shape[0], NUM_CLASSES):
        raise ValueError("volume model must return B,4,Z,H,W logits")
    if logits.shape[2:] != image.shape[2:] or not torch.isfinite(logits).all():
        raise ValueError("volume model returned non-finite or misaligned logits")
    return logits


def _context(image_tzyx: np.ndarray, t: int, z: int) -> np.ndarray:
    if image_tzyx.ndim != 4 or not (0 <= t < image_tzyx.shape[0]) or not (0 <= z < image_tzyx.shape[1]):
        raise ValueError("native context index is out of range")
    zdim = image_tzyx.shape[1]
    indices = [max(0, z - 1), z, min(zdim - 1, z + 1)]
    return np.asarray(image_tzyx[t, indices], dtype=np.float32)


def _full_stencil_support(
    source_support: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    """Return support whose *entire* linear interpolation stencil is valid.

    ``order=0`` support is insufficient for image resampling: a target center
    can be nearest to a valid source voxel while the linear image stencil also
    samples an invalid voxel.  Interpolating a float support mask with the same
    transform and requiring an all-one result is the conservative forward gate.
    """
    from scipy import ndimage

    values = ndimage.affine_transform(
        np.asarray(source_support, dtype=np.float32),
        np.asarray(matrix, dtype=float),
        offset=np.asarray(offset, dtype=float),
        output_shape=tuple(int(x) for x in output_shape),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return values >= (1.0 - 1e-6)


def _starts(size: int, tile: int, overlap: int) -> list[int]:
    if type(size) is not int or size < 1 or type(tile) is not int or tile < 1:
        raise ValueError("native tile dimensions must be positive integers")
    if type(overlap) is not int or overlap < 0 or overlap >= tile:
        raise ValueError("tile overlap must satisfy 0 <= overlap < tile")
    stride = tile - overlap
    out: list[int] = []
    start = 0
    while start < size:
        out.append(start)
        if start + tile >= size:
            break
        start += stride
    return out


def _tile_metadata(*, native_shape: tuple[int, int, int], current_shape: tuple[int, int, int],
                   z_indices: Sequence[int], top: int, left: int, height: int, width: int,
                   affine: Any, native_affine: Any, geometric_valid: np.ndarray,
                   flip: bool = False, resample: Mapping[str, Any] | None = None,
                   pad_lrtb: Sequence[int] = (0, 0, 0, 0)) -> dict[str, Any]:
    if geometric_valid.shape != (len(z_indices), height, width):
        raise ValueError("tile support shape does not match tile metadata")
    transform = {
        "z_indices": [int(x) for x in z_indices],
        "crop": {
            "top": int(top), "left": int(left), "height": int(height), "width": int(width),
            "original_hw": [int(current_shape[1]), int(current_shape[2])],
            "pad_lrtb": [int(x) for x in pad_lrtb],
        },
        "flip": bool(flip), "native_shape_zyx": list(native_shape), "current_shape_zyx": list(current_shape),
        "geometric_valid": geometric_valid.tolist(),
    }
    if resample is not None:
        transform["resample"] = dict(resample)
    return {
        "patient": "", "frame": 0, "z_indices": [int(x) for x in z_indices],
        "native_shape_zyx": list(native_shape), "current_shape_zyx": list(current_shape),
        "sample_shape_zyx": [len(z_indices), height, width],
        "native_affine_xyz_to_world_mm": np.asarray(native_affine, dtype=float).tolist(),
        "current_affine_xyz_to_world_mm": np.asarray(affine, dtype=float).tolist(),
        "axis_order": AXIS_ORDER, "reversible_transform": transform,
    }


def predict_native_volume(model: nn.Module, image_tzyx: np.ndarray, affine_xyz_to_world_mm: Any, *,
                          slab_depth: int = 3, tile_size: int | Sequence[int] = 256,
                          tile_overlap: int = 0, slab_overlap: int = 0,
                          native_valid: np.ndarray | None = None,
                          spacing_xyz_mm: Sequence[float] | None = None,
                          fov_xyz_mm: Sequence[float] | None = None,
                          resampling: str = "native_z", flip: bool = False,
                          device: str | torch.device = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Infer a complete native cine and merge tiled probabilities.

    Returns ``(probabilities, support)`` with shapes ``T,4,Z,Y,X`` and
    ``T,Z,Y,X``. Context channels use the existing edge-neighbour rule, while
    slab/tile support is explicit and no padded plane contributes to the
    native merge. When configured spacing/FOV is supplied, frames are
    resampled with the training bridge and probabilities/support are inverted
    to the native grid before return. No labels are accepted by this function.
    """
    image = np.asarray(image_tzyx)
    if image.ndim != 4 or any(int(x) < 1 for x in image.shape) or not np.isfinite(image).all():
        raise ValueError("evaluation image must be finite T,Z,Y,X")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("evaluation image must use normalized [0,1] intensities")
    if native_valid is None:
        native_valid_array = np.ones(image.shape, dtype=bool)
    else:
        native_valid_array = np.asarray(native_valid, dtype=bool)
        if native_valid_array.shape != image.shape:
            raise ValueError("native_valid must match image T,Z,Y,X")
    affine = np.asarray(affine_xyz_to_world_mm, dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
            or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12):
        raise ValueError("evaluation affine must be finite 4x4")
    if type(slab_depth) is not int or slab_depth < 1:
        raise ValueError("slab_depth must be a positive integer")
    if type(slab_overlap) is not int or slab_overlap < 0 or slab_overlap >= slab_depth:
        raise ValueError("slab_overlap must satisfy 0 <= overlap < slab_depth")
    if isinstance(tile_size, int):
        tile_h = tile_w = tile_size
    else:
        values = tuple(int(x) for x in tile_size)
        if len(values) != 2:
            raise ValueError("tile_size must be an integer or (height,width)")
        tile_h, tile_w = values
    if resampling not in {"native_z", "native", "configured", "configured_physical_fov"}:
        raise ValueError("evaluation resampling mode is unsupported")
    if resampling in {"native_z", "native"} and spacing_xyz_mm is not None:
        raise ValueError("native evaluation cannot silently ignore configured spacing_xyz_mm")
    if resampling == "configured_physical_fov" and fov_xyz_mm is None:
        raise ValueError("configured_physical_fov evaluation requires fov_xyz_mm")
    spacing: tuple[float, float, float] | None = None
    if spacing_xyz_mm is not None:
        values = tuple(float(x) for x in spacing_xyz_mm)
        if len(values) != 3 or any(not np.isfinite(x) or x <= 0 for x in values):
            raise ValueError("evaluation spacing_xyz_mm must be a positive XYZ triple")
        spacing = values
    fov: tuple[float, float, float] | None = None
    if fov_xyz_mm is not None:
        values = tuple(float(x) for x in fov_xyz_mm)
        if len(values) != 3 or any(not np.isfinite(x) or x <= 0 for x in values):
            raise ValueError("evaluation fov_xyz_mm must be a positive XYZ triple")
        fov = values
    if resampling in {"configured", "configured_physical_fov"} and spacing is None:
        raise ValueError("configured evaluation requires spacing_xyz_mm; refusing native-only fallback")
    native_shape = tuple(int(x) for x in image.shape[1:])
    native_affine = affine.copy()
    native_spacing = np.linalg.norm(native_affine[:3, :3], axis=0)
    fov_shape_xyz: tuple[int, int, int] | None = None
    if fov is not None:
        fov_spacing = np.asarray(spacing if spacing is not None else native_spacing, dtype=float)
        fov_shape_xyz = _target_shape_from_fov(fov, fov_spacing)
        slab_depth = max(int(slab_depth), int(fov_shape_xyz[2]))
        # A physical XY FOV is the model's emitted window, independent of the
        # caller's memory-oriented tile hint.  Edge windows are padded with
        # invalid support and inverted back to native coordinates below.
        tile_h, tile_w = int(fov_shape_xyz[1]), int(fov_shape_xyz[0])
    model_device = torch.device(device)
    # A frozen evaluation must not move, train/eval-toggle, or otherwise
    # mutate the caller's live student/EMA object. One detached copy is made
    # per complete cine, not per tile.
    selected = copy.deepcopy(model).to(model_device)
    previous_training = selected.training
    selected.eval()
    probs_out = np.zeros((image.shape[0], NUM_CLASSES, image.shape[1], image.shape[2], image.shape[3]), dtype=np.float32)
    counts_out = np.zeros((image.shape[0], image.shape[1], image.shape[2], image.shape[3]), dtype=np.float32)
    try:
        with torch.inference_mode():
            for frame in range(image.shape[0]):
                frame_valid = native_valid_array[frame]
                # Invalid native values are untrusted input, not merely output
                # support.  Zero them before context construction so a spatial
                # receptive field (including 2.5-D neighbour slices) cannot
                # observe a value that will later be marked unsupported.
                native_frame = np.where(frame_valid, image[frame], 0.0).astype(np.float32, copy=False)
                resample_info: dict[str, Any] | None = None
                current_affine = native_affine
                if spacing is not None:
                    current_frame, _resampled_valid, resample_info = _resample_zyx(
                        native_frame, native_affine, spacing, order=1, cval=0.0,
                    )
                    # Preserve caller support through the same physical
                    # transform.  Both the declared native support and the
                    # geometric source grid use full linear-stencil validity;
                    # nearest-centre support would let invalid neighbours leak
                    # into a valid model input.
                    matrix = np.asarray(resample_info["native_to_current_matrix_zyx"], dtype=float)
                    offset = np.asarray(resample_info["native_to_current_offset_zyx"], dtype=float)
                    current_valid = _full_stencil_support(
                        frame_valid.astype(np.float32), matrix, offset, tuple(current_frame.shape)
                    )
                    current_valid &= _full_stencil_support(
                        np.ones(native_frame.shape, dtype=np.float32), matrix, offset,
                        tuple(current_frame.shape),
                    )
                    # ``_resample_zyx`` retains a centre-in-grid receipt; keep
                    # it as an additional conservative guard for transformed
                    # grids with unusual affine directions.
                    current_valid &= np.asarray(_resampled_valid, dtype=bool)
                    current_frame = np.where(current_valid, current_frame, 0.0).astype(np.float32, copy=False)
                    current_affine = np.asarray(resample_info["current_affine_xyz_to_world_mm"], dtype=float)
                else:
                    current_frame = native_frame
                    current_valid = frame_valid.copy()
                current_shape = tuple(int(x) for x in current_frame.shape)
                # For a configured physical grid, merge all tile/slab outputs
                # on that complete current frame before one inverse transform.
                # Inverting each crop independently makes a linear native
                # stencil appear incomplete at every tile boundary.
                current_probability_sum = (
                    np.zeros((NUM_CLASSES, *current_shape), dtype=np.float32)
                    if spacing is not None else None
                )
                current_probability_count = (
                    np.zeros(current_shape, dtype=np.float32) if spacing is not None else None
                )
                y_starts = _starts(current_shape[1], tile_h, tile_overlap)
                x_starts = _starts(current_shape[2], tile_w, tile_overlap)
                z_starts = _starts(current_shape[0], slab_depth, slab_overlap)
                for z_start in z_starts:
                    if fov_shape_xyz is not None:
                        z_indices = list(range(z_start, z_start + slab_depth))
                    else:
                        z_indices = list(range(z_start, min(z_start + slab_depth, current_shape[0])))
                    for top in y_starts:
                        for left in x_starts:
                            height = min(tile_h, current_shape[1] - top)
                            width = min(tile_w, current_shape[2] - left)
                            tile = np.stack([
                                (_context(current_frame[None], 0, z)[:, top:top + height, left:left + width]
                                 if 0 <= z < current_shape[0] else
                                 np.zeros((3, height, width), dtype=np.float32))
                                for z in z_indices
                            ], axis=1)
                            # Out-of-grid Z indices are physical FOV padding,
                            # not repeated support.  Numpy indexing above is
                            # guarded because these indices are intentionally
                            # outside the current grid.
                            if fov_shape_xyz is not None:
                                valid = np.stack([
                                    current_valid[z, top:top + height, left:left + width]
                                    if 0 <= z < current_shape[0] else np.zeros((height, width), dtype=bool)
                                    for z in z_indices
                                ], axis=0)
                                pad_bottom = tile_h - height
                                pad_right = tile_w - width
                                if pad_bottom or pad_right:
                                    # Padded FOV positions are unsupported;
                                    # edge replication would expose a valid
                                    # neighbour through a convolution before
                                    # the support mask can abstain the output.
                                    tile = np.pad(tile, ((0, 0), (0, 0), (0, pad_bottom), (0, pad_right)),
                                                  mode="constant", constant_values=0.0)
                                    valid = np.pad(valid, ((0, 0), (0, pad_bottom), (0, pad_right)),
                                                    mode="constant", constant_values=False)
                                metadata_height, metadata_width = tile_h, tile_w
                                pad_lrtb = (0, pad_right, 0, pad_bottom)
                            else:
                                valid = current_valid[z_indices, top:top + height, left:left + width]
                                metadata_height, metadata_width = height, width
                                pad_lrtb = (0, 0, 0, 0)
                            if flip:
                                tile = tile[..., ::-1].copy()
                                valid = valid[..., ::-1].copy()
                            metadata = _tile_metadata(
                                native_shape=native_shape, current_shape=current_shape,
                                z_indices=z_indices, top=top, left=left,
                                height=metadata_height, width=metadata_width,
                                affine=current_affine, native_affine=native_affine,
                                geometric_valid=valid, flip=flip, resample=resample_info,
                                pad_lrtb=pad_lrtb,
                            )
                            metadata["patient"] = "inference"; metadata["frame"] = int(frame)
                            input_tensor = torch.from_numpy(tile[None]).to(model_device)
                            valid_tensor = torch.from_numpy(valid[None]).to(model_device)
                            logits = _model_forward(selected, input_tensor, valid_tensor)
                            probabilities = logits.softmax(1)[0].detach().cpu().numpy()
                            if spacing is None:
                                native, support = invert_probabilities(probabilities, metadata)
                                probs_out[frame] += np.where(support[None], native, 0.0)
                                counts_out[frame] += support.astype(np.float32)
                            else:
                                # Undo the coupled flip and place only actual
                                # current-grid voxels.  FOV padding and
                                # out-of-grid slab planes remain unsupported.
                                if flip:
                                    probabilities = probabilities[..., ::-1].copy()
                                    valid = valid[..., ::-1].copy()
                                assert current_probability_sum is not None
                                assert current_probability_count is not None
                                for iz, current_z in enumerate(z_indices):
                                    if not 0 <= current_z < current_shape[0]:
                                        continue
                                    local_h = min(height, current_shape[1] - top)
                                    local_w = min(width, current_shape[2] - left)
                                    if local_h <= 0 or local_w <= 0:
                                        continue
                                    local_valid = valid[iz, :local_h, :local_w]
                                    if not local_valid.any():
                                        continue
                                    dst = (slice(None), current_z,
                                           slice(top, top + local_h), slice(left, left + local_w))
                                    current_probability_sum[dst] += np.where(
                                        local_valid[None], probabilities[:, iz, :local_h, :local_w], 0.0
                                    )
                                    current_probability_count[current_z, top:top + local_h, left:left + local_w] += local_valid.astype(np.float32)
                if spacing is not None:
                    assert current_probability_sum is not None
                    assert current_probability_count is not None
                    current_probabilities = np.zeros_like(current_probability_sum)
                    np.divide(
                        current_probability_sum,
                        current_probability_count[None],
                        out=current_probabilities,
                        where=current_probability_count[None] > 0,
                    )
                    frame_metadata = _tile_metadata(
                        native_shape=native_shape, current_shape=current_shape,
                        z_indices=list(range(current_shape[0])), top=0, left=0,
                        height=current_shape[1], width=current_shape[2],
                        affine=current_affine, native_affine=native_affine,
                        geometric_valid=current_probability_count > 0,
                        flip=False, resample=resample_info,
                    )
                    native, support = invert_probabilities(current_probabilities, frame_metadata)
                    probs_out[frame] += np.where(support[None], native, 0.0)
                    counts_out[frame] += support.astype(np.float32)
    finally:
        if previous_training:
            selected.train()
    np.divide(probs_out, counts_out[:, None], out=probs_out, where=counts_out[:, None] > 0)
    # Inverse resampling can interpolate support back onto a native voxel that
    # was declared invalid.  The native declaration is authoritative at the
    # final merge boundary, and unsupported probabilities are always zeroed.
    support_out = (counts_out > 0) & native_valid_array
    probs_out = np.where(support_out[:, None], probs_out, 0.0).astype(np.float32, copy=False)
    return probs_out, support_out


def _evaluation_records(manifest: Path, manifest_sha256: str, geometry: Path, geometry_sha256: str, *,
                        split: str | None, synthetic: bool,
                        require_physical_masks: bool = False) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if _sha(manifest) != manifest_sha256:
        raise ValueError("evaluation manifest hash mismatch")
    if _sha(geometry) != geometry_sha256:
        raise ValueError("evaluation geometry hash mismatch")
    payload = _json(manifest)
    if payload.get("schema") not in _EVAL_SCHEMAS or payload.get("role") not in _EVAL_ROLES:
        raise ValueError("separate frozen evaluation manifest required")
    if bool(payload.get("synthetic", False)) != bool(synthetic):
        raise ValueError("synthetic provenance mismatch; pass --synthetic only for manufactured fixtures")
    declared_split = payload.get("split", payload.get("scope", payload.get("role")))
    if split is not None and declared_split not in {split, payload.get("role")}:
        raise ValueError("evaluation split does not match manifest")
    sidecar = load_sidecar(geometry, geometry_sha256)
    if bool(sidecar.get("synthetic", False)) != bool(synthetic):
        raise ValueError("geometry synthetic provenance does not match evaluation manifest")
    if sidecar.get("source_manifest_sha256") not in (None, manifest_sha256):
        raise ValueError("geometry sidecar source manifest hash mismatch")
    records = payload.get("patients", payload.get("records"))
    if not isinstance(records, list) or not records:
        raise ValueError("evaluation manifest requires non-empty patients/records")
    ids = [str(item.get("id")) for item in records if isinstance(item, Mapping)]
    if len(ids) != len(records) or any(not pid or pid == "None" for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("evaluation patient IDs must be unique")
    declared_train_ids = payload.get("train_patient_ids", payload.get("training_patient_ids", payload.get("train_patients", [])))
    if declared_train_ids is not None:
        if not isinstance(declared_train_ids, list) or any(not isinstance(pid, str) for pid in declared_train_ids):
            raise ValueError("train_patient_ids must be a list of strings")
        if set(ids) & set(declared_train_ids):
            raise ValueError("evaluation patient overlaps declared TRAIN patients")
    normalized: list[dict[str, Any]] = []
    for record, pid in zip(records, ids, strict=True):
        if not isinstance(record, Mapping) or not isinstance(record.get("image"), str) or not isinstance(record.get("image_sha256"), str):
            raise ValueError(f"evaluation image hash is required for {pid}")
        image_path = _resolve(manifest.parent, record["image"])
        if not image_path.is_file() or _sha(image_path) != record["image_sha256"]:
            raise ValueError(f"evaluation image hash mismatch for {pid}")
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        if image.ndim != 4 or any(int(x) < 1 for x in image.shape) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"invalid native evaluation image for {pid}")
        if pid not in sidecar.get("patients", {}):
            raise ValueError(f"geometry sidecar missing evaluation patient {pid}")
        geo = validate_geometry_record(sidecar["patients"][pid], expected_shape=tuple(image.shape))
        if geo.get("image_sha256") != record["image_sha256"]:
            raise ValueError(f"geometry/image hash mismatch for {pid}")
        masks = record.get("masks")
        if masks is not None and (not isinstance(masks, Mapping)
                                  or any(not isinstance(k, str) for k in masks)):
            raise ValueError(f"evaluation masks must be a frame-keyed object for {pid}")
        if isinstance(masks, Mapping):
            for frame_key in masks:
                try:
                    frame_index = int(frame_key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"evaluation mask frame key must be an integer for {pid}: {frame_key}") from exc
                if not 0 <= frame_index < int(image.shape[0]):
                    raise ValueError(f"evaluation mask frame is outside native image geometry for {pid}: {frame_key}")
        source = geo.get("source_image")
        if not synthetic and source is None:
            raise ValueError(f"real evaluation requires hash-pinned source_image for {pid}")
        physical_masks = geo.get("mask_sources")
        if require_physical_masks:
            if not isinstance(masks, Mapping) or not masks:
                raise ValueError(f"real frozen scoring requires masks for {pid}")
            if not isinstance(physical_masks, Mapping):
                raise ValueError(f"real frozen scoring requires hash-pinned mask_sources for {pid}")
            if {str(k) for k in physical_masks} != {str(k) for k in masks}:
                raise ValueError(f"real mask_sources/frame set does not match cached masks for {pid}")
            for raw_frame, raw_item in physical_masks.items():
                try:
                    raw_frame_int = int(raw_frame)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"physical mask frame key must be an integer for {pid}: {raw_frame}") from exc
                frame_map = [int(x) for x in geo["temporal_frame_mapping"]]
                if not 0 <= raw_frame_int < int(image.shape[0]) and raw_frame_int not in frame_map:
                    raise ValueError(f"physical mask frame is absent from temporal mapping for {pid}: {raw_frame}")
                if not isinstance(raw_item, Mapping):
                    raise ValueError(f"invalid physical mask source metadata for {pid} frame {raw_frame}")
                source_item = raw_item.get("source", raw_item)
                if not isinstance(source_item, Mapping):
                    raise ValueError(f"invalid physical mask source metadata for {pid} frame {raw_frame}")
                source_path = source_item.get("path", source_item.get("source_path"))
                source_sha = source_item.get("sha256", source_item.get("source_sha256"))
                if not isinstance(source_path, str) or not isinstance(source_sha, str) or not _SHA256_RE.fullmatch(source_sha):
                    raise ValueError(f"physical mask source requires path and sha256 for {pid} frame {raw_frame}")
        if source is not None:
            if not isinstance(source, Mapping):
                raise ValueError(f"source_image metadata must be an object for {pid}")
            source_path = source.get("path", source.get("source_path"))
            source_sha = source.get("sha256", source.get("source_sha256"))
            if not isinstance(source_path, str) or not isinstance(source_sha, str):
                raise ValueError(f"source_image metadata requires path and sha256 for {pid}")
            validate_nifti_correspondence(
                _resolve(manifest.parent, source_path), image_sha256=source_sha,
                expected_shape_tzyx=tuple(int(x) for x in image.shape),
                expected_affine_xyz_to_world_mm=geo["affine_xyz_to_world_mm"],
                expected_orientation=str(geo["orientation"]),
                expected_frame_mapping=[int(x) for x in geo["temporal_frame_mapping"]],
                masks=None, base_dir=manifest.parent,
                # Bind the already-loaded prepared image before any model
                # inference.  Image-only mode must reject a shape/affine-valid
                # but voxel-mismatched raw NIfTI without opening labels.
                prepared_image_tzyx=image,
            )
        normalized.append({"id": pid, "image": image_path, "record": dict(record), "geometry": geo})
    return payload, sidecar, normalized


def _mask_frame_index(item: Mapping[str, Any], frame: str, geometry: Mapping[str, Any], pid: str) -> tuple[int, int]:
    """Validate cached and physical frame identities without opening a mask."""
    try:
        frame_index = int(frame)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"evaluation mask frame key must be an integer for {pid}: {frame}") from exc
    mapping = geometry.get("temporal_frame_mapping")
    if not isinstance(mapping, list) or not 0 <= frame_index < len(mapping):
        raise ValueError(f"evaluation mask frame is outside native image geometry for {pid}: {frame}")
    raw_frame = mapping[frame_index]
    for key in ("frame", "frame_index", "native_frame", "cache_frame"):
        if key in item:
            value = item[key]
            if type(value) is not int or value != frame_index:
                raise ValueError(f"evaluation mask {key} disagrees with cached frame for {pid} frame {frame}")
    for key in ("source_frame", "raw_frame", "temporal_frame_id"):
        if key in item:
            value = item[key]
            if type(value) is not int or value != raw_frame:
                raise ValueError(f"evaluation mask {key} disagrees with source frame for {pid} frame {frame}")
    return frame_index, int(raw_frame)


def _mask_source_metadata(base: Path, item: Mapping[str, Any], *, pid: str, frame: str) -> None:
    """Check optional raw-source claims and hashes after scoring is authorized."""
    source = item.get("source")
    if source is not None and not isinstance(source, Mapping):
        raise ValueError(f"invalid source metadata for {pid} frame {frame}")
    direct_path = item.get("source_path")
    direct_sha = item.get("source_sha256")
    if direct_path is not None or direct_sha is not None:
        if not isinstance(direct_path, str) or not isinstance(direct_sha, str):
            raise ValueError(f"source_path/source_sha256 must be supplied together for {pid} frame {frame}")
        source = {"path": direct_path, "sha256": direct_sha}
    if source is None:
        if item.get("source_image_sha256") is not None:
            raise ValueError(f"source_image_sha256 requires a source path for {pid} frame {frame}")
        return
    source_path = source.get("path", source.get("source_path"))
    source_sha = source.get("sha256", source.get("source_sha256"))
    if not isinstance(source_path, str) or not isinstance(source_sha, str) or not _SHA256_RE.fullmatch(source_sha):
        raise ValueError(f"source metadata requires a valid path and sha256 for {pid} frame {frame}")
    path = _resolve(base, source_path)
    if not path.is_file() or _sha(path) != source_sha:
        raise ValueError(f"evaluation mask source hash mismatch for {pid} frame {frame}")


def _validate_mask_contract(base: Path, item: Mapping[str, Any], *, shape: tuple[int, int, int],
                            geometry: Mapping[str, Any], pid: str, frame: str,
                            synthetic: bool) -> tuple[int, int]:
    if not isinstance(item, Mapping):
        raise ValueError(f"invalid frozen mask metadata for {pid} frame {frame}")
    unknown = set(str(key) for key in item) - _MASK_METADATA_KEYS
    if unknown:
        raise ValueError(f"unsupported frozen mask metadata for {pid} frame {frame}: {sorted(unknown)}")
    if not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
        raise ValueError(f"invalid frozen mask metadata for {pid} frame {frame}")
    if not _SHA256_RE.fullmatch(str(item["sha256"])):
        raise ValueError(f"invalid frozen mask hash for {pid} frame {frame}")
    frame_index, raw_frame = _mask_frame_index(item, frame, geometry, pid)
    expected_affine = np.asarray(geometry["affine_xyz_to_world_mm"], dtype=float)
    for key in ("affine_xyz_to_world_mm", "native_affine_xyz_to_world_mm",
                "reference_affine_xyz_to_world_mm", "mask_affine_xyz_to_world_mm"):
        if key not in item:
            continue
        affine = np.asarray(item[key], dtype=float)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError(f"invalid mask affine for {pid} frame {frame}")
        if not np.allclose(affine, expected_affine, atol=1e-5, rtol=1e-6):
            raise ValueError(f"mask affine does not match native image geometry for {pid} frame {frame}")
    if "units" in item and item["units"] != "mm":
        raise ValueError(f"mask physical units must be mm for {pid} frame {frame}")
    for key in ("axis_order", "voxel_axis_order", "array_axis_order"):
        if key in item:
            value = item[key]
            if isinstance(value, (list, tuple)):
                value = ",".join(str(x) for x in value)
            if str(value) not in {AXIS_ORDER, "Y,X", "X,Y"}:
                raise ValueError(f"mask axis order is unsupported for {pid} frame {frame}")
    for key in ("orientation", "slice_order"):
        if key in item and str(item[key]) != str(geometry.get(key)):
            raise ValueError(f"mask {key} does not match native image geometry for {pid} frame {frame}")
    if "native_shape_zyx" in item and tuple(int(x) for x in item["native_shape_zyx"]) != tuple(shape):
        raise ValueError(f"mask native shape does not match image geometry for {pid} frame {frame}")
    source_image_sha = item.get("source_image_sha256")
    if source_image_sha is not None:
        if not isinstance(source_image_sha, str) or not _SHA256_RE.fullmatch(source_image_sha):
            raise ValueError(f"invalid source image hash for {pid} frame {frame}")
        geometry_source = geometry.get("source_image")
        expected_source = geometry_source.get("sha256", geometry_source.get("source_sha256")) if isinstance(geometry_source, Mapping) else None
        if expected_source != source_image_sha:
            raise ValueError(f"mask source image hash does not match image geometry for {pid} frame {frame}")
    _mask_source_metadata(base, item, pid=pid, frame=frame)
    if not synthetic and not isinstance(geometry.get("mask_sources"), Mapping):
        raise ValueError(f"real frozen scoring requires physical mask provenance for {pid} frame {frame}")
    return frame_index, raw_frame


def _read_evaluation_mask(base: Path, item: Mapping[str, Any], shape: tuple[int, int, int], pid: str, frame: str,
                          *, geometry: Mapping[str, Any], synthetic: bool) -> np.ndarray:
    _validate_mask_contract(base, item, shape=shape, geometry=geometry, pid=pid, frame=frame, synthetic=synthetic)
    if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
        raise ValueError(f"invalid frozen mask metadata for {pid} frame {frame}")
    path = _resolve(base, item["path"])
    if not path.is_file() or _sha(path) != item["sha256"]:
        raise ValueError(f"frozen mask hash mismatch for {pid} frame {frame}")
    value = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.int64)
    if value.shape != shape or not np.isin(value, [0, 1, 2, 3, UNKNOWN_LABEL]).all():
        raise ValueError(f"frozen mask geometry/ontology mismatch for {pid} frame {frame}")
    return value


def _patient_metrics(frame_rows: list[dict[str, Any]], ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    patients: dict[str, Any] = {}
    for pid in ids:
        subset = [row for row in frame_rows if row["patient"] == pid]
        patients[pid] = {}
        for index, name in enumerate(("BG", "RV", "MYO", "LV")):
            values = [row["raw"][index]["dice"] for row in subset if row["raw"][index]["dice"] is not None]
            patients[pid][name] = float(np.mean(values)) if values else None
    macro: dict[str, Any] = {}
    for name in ("BG", "RV", "MYO", "LV"):
        values = [row[name] for row in patients.values() if row[name] is not None]
        macro[name] = {"dice": float(np.mean(values)) if values else None, "present_patients": len(values)}
    return patients, macro


def evaluate_manifest(model: nn.Module, manifest: str | Path, manifest_sha256: str, geometry: str | Path,
                      geometry_sha256: str, *, teacher: nn.Module | None = None,
                      prediction_model: str = "student", split: str | None = None,
                      allow_labels: bool = False, synthetic: bool = False, slab_depth: int = 3,
                      tile_size: int | Sequence[int] = 256, tile_overlap: int = 0, slab_overlap: int = 0,
                      config: Mapping[str, Any] | None = None,
                      checkpoint_sha256: str | None = None, predictions_dir: str | Path | None = None,
                      device: str | torch.device = "cpu", cli_flags: Mapping[str, Any] | None = None,
                      evaluation_source_sha256: str | None = None) -> dict[str, Any]:
    """Run frozen native inference and optional explicit frozen scoring.

    Every record is predicted before any reference mask is opened. Set
    ``allow_labels=True`` only for a declared development/external-frozen
    protocol; the default is image-only deployment inference.
    """
    if prediction_model not in {"student", "ema"}:
        raise ValueError("prediction_model must be student or ema")
    selected = teacher if prediction_model == "ema" else model
    if selected is None:
        raise ValueError("EMA prediction requested without teacher")
    if evaluation_source_sha256 is None:
        # Hash the executed native-volume source tree, not merely the caller's
        # checkpoint manifest, so receipts identify the implementation that
        # produced the frozen probabilities.
        from .checkpoint import source_sha256 as _source_sha256
        evaluation_source_sha256 = _source_sha256()
    geometry_config = config.get("geometry", {}) if isinstance(config, Mapping) else {}
    if not isinstance(geometry_config, Mapping):
        raise ValueError("evaluation config.geometry must be an object")
    configured_resampling = str(geometry_config.get("resampling", "native_z"))
    configured_spacing = geometry_config.get("spacing_xyz_mm")
    configured_fov = geometry_config.get("fov_xyz_mm")
    configured_flip = bool(geometry_config.get("flip", False))
    if configured_resampling in {"configured", "configured_physical_fov"} and configured_spacing is None:
        raise ValueError("configured evaluation requires spacing_xyz_mm; refusing native-only fallback")
    if configured_resampling == "configured_physical_fov" and configured_fov is None:
        raise ValueError("configured_physical_fov evaluation requires fov_xyz_mm")
    manifest_path = Path(manifest).resolve()
    geometry_path = Path(geometry).resolve()
    payload, _sidecar, records = _evaluation_records(
        manifest_path, manifest_sha256, geometry_path, geometry_sha256,
        split=split, synthetic=synthetic,
        require_physical_masks=bool(allow_labels and not synthetic),
    )
    # Preserve both the caller's declared inference recipe and the effective
    # per-record windows actually consumed by predict_native_volume.  A
    # physical FOV can enlarge slab/tile dimensions, so a single CLI integer
    # is not sufficient to reconstruct the executed protocol.
    declared_tile = ([int(tile_size), int(tile_size)] if isinstance(tile_size, int)
                     else [int(x) for x in tile_size])
    declared_protocol = {
        "slab_depth": int(slab_depth), "tile_size": declared_tile,
        "tile_overlap": int(tile_overlap), "slab_overlap": int(slab_overlap),
        "spacing_xyz_mm": list(configured_spacing) if configured_spacing is not None else None,
        "fov_xyz_mm": list(configured_fov) if configured_fov is not None else None,
        "flip": bool(configured_flip), "resampling": configured_resampling,
    }
    effective_protocols: dict[str, Any] = {}
    for item in records:
        affine = np.asarray(item["geometry"]["affine_xyz_to_world_mm"], dtype=float)
        native_spacing = np.linalg.norm(affine[:3, :3], axis=0)
        fov_shape_xyz = None
        if configured_fov is not None:
            fov_spacing = np.asarray(configured_spacing if configured_spacing is not None else native_spacing, dtype=float)
            fov_shape_xyz = _target_shape_from_fov(configured_fov, fov_spacing)
        effective_protocols[item["id"]] = {
            "slab_depth": max(int(slab_depth), int(fov_shape_xyz[2])) if fov_shape_xyz is not None else int(slab_depth),
            "tile_size": ([int(fov_shape_xyz[1]), int(fov_shape_xyz[0])] if fov_shape_xyz is not None
                           else list(declared_tile)),
            "tile_overlap": int(tile_overlap), "slab_overlap": int(slab_overlap),
            "spacing_xyz_mm": list(configured_spacing) if configured_spacing is not None else None,
            "fov_xyz_mm": list(configured_fov) if configured_fov is not None else None,
            "flip": bool(configured_flip), "resampling": configured_resampling,
            "fov_derived_window": fov_shape_xyz is not None,
        }
    frozen: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for item in records:
        image = np.load(item["image"], mmap_mode="r", allow_pickle=False)
        probs, support = predict_native_volume(selected, image, item["geometry"]["affine_xyz_to_world_mm"],
                                                slab_depth=slab_depth, tile_size=tile_size,
                                                tile_overlap=tile_overlap, slab_overlap=slab_overlap,
                                                spacing_xyz_mm=configured_spacing, fov_xyz_mm=configured_fov,
                                                resampling=configured_resampling, flip=configured_flip,
                                                device=device)
        frozen.append((item, probs, support))
    frame_rows: list[dict[str, Any]] = []
    pooled = np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=np.int64)
    labels_read = 0
    if allow_labels:
        for item, probabilities, support in frozen:
            masks = item["record"].get("masks")
            if not isinstance(masks, Mapping) or not masks:
                raise ValueError("explicit frozen scoring requires masks for every evaluated patient")
            prediction = probabilities.argmax(axis=1).astype(np.uint8)
            loaded_masks: dict[str, np.ndarray] = {}
            for frame_key, mask_item in sorted(masks.items(), key=lambda kv: int(kv[0])):
                frame = int(frame_key)
                if not 0 <= frame < prediction.shape[0]:
                    raise ValueError(f"frozen mask frame is out of range for {item['id']}")
                labels = _read_evaluation_mask(
                    manifest_path.parent, mask_item, tuple(prediction.shape[1:]), item["id"], str(frame_key),
                    geometry=item["geometry"], synthetic=synthetic,
                )
                loaded_masks[str(frame_key)] = labels
            # Real scoring is permitted only with hash-pinned physical raw
            # sources and an exact raw-to-cache correspondence.  This call is
            # deliberately after all image-only predictions are frozen and
            # after explicit scoring authorization, so image-only evaluation
            # never opens a reference mask or its raw source.
            if not synthetic:
                physical_image = item["geometry"].get("source_image")
                physical_masks = item["geometry"].get("mask_sources")
                if not isinstance(physical_image, Mapping) or not isinstance(physical_masks, Mapping):
                    raise ValueError(f"real frozen scoring requires physical image/mask provenance for {item['id']}")
                source_path = physical_image.get("path", physical_image.get("source_path"))
                source_sha = physical_image.get("sha256", physical_image.get("source_sha256"))
                if not isinstance(source_path, str) or not isinstance(source_sha, str):
                    raise ValueError(f"real frozen scoring requires hash-pinned source_image for {item['id']}")
                validate_nifti_correspondence(
                    _resolve(manifest_path.parent, source_path), image_sha256=source_sha,
                    expected_shape_tzyx=tuple(int(x) for x in np.load(item["image"], mmap_mode="r", allow_pickle=False).shape),
                    expected_affine_xyz_to_world_mm=item["geometry"]["affine_xyz_to_world_mm"],
                    expected_orientation=str(item["geometry"]["orientation"]),
                    expected_frame_mapping=[int(x) for x in item["geometry"]["temporal_frame_mapping"]],
                    masks=physical_masks, base_dir=manifest_path.parent,
                    prepared_image_tzyx=np.load(item["image"], mmap_mode="r", allow_pickle=False),
                    prepared_masks_tzyx=loaded_masks,
                )
            for frame_key, labels in sorted(loaded_masks.items(), key=lambda kv: int(kv[0])):
                frame = int(frame_key)
                pred = np.where(support[frame], prediction[frame], UNKNOWN_LABEL).astype(np.uint8)
                cm = confusion_matrix(labels, pred)
                pooled += cm
                frame_rows.append({"patient": item["id"], "frame": frame, "raw": class_metrics(cm),
                                   "raw_confusion": cm.tolist(), "supported_voxels": int(support[frame].sum())})
                labels_read += 1
    prediction_files: dict[str, Any] = {}
    if predictions_dir is not None:
        prediction_root = Path(predictions_dir)
        if prediction_root.exists():
            raise ValueError(f"prediction output directory already exists: {prediction_root}")
        prediction_root.mkdir(parents=True)
        for item, probabilities, support in frozen:
            probability_path = prediction_root / f"{item['id']}.npy"
            support_path = prediction_root / f"{item['id']}_support.npy"
            np.save(probability_path, probabilities)
            np.save(support_path, support)
            prediction_files[item["id"]] = {
                "probabilities": probability_path.name, "probabilities_sha256": _sha(probability_path),
                "support": support_path.name, "support_sha256": _sha(support_path),
            }
    patients, macro = _patient_metrics(frame_rows, [item["id"] for item in records])
    return {
        "schema": "self_audit.volume.frozen_eval.v2", "scope": payload.get("scope", payload.get("role")),
        "split": split or payload.get("split", payload.get("role")), "synthetic": bool(synthetic),
        "prediction_model": prediction_model, "manifest_sha256": manifest_sha256,
        "geometry_sha256": geometry_sha256, "checkpoint_sha256": checkpoint_sha256,
        "labels_allowed": bool(allow_labels), "labels_read": int(labels_read),
        "patients": patients, "frames": frame_rows, "macro_patient": macro,
        "pooled_confusion": pooled.tolist(),
        "native_probability_shapes": {item["id"]: list(pair[1].shape) for item, pair in zip(records, frozen, strict=True)},
        "native_support_voxels": {item["id"]: int(pair[2].sum()) for item, pair in zip(records, frozen, strict=True)},
        "prediction_files": prediction_files,
        "inference_protocol": {"declared": declared_protocol, "effective_by_record": effective_protocols},
        "cli_flags": dict(cli_flags or {}),
        "evaluation_source_sha256": evaluation_source_sha256,
        "executed_source_sha256": evaluation_source_sha256,
        "source_sha256": evaluation_source_sha256,
        "quality": "FROZEN_DEVELOPMENT_ONLY" if allow_labels else "FROZEN_IMAGE_ONLY",
        "physical_boundary_metrics": "NOT_COMPUTED; no spacing-certified HD95 boundary protocol",
        "warning": "Synthetic/CPU results are software evidence only; no patient accuracy claim is made.",
    }


def evaluate_frozen(model: nn.Module, batches: Iterable[Mapping[str, Any]], *, teacher: nn.Module | None = None,
                    prediction_model: str = "student", device: str | torch.device = "cpu") -> dict[str, Any]:
    """Backward-compatible fixed-batch scorer used by engine smoke tests."""
    if prediction_model not in {"student", "ema"}:
        raise ValueError("prediction_model must be student or ema")
    selected_source = teacher if prediction_model == "ema" else model
    if selected_source is None:
        raise ValueError("EMA prediction requested without teacher")
    selected = copy.deepcopy(selected_source).to(device)
    previous = selected.training
    selected.eval()
    frame_rows: list[dict[str, Any]] = []
    pooled = np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=np.int64)
    batches_seen = 0
    total_ce = 0.0
    total_voxels = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            image = torch.as_tensor(batch["image"], device=device)
            if image.ndim != 5 or image.shape[1] != 3:
                raise ValueError("frozen volume image must be B,3,Z,H,W")
            valid = batch.get("geometric_valid")
            valid = (torch.ones(image.shape[0:1] + image.shape[2:], dtype=torch.bool, device=device)
                     if valid is None else torch.as_tensor(valid, device=device, dtype=torch.bool))
            logits = _model_forward(selected, image, valid)
            labels = batch.get("labels")
            batches_seen += 1
            if labels is None:
                continue
            labels = torch.as_tensor(labels, device=device, dtype=torch.long)
            label_valid = batch.get("label_valid")
            label_valid = (labels != UNKNOWN_LABEL if label_valid is None
                           else torch.as_tensor(label_valid, device=device, dtype=torch.bool))
            support = valid & label_valid & (labels != UNKNOWN_LABEL)
            if support.any():
                ce = F.cross_entropy(logits, labels.clamp(0, NUM_CLASSES - 1), reduction="none")
                total_ce += float(ce[support].sum())
                total_voxels += int(support.sum())
            prediction = logits.argmax(1)
            metadata = batch.get("metadata")
            metadata = ([{"patient": f"batch_{batch_index}", "frame": batch_index} for _ in range(image.shape[0])]
                        if metadata is None else metadata)
            if len(metadata) != image.shape[0]:
                raise ValueError("batch metadata must contain one item per volume")
            for index, meta in enumerate(metadata):
                current_support = support[index].cpu().numpy()
                if not current_support.any():
                    continue
                # The same geometric/label support gates both target and
                # prediction.  Leaving an invalid target class in the
                # confusion matrix while abstaining the prediction would
                # manufacture a false negative (and can make a perfect
                # partially-supported frame score below one).
                y = labels[index].cpu().numpy()
                y = np.where(current_support, y, UNKNOWN_LABEL).astype(np.uint8)
                p = prediction[index].cpu().numpy()
                p = np.where(current_support, p, UNKNOWN_LABEL).astype(np.uint8)
                cm = confusion_matrix(y, p); pooled += cm
                frame_rows.append({"patient": str(meta.get("patient", f"batch_{batch_index}")),
                                   "frame": int(meta.get("frame", index)), "raw": class_metrics(cm),
                                   "raw_confusion": cm.tolist(), "supported_voxels": int(current_support.sum())})
    if previous:
        selected.train()
    ids = sorted({row["patient"] for row in frame_rows})
    patients, macro = _patient_metrics(frame_rows, ids)
    return {"schema": "self_audit.volume.frozen_eval.v2", "prediction_model": prediction_model,
            "batches": batches_seen, "valid_voxels": total_voxels,
            "cross_entropy": total_ce / total_voxels if total_voxels else None,
            "patients": patients, "frames": frame_rows, "macro_patient": macro,
            "pooled_confusion": pooled.tolist(), "quality": "FROZEN_DEVELOPMENT_ONLY",
            "physical_boundary_metrics": "NOT_COMPUTED; no spacing-certified HD95 boundary protocol"}


def _build_architecture_only(config: Mapping[str, Any], *, synthetic: bool) -> tuple[nn.Module, dict[str, Any]]:
    """Build the declared architecture without reading a pretrained weight."""
    from .models import build_volume_model
    # The model owner provides the single source-pinned implementation. This
    # explicit export-only flag avoids a process-global monkeypatch and keeps
    # ordinary training construction strict about pretrained assets.
    return build_volume_model(config, synthetic=synthetic, architecture_only=True)


def export_student(model: nn.Module, folder: str | Path, *, config: Mapping[str, Any],
                   source_checkpoint_sha256: str | None = None, prediction_model: str = "student",
                   synthetic: bool | None = None) -> str:
    """Write a compact student-only export with strict provenance metadata."""
    if prediction_model != "student":
        raise ValueError("export_student only exports the student model")
    if not isinstance(config, Mapping):
        raise ValueError("student export requires the complete volume config")
    normalized_config = validate_volume_config(config, allow_full_run=True)
    if synthetic is None:
        data = config.get("data")
        synthetic = bool(data.get("synthetic")) if isinstance(data, Mapping) and "synthetic" in data else False
    folder = Path(folder)
    if folder.exists():
        raise ValueError(f"export destination already exists: {folder}")
    folder.mkdir(parents=True)
    weights_path = folder / "weights.pt"
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({"student": state}, weights_path)
    from .checkpoint import source_sha256
    metadata = {
        "schema": "self_audit.volume.student_export.v2", "config": normalized_config,
        "config_sha256": config_sha256(normalized_config), "prediction_model": "student",
        "synthetic": bool(synthetic), "source_checkpoint_sha256": source_checkpoint_sha256,
        "teacher_included": False, "external_teacher_required": False, "teacher_memory_included": False,
        "pretrained_weights_required_at_load": False,
        "architecture_source": {"backbone": normalized_config["model"]["backbone"],
                                 "source_repo": normalized_config["model"].get("source_repo", ""),
                                 "source_commit": normalized_config["model"].get("source_commit", "")},
        "source_sha256": source_sha256(), "payload_sha256": _sha(weights_path),
    }
    manifest_path = folder / "MANIFEST.json"
    manifest_path.write_text(json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return _sha(manifest_path)


def load_exported_student(folder: str | Path, expected_sha256: str, *,
                          device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load a teacher/cache/weight-independent student export strictly."""
    folder = Path(folder); manifest_path = folder / "MANIFEST.json"
    if _sha(manifest_path) != expected_sha256:
        raise ValueError("student export manifest hash mismatch")
    metadata = _json(manifest_path)
    if metadata.get("schema") != "self_audit.volume.student_export.v2" or metadata.get("teacher_included"):
        raise ValueError("invalid or teacher-bearing student export")
    if metadata.get("external_teacher_required") or metadata.get("teacher_memory_included"):
        raise ValueError("student export requires forbidden teacher/cache state")
    config = metadata.get("config")
    if not isinstance(config, Mapping) or metadata.get("config_sha256") != config_sha256(config):
        raise ValueError("student export config hash mismatch")
    weights_path = folder / "weights.pt"
    if _sha(weights_path) != metadata.get("payload_sha256"):
        raise ValueError("student export payload hash mismatch")
    from .checkpoint import source_sha256
    if metadata.get("source_sha256") != source_sha256():
        raise ValueError("executed volume source differs from student export")
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"student"} or not isinstance(payload["student"], Mapping):
        raise ValueError("student export payload contains forbidden teacher/optimizer state")
    model, _ = _build_architecture_only(config, synthetic=bool(metadata.get("synthetic", False)))
    model.load_state_dict(payload["student"], strict=True)
    return model.to(device).eval(), metadata


def load_checkpoint_student_only(folder: str | Path, expected_sha256: str, *,
                                device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load only a checkpoint's student state for deployment export."""
    from .checkpoint import _file_sha256, _validate_manifest
    folder = Path(folder)
    metadata = _validate_manifest(folder, expected_sha256)
    config = validate_volume_config(metadata["config"], allow_full_run=True)
    if metadata.get("config_sha256") != config_sha256(config):
        raise ValueError("checkpoint config hash mismatch")
    payload_path = folder / "weights.pt"
    if _file_sha256(payload_path) != metadata.get("payload_sha256"):
        raise ValueError("checkpoint payload hash mismatch")
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError) as exc:
        # Legacy v1 checkpoints embed NumPy/Python RNG objects alongside the
        # student tensor. The manifest and payload were hash-verified above;
        # fall back to the trusted local pickle only to extract ``student``.
        # No teacher/optimizer object is constructed or used after loading.
        if "Weights only load failed" not in str(exc):
            raise
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "student" not in payload or not isinstance(payload["student"], Mapping):
        raise ValueError("checkpoint lacks a strict student state")
    synthetic = bool((metadata.get("data_provenance") or {}).get("synthetic", False))
    model, _ = _build_architecture_only(config, synthetic=synthetic)
    model.load_state_dict(payload["student"], strict=True)
    return model.to(device).eval(), metadata


def export_predictions(probabilities: Any, metadata: Mapping[str, Any], out: str | Path) -> dict[str, Any]:
    """Export native-grid probabilities/support before any argmax."""
    native, support = invert_probabilities(probabilities, metadata)
    out = Path(out)
    if out.exists():
        raise ValueError(f"prediction output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    npy = out.with_suffix(".npy"); support_path = out.with_name(out.stem + "_support.npy")
    np.save(npy, native); np.save(support_path, support)
    receipt = {"schema": "self_audit.volume.predictions.v1", "probabilities": str(npy),
               "support": str(support_path), "native_shape_czyx": list(native.shape), "argmax_applied": False}
    out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


__all__ = ["evaluate_frozen", "evaluate_manifest", "export_predictions", "export_student",
           "load_checkpoint_student_only", "load_exported_student", "predict_native_volume"]
