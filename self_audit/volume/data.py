"""Native volume/slab access with explicit geometry and support masks.

This loader is an additive path.  It does not modify :mod:`self_audit.data`
and never enumerates or opens mask paths for unlabeled records.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy import ndimage

from self_audit.model import digest

from .geometry import (
    AXIS_ORDER,
    UNKNOWN_LABEL,
    forbidden_mask_reference,
    load_sidecar,
    sha256_file,
    validate_geometry_record,
    validate_nifti_correspondence,
)


def _as_shape3(value: Any, name: str) -> tuple[int, int, int]:
    v = tuple(int(x) for x in value)
    if len(v) != 3 or any(x < 1 for x in v):
        raise ValueError(f"{name} must contain three positive integers")
    return v  # type: ignore[return-value]


def _crop_shape(crop_size: Any) -> tuple[int, int] | None:
    if crop_size is None or crop_size is False or crop_size == 0:
        return None
    if isinstance(crop_size, int):
        if crop_size < 1:
            raise ValueError("crop_size must be positive")
        return (crop_size, crop_size)
    vals = tuple(int(x) for x in crop_size)
    if len(vals) != 2 or any(x < 1 for x in vals):
        raise ValueError("crop_size must be an int or (height,width)")
    return vals  # type: ignore[return-value]


def _target_shape_from_fov(fov_xyz_mm: Any, spacing_xyz_mm: Any) -> tuple[int, int, int]:
    fov = np.asarray(fov_xyz_mm, dtype=float)
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    if fov.shape != (3,) or spacing.shape != (3,) or np.any(~np.isfinite(fov)) or np.any(fov <= 0) or np.any(spacing <= 0):
        raise ValueError("fov_xyz_mm and spacing_xyz_mm must be positive XYZ triples")
    # Number of voxel centers covering the requested physical extent.
    return tuple(int(max(1, np.floor(fov[i] / spacing[i] + 0.5) + 1)) for i in range(3))


def _resample_zyx(
    volume: np.ndarray,
    source_affine: np.ndarray,
    target_spacing_xyz: tuple[float, float, float],
    *,
    order: int,
    cval: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resample Z,Y,X data while retaining an affine in XYZ convention."""
    source_spacing = np.linalg.norm(source_affine[:3, :3], axis=0)
    direction = source_affine[:3, :3] @ np.diag(1.0 / source_spacing)
    target_spacing = np.asarray(target_spacing_xyz, dtype=float)
    source_shape_xyz = np.asarray(volume.shape[::-1], dtype=float)
    target_shape_xyz = np.maximum(1, np.floor((source_shape_xyz - 1.0) * source_spacing / target_spacing + 0.5).astype(int) + 1)
    target_affine = source_affine.copy()
    target_affine[:3, :3] = direction @ np.diag(target_spacing)
    # scipy uses Z,Y,X arrays, whereas affines use X,Y,Z columns.
    xyz_matrix = np.linalg.inv(source_affine[:3, :3]) @ target_affine[:3, :3]
    xyz_offset = np.linalg.inv(source_affine[:3, :3]) @ (target_affine[:3, 3] - source_affine[:3, 3])
    matrix = xyz_matrix[::-1, ::-1]
    offset = xyz_offset[::-1]
    result = ndimage.affine_transform(
        np.asarray(volume), matrix, offset=offset,
        output_shape=tuple(target_shape_xyz[::-1]), order=order,
        mode="constant", cval=cval, prefilter=(order > 1),
    )
    # Transform a support mask conservatively.  A target voxel is valid only
    # when its center lies within source support; no guessed extrapolation.
    source_support = np.ones(volume.shape, dtype=np.uint8)
    valid = ndimage.affine_transform(source_support, matrix, offset=offset,
                                     output_shape=result.shape, order=0,
                                     mode="constant", cval=0) > 0
    # Mapping native -> current in ZYX coordinates, needed for inversion.
    return result, valid, {
        "current_affine_xyz_to_world_mm": target_affine.tolist(),
        "native_to_current_matrix_zyx": matrix.tolist(),
        "native_to_current_offset_zyx": offset.tolist(),
        "source_shape_zyx": list(volume.shape),
        "current_shape_zyx": list(result.shape),
    }


class VolumeTrainingData:
    """Hash-pinned bridge from the v2 TRAIN manifest to native slabs."""

    def __init__(self, manifest, manifest_sha256, geometry_path, geometry_sha256,
                 synthetic=False, max_cached_arrays=8):
        if type(max_cached_arrays) is not int or max_cached_arrays < 1:
            raise ValueError("max_cached_arrays must be a positive integer")
        self.path = Path(manifest).resolve()
        self.geometry_path = Path(geometry_path).resolve()
        if digest(self.path) != manifest_sha256:
            raise ValueError("training manifest hash mismatch")
        self.manifest_sha256 = str(manifest_sha256)
        self.geometry_sha256 = str(geometry_sha256)
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if m.get("schema") != "self_audit.data.v2" or m.get("role") != "training":
            raise ValueError("not a v2 training manifest")
        if type(m.get("synthetic")) is not bool:
            raise ValueError("training manifest requires an explicit synthetic identity")
        if bool(m.get("synthetic")) != bool(synthetic):
            raise ValueError("synthetic/patient identity mismatch")
        self.geometry = load_sidecar(self.geometry_path, self.geometry_sha256)
        if type(self.geometry.get("synthetic")) is not bool:
            raise ValueError("geometry sidecar requires an explicit synthetic identity")
        if bool(self.geometry["synthetic"]) != bool(m["synthetic"]) or bool(self.geometry["synthetic"]) != bool(synthetic):
            raise ValueError("synthetic manifest/sidecar/caller identity mismatch")
        if self.geometry.get("source_manifest_sha256") not in (None, self.manifest_sha256):
            raise ValueError("geometry sidecar source manifest hash mismatch")
        self.labeled = list(m.get("labeled", [])); self.unlabeled = list(m.get("unlabeled", []))
        if len(self.labeled) < 1 or len(self.unlabeled) < 1:
            raise ValueError("native loader requires labeled and unlabeled TRAIN records")
        ids = [str(r.get("id")) for r in self.labeled + self.unlabeled]
        if any(not x or x == "None" for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("TRAIN patient IDs must be unique")
        if m.get("label_budget") != len(self.labeled):
            raise ValueError("audit labels must be counted in the declared label budget")
        eval_ids = set(m.get("evaluation_ids", []))
        if eval_ids & set(ids):
            raise ValueError("TRAIN/evaluation patient overlap")
        for rec in self.unlabeled:
            if set(rec) != {"id", "image", "image_sha256"}:
                raise ValueError("unlabeled records must contain image only; mask references are forbidden")
        for rec in self.labeled:
            if "masks" not in rec or not rec["masks"]:
                raise ValueError("labeled records require a non-empty masks mapping")
        if self.geometry["patients"].keys() < {r["id"] for r in self.labeled + self.unlabeled}:
            raise ValueError("geometry sidecar is missing a TRAIN patient")
        # Reject all nested U mask references before any source provenance
        # validator, raw NIfTI path, hash or prepared payload is touched.
        for rec in self.unlabeled:
            geo = self.geometry["patients"][rec["id"]]
            forbidden = forbidden_mask_reference(geo)
            if forbidden is not None:
                raise ValueError(f"unlabeled geometry contains forbidden mask reference: {forbidden}")
        # A non-synthetic sidecar is a physical-data receipt, not a shape-only
        # identity.  Require raw image provenance for every role and complete
        # raw mask coverage for every labeled TRAIN frame before sampling.
        if not bool(synthetic):
            for rec in self.labeled + self.unlabeled:
                geo = self.geometry["patients"][rec["id"]]
                source = geo.get("source_image")
                if not isinstance(source, Mapping):
                    raise ValueError(f"real geometry requires source_image provenance for {rec['id']}")
                raw_path = source.get("path", source.get("source_path"))
                raw_sha = source.get("sha256", source.get("source_sha256"))
                if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_sha, str):
                    raise ValueError(f"real geometry source_image requires path and sha256 for {rec['id']}")
                if rec in self.labeled:
                    mask_sources = geo.get("mask_sources")
                    manifest_frames = set(str(k) for k in rec.get("masks", {}))
                    if not isinstance(mask_sources, Mapping) or set(str(k) for k in mask_sources) != manifest_frames:
                        raise ValueError(f"real geometry mask_sources do not cover labeled frames for {rec['id']}")
        # Bind every sidecar record to the hash-pinned manifest before any
        # payload is sampled.  A caller cannot replace a source image hash in
        # the sidecar and defer detection until an optimizer consumes it.
        for rec in self.labeled + self.unlabeled:
            geo = self.geometry["patients"][rec["id"]]
            if geo.get("image_sha256") != rec.get("image_sha256"):
                raise ValueError("geometry sidecar image source hash differs from TRAIN manifest")
        self.max_cached_arrays = max_cached_arrays
        # ``cache`` is an LRU of the currently retained mmap handles.  Eviction
        # only drops this owner reference: arrays returned to callers are not
        # closed, so views remain valid even after their cache entry is evicted.
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.label_reads: set[str] = set()
        self._provenance_checked: set[str] = set()

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.path.parent / p

    def _array(self, path: str | Path, sha: str) -> np.ndarray:
        p = self._resolve(path)
        if digest(p) != sha:
            raise ValueError(f"payload hash mismatch: {p}")
        key = str(p)
        if key in self.cache:
            # A successful hash check is required on every access, including
            # hits, before promoting the handle to most-recently-used.
            self.cache.move_to_end(key)
            return self.cache[key]
        array = np.load(p, mmap_mode="r", allow_pickle=False)
        self.cache[key] = array
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_cached_arrays:
            # Never call close() on an evicted mmap: a sample may still hold a
            # view returned by this method.  Python's normal refcounting keeps
            # that mapping alive until the final caller releases it.
            self.cache.popitem(last=False)
        return array

    def _record_image(self, record: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        image = self._array(record["image"], record["image_sha256"])
        if image.ndim != 4 or any(x < 1 for x in image.shape) or not np.isfinite(image).all():
            raise ValueError("image must be finite T,Z,Y,X")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("image must use the existing normalized [0,1] convention")
        geo = validate_geometry_record(self.geometry["patients"][record["id"]], expected_shape=tuple(image.shape))
        if geo.get("image_sha256") != record.get("image_sha256"):
            raise ValueError("geometry sidecar image source hash differs from TRAIN manifest")
        if geo.get("source_image") is not None and record["id"] not in self._provenance_checked:
            source = geo["source_image"]
            if not isinstance(source, Mapping):
                raise ValueError("source_image physical metadata must be an object")
            raw_path = source.get("path", source.get("source_path"))
            raw_sha = source.get("sha256", source.get("source_sha256"))
            if not isinstance(raw_path, str) or not isinstance(raw_sha, str):
                raise ValueError("source_image requires path and sha256")
            prepared_masks = None
            mask_sources = geo.get("mask_sources")
            if mask_sources:
                if not isinstance(mask_sources, Mapping):
                    raise ValueError("mask_sources physical metadata must be an object")
                prepared_masks = {}
                for frame_key, item in record.get("masks", {}).items():
                    if not isinstance(item, Mapping):
                        raise ValueError("mask record must be an object")
                    prepared_masks[str(frame_key)] = np.asarray(
                        self._array(item["path"], item["sha256"])
                    )
            validate_nifti_correspondence(
                self._resolve(raw_path), image_sha256=raw_sha,
                expected_shape_tzyx=tuple(int(x) for x in image.shape),
                expected_affine_xyz_to_world_mm=geo["affine_xyz_to_world_mm"],
                expected_orientation=str(geo["orientation"]),
                expected_frame_mapping=[int(x) for x in geo["temporal_frame_mapping"]],
                masks=mask_sources if mask_sources else None,
                base_dir=self.path.parent,
                prepared_image_tzyx=image,
                prepared_masks_tzyx=prepared_masks,
            )
            self._provenance_checked.add(record["id"])
        return image, geo

    def _mask(self, record: Mapping[str, Any], t: int, shape: tuple[int, ...]) -> np.ndarray:
        if "masks" not in record or str(t) not in record["masks"]:
            raise ValueError(f"labeled record has no mask for frame {t}")
        item = record["masks"][str(t)]
        mask = np.asarray(self._array(item["path"], item["sha256"]), dtype=np.int64)
        if mask.shape != shape[1:] or not np.isin(mask, [0, 1, 2, 3, UNKNOWN_LABEL]).all():
            raise ValueError("mask shape/ontology mismatch")
        self.label_reads.add(record["id"])
        return mask

    @staticmethod
    def _choose_slab(center: int, depth: int, zdim: int) -> list[int]:
        if depth < 1:
            raise ValueError("slab_depth must be positive")
        start = int(center) - (depth - 1) // 2
        # Preserve the requested slab extent.  Out-of-grid planes are explicit
        # invalid padding (``-1``), never clipped duplicates of an edge slice.
        # This keeps the physical Z grid and valid-voxel accounting honest.
        return [start + i for i in range(depth)]

    @staticmethod
    def _context(image_tzyx: np.ndarray, t: int, z: int) -> np.ndarray:
        zdim = image_tzyx.shape[1]
        idx = [max(0, z - 1), z, min(zdim - 1, z + 1)]
        return np.asarray(image_tzyx[t, idx], dtype=np.float32)

    @staticmethod
    def _allowed_frames(record: Mapping[str, Any], *, labeled: bool, n_frames: int) -> list[int]:
        """Return frame IDs that may be sampled for this supervision role.

        Labeled TRAIN records can be sparse (for example ED/ES only).  Sampling
        from ``range(T)`` and then looking up a mask silently attempts to use
        an unavailable frame.  U records have no mask references and therefore
        retain the complete cine frame schedule.
        """
        if not labeled:
            return list(range(n_frames))
        masks = record.get("masks")
        if not isinstance(masks, Mapping):
            raise ValueError("labeled record requires a frame->mask mapping")
        frames = sorted({int(k) for k in masks})
        if not frames or any(t < 0 or t >= n_frames for t in frames):
            raise ValueError("labeled mask frame IDs must be within the cine")
        return frames

    def _resample_if_requested(self, image_zyx: np.ndarray, labels_zyx: np.ndarray | None, geo: Mapping[str, Any], spacing_xyz_mm: Any):
        source_affine = np.asarray(geo["affine_xyz_to_world_mm"], dtype=float)
        target = tuple(float(x) for x in spacing_xyz_mm) if spacing_xyz_mm is not None else None
        if target is None:
            return image_zyx, labels_zyx, np.ones(image_zyx.shape, bool), source_affine, None
        if len(target) != 3 or any(not np.isfinite(x) or x <= 0 for x in target):
            raise ValueError("spacing_xyz_mm must be a positive XYZ triple")
        image, valid, tr = _resample_zyx(image_zyx, source_affine, target, order=1, cval=0.0)
        labels = None
        if labels_zyx is not None:
            labels = ndimage.affine_transform(labels_zyx, np.asarray(tr["native_to_current_matrix_zyx"]),
                                              offset=np.asarray(tr["native_to_current_offset_zyx"]),
                                              output_shape=image.shape, order=0, mode="constant", cval=UNKNOWN_LABEL)
            labels = labels.astype(np.int64, copy=False)
        return image.astype(np.float32, copy=False), labels, valid, np.asarray(tr["current_affine_xyz_to_world_mm"]), tr

    def _sample_one(self, record: Mapping[str, Any], rng: np.random.Generator, *, t: int | None,
                    center_z: int | None, labeled: bool, slab_depth: int, crop_size: Any,
                    spacing_xyz_mm: Any, fov_xyz_mm: Any, flip: bool,
                    crop_origin: tuple[int, int] | None = None) -> dict[str, Any]:
        image, geo = self._record_image(record)
        allowed_frames = self._allowed_frames(record, labeled=labeled, n_frames=image.shape[0])
        if t is None:
            t = int(rng.choice(allowed_frames))
        if not 0 <= t < image.shape[0] or (labeled and t not in allowed_frames):
            raise ValueError("frame index out of range")
        mask = self._mask(record, t, image.shape) if labeled else None
        if center_z is None:
            center_z = int(rng.integers(image.shape[1]))
        if fov_xyz_mm is not None:
            fov_spacing = tuple(float(x) for x in spacing_xyz_mm) if spacing_xyz_mm is not None else tuple(float(x) for x in geo["spacing_xyz_mm"])
            fshape = _target_shape_from_fov(fov_xyz_mm, fov_spacing)
            # A physical Z FOV is an explicit slab-depth request.  Keeping an
            # explicit larger slab is allowed; a smaller one would violate it.
            slab_depth = max(int(slab_depth), int(fshape[2]))
        current_affine = np.asarray(geo["affine_xyz_to_world_mm"], dtype=float)
        resample_info = None
        current_image = np.asarray(image[t], dtype=np.float32)
        current_mask = mask
        geometric_full = np.ones(current_image.shape, dtype=bool)
        native_z_indices = None
        if spacing_xyz_mm is not None:
            current_image, geometric_full, resample_info = _resample_zyx(
                current_image, current_affine, tuple(float(x) for x in spacing_xyz_mm), order=1, cval=0.0)
            current_affine = np.asarray(resample_info["current_affine_xyz_to_world_mm"], dtype=float)
            if mask is not None:
                current_mask = ndimage.affine_transform(
                    mask, np.asarray(resample_info["native_to_current_matrix_zyx"]),
                    offset=np.asarray(resample_info["native_to_current_offset_zyx"]),
                    output_shape=current_image.shape, order=0, mode="constant", cval=UNKNOWN_LABEL,
                ).astype(np.int64, copy=False)
            native_z_indices = [int(x) for x in self._choose_slab(center_z, slab_depth, image.shape[1])]
            if image.shape[1] > 1:
                center_z = int(round(center_z * (current_image.shape[0] - 1) / (image.shape[1] - 1)))
        z_indices_current = self._choose_slab(center_z, slab_depth, current_image.shape[0])
        # Global-cine context retains the historical edge behavior for a valid
        # center (z=0 -> [0,0,1]); only the newly requested slab planes outside
        # the physical grid are zero-filled and invalid.
        blank_context = np.zeros((3, current_image.shape[1], current_image.shape[2]), dtype=np.float32)
        slab_images = np.stack([
            blank_context if not 0 <= z < current_image.shape[0] else self._context(current_image[None], 0, z)
            for z in z_indices_current
        ], axis=0)
        # Above stack is Z,3,Y,X; transpose to 3,Z,Y,X for model input.
        slab_images = slab_images.transpose(1, 0, 2, 3)
        slab_labels = None
        if current_mask is not None:
            slab_labels = np.stack([
                np.full(current_mask.shape[1:], UNKNOWN_LABEL, dtype=np.int64)
                if not 0 <= z < current_mask.shape[0] else current_mask[z]
                for z in z_indices_current
            ], axis=0)
        geometric_valid = np.stack([
            np.zeros(current_image.shape[1:], dtype=bool)
            if not 0 <= z < geometric_full.shape[0] else geometric_full[z]
            for z in z_indices_current
        ], axis=0)
        # The physical FOV chooses a deterministic XY window when no explicit
        # crop was provided.  Z FOV is represented by the declared slab depth.
        crop_hw = _crop_shape(crop_size)
        if crop_hw is None and fov_xyz_mm is not None:
            current_spacing = np.linalg.norm(current_affine[:3, :3], axis=0)
            fshape = _target_shape_from_fov(fov_xyz_mm, current_spacing)
            crop_hw = (fshape[1], fshape[0])  # Y,X; fov XYZ -> x/y
        crop_info = {"top": 0, "left": 0, "height": slab_images.shape[-2], "width": slab_images.shape[-1],
                     "original_hw": [int(slab_images.shape[-2]), int(slab_images.shape[-1])],
                     "pad_lrtb": [0, 0, 0, 0]}
        # Keep the complete current physical grid in metadata for native
        # export; the emitted slab depth is represented by z_indices.
        grid_shape = tuple(int(x) for x in current_image.shape)
        if crop_hw is not None:
            h, w = slab_images.shape[-2:]; ch, cw = crop_hw
            ph, pw = max(0, ch - h), max(0, cw - w)
            pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
            if ph or pw:
                slab_images = np.pad(slab_images, ((0, 0), (0, 0), (pad[2], pad[3]), (pad[0], pad[1])), mode="edge")
                geometric_valid = np.pad(geometric_valid, ((0, 0), (pad[2], pad[3]), (pad[0], pad[1])), constant_values=False)
                if slab_labels is not None:
                    slab_labels = np.pad(slab_labels, ((0, 0), (pad[2], pad[3]), (pad[0], pad[1])), constant_values=UNKNOWN_LABEL)
                h, w = slab_images.shape[-2:]
            if crop_origin is None:
                top = int(rng.integers(h - ch + 1)); left = int(rng.integers(w - cw + 1))
            else:
                top, left = (int(crop_origin[0]), int(crop_origin[1]))
                if not (0 <= top <= h - ch and 0 <= left <= w - cw):
                    raise ValueError("crop_origin outside available grid")
            slab_images = slab_images[..., top:top + ch, left:left + cw]
            geometric_valid = geometric_valid[..., top:top + ch, left:left + cw]
            if slab_labels is not None:
                slab_labels = slab_labels[..., top:top + ch, left:left + cw]
            crop_info = {"top": top, "left": left, "height": ch, "width": cw,
                         "original_hw": [int(h), int(w)], "pad_lrtb": list(pad)}
        pre_crop_shape = grid_shape
        if flip:
            slab_images = slab_images[..., ::-1].copy(); geometric_valid = geometric_valid[..., ::-1].copy()
            if slab_labels is not None:
                slab_labels = slab_labels[..., ::-1].copy()
        if slab_labels is None:
            # Keep a typed support tensor for U examples.  It is all false and
            # never acts as an implicit mask/label source.
            label_valid = np.zeros_like(geometric_valid, dtype=bool)
        else:
            label_valid = geometric_valid & (slab_labels != UNKNOWN_LABEL)
        transform = {
            "z_indices": z_indices_current,
            "crop": crop_info,
            "flip": bool(flip),
            "native_shape_zyx": list(image.shape[1:]),
            # current_shape_zyx is the uncropped resampled slab grid used by
            # invert_probabilities; sample_shape_zyx is the emitted crop.
            "current_shape_zyx": list(pre_crop_shape),
            # Persist emitted support so native inversion can require the full
            # interpolation stencil.  It is metadata, not a label source.
            "geometric_valid": geometric_valid.tolist(),
        }
        if resample_info:
            transform["resample"] = resample_info
            transform["native_z_indices"] = native_z_indices
        meta = {
            "patient": record["id"], "frame": int(t), "z_indices": z_indices_current,
            "native_shape_zyx": list(image.shape[1:]), "current_shape_zyx": list(pre_crop_shape),
            "sample_shape_zyx": list(slab_images.shape[1:]),
            "native_affine_xyz_to_world_mm": geo["affine_xyz_to_world_mm"],
            "current_affine_xyz_to_world_mm": current_affine.tolist(),
            "spacing_xyz_mm": np.linalg.norm(current_affine[:3, :3], axis=0).tolist(),
            "axis_order": AXIS_ORDER, "reversible_transform": transform,
            "source_image_sha256": record["image_sha256"],
        }
        return {"image": torch.from_numpy(np.ascontiguousarray(slab_images, dtype=np.float32)),
                "labels": None if slab_labels is None else torch.from_numpy(np.ascontiguousarray(slab_labels, dtype=np.int64)),
                "geometric_valid": torch.from_numpy(np.ascontiguousarray(geometric_valid, dtype=bool)),
                "label_valid": torch.from_numpy(np.ascontiguousarray(label_valid, dtype=bool)),
                "metadata": meta}

    def sample_batch(self, index, rng, batch_size, *, labeled, slab_depth, crop_size,
                     spacing_xyz_mm=None, fov_xyz_mm=None, flip=False):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(slab_depth) is not int or slab_depth < 1:
            raise ValueError("slab_depth must be a positive integer")
        records = self.labeled if labeled else self.unlabeled
        if not 0 <= int(index) < len(records):
            raise IndexError("patient index out of range")
        items = [self._sample_one(records[index], rng, t=None, center_z=None, labeled=labeled,
                                  slab_depth=slab_depth, crop_size=crop_size,
                                  spacing_xyz_mm=spacing_xyz_mm, fov_xyz_mm=fov_xyz_mm, flip=flip)
                 for _ in range(batch_size)]
        result = {
            "image": torch.stack([x["image"] for x in items]),
            "geometric_valid": torch.stack([x["geometric_valid"] for x in items]),
            "label_valid": torch.stack([x["label_valid"] for x in items]),
            "labels": None if not labeled else torch.stack([x["labels"] for x in items]),
            "metadata": [x["metadata"] for x in items],
        }
        return result

    def sample_temporal_pair(self, index, rng, *, labeled=False, slab_depth=1, crop_size=0,
                             spacing_xyz_mm=None, fov_xyz_mm=None, flip=False):
        records = self.labeled if labeled else self.unlabeled
        if not 0 <= int(index) < len(records):
            raise IndexError("patient index out of range")
        image, _ = self._record_image(records[index])
        if image.shape[0] < 2:
            raise ValueError("temporal pair requires at least two adjacent frames")
        allowed = self._allowed_frames(records[index], labeled=labeled, n_frames=image.shape[0])
        pairs = [t for t in allowed if t + 1 in set(allowed)] if labeled else list(range(image.shape[0] - 1))
        if not pairs:
            raise ValueError("temporal pair requires two adjacent labeled frames")
        t = int(rng.choice(pairs)); z = int(rng.integers(image.shape[1]))
        a = self._sample_one(records[index], rng, t=t, center_z=z, labeled=labeled,
                             slab_depth=slab_depth, crop_size=crop_size, spacing_xyz_mm=spacing_xyz_mm,
                             fov_xyz_mm=fov_xyz_mm, flip=flip)
        a_crop = a["metadata"]["reversible_transform"]["crop"]
        b = self._sample_one(records[index], rng, t=t + 1, center_z=z, labeled=labeled,
                             slab_depth=slab_depth, crop_size=crop_size, spacing_xyz_mm=spacing_xyz_mm,
                             fov_xyz_mm=fov_xyz_mm, flip=flip,
                             crop_origin=(int(a_crop["top"]), int(a_crop["left"])))
        return {
            "image": a["image"], "adjacent_image": b["image"],
            "geometric_valid": a["geometric_valid"], "adjacent_geometric_valid": b["geometric_valid"],
            "labels": a["labels"], "adjacent_labels": b["labels"],
            "label_valid": a["label_valid"], "adjacent_label_valid": b["label_valid"],
            "metadata": a["metadata"], "adjacent_metadata": b["metadata"],
        }


__all__ = ["VolumeTrainingData"]
