"""Deterministic, physically-aware region learning from unlabeled volumes (G7).

This module implements deterministic physical supervoxels/regions and region
consistency objectives for semi-supervised native-volume learning.

Key scientific contracts:
1. Deterministic and physically aware: seed placement and spatial distance
   respect anisotropic voxel spacing (mm) from geometry metadata. Missing physical
   metadata fails closed (no silent isotropic 1mm default).
2. Unlabeled-safety: strictly NO U/VAL/TEST masks are accessed. Boundaries are
   derived purely from image gradients and detached teacher predictions.
3. Detached targets: region assignments, boundary masks, and teacher probability
   targets are strictly detached; gradients flow ONLY through current student logits.
4. Cadence & caching: immutable sample-identity cache keyed by full image bytes,
   full support, config, and loader metadata. Separate frozen teacher version per
   refresh interval; cadence is preserved without using new teacher hashes to
   invalidate every step. Cache is bounded with strict hash validation and exact resume.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RegionConfig:
    """Configuration for physical region learning."""

    mode: str = "none"  # "none" | "image" | "ema" (aliases: "image_only", "ema_derived", "off")
    weight: float = 0.1
    refresh_every: int = 1
    # Physical and supervoxel clustering parameters
    spatial_weight: float = 1.0  # balance between intensity/features and physical distance
    target_num_regions: int = 32  # approximate target supervoxels per volume/slab
    min_region_size: int = 8  # minimum voxels per supervoxel
    iterations: int = 4  # deterministic clustering iterations
    # Boundary downweighting / rejection
    boundary_threshold: float = 0.2  # normalized gradient threshold for boundary detection
    boundary_downweight: float = 0.1  # relative weight for boundary voxels
    boundary_rejection: bool = False  # if True, completely zero out boundary voxels
    # Objective
    loss_type: str = "mse"  # "mse" | "kl"
    # Host cache bounds
    max_cache_size: int = 64

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> RegionConfig:
        if cfg is None:
            return cls()
        if isinstance(cfg, RegionConfig):
            return copy.deepcopy(cfg)
        d = dict(cfg)
        # Normalize mode aliases
        mode = str(d.get("mode", "none")).lower()
        if mode in {"image_only", "r01", "image_supervoxels"}:
            mode = "image"
        elif mode in {"ema_derived", "teacher_derived", "r02"}:
            mode = "ema"
        elif mode in {"off", "r00"}:
            mode = "none"
        d["mode"] = mode
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class RegionTargets:
    """Serializable, detached region targets produced for a volume batch."""

    mode: str
    region_labels: Tensor  # (B, Z, H, W) int64; -1 for invalid or unassigned voxels
    region_weights: Tensor  # (B, Z, H, W) float32 in [0, 1]; incorporates boundary downweighting
    target_probabilities: Tensor | None  # (B, C, Z, H, W) float32, detached, or None
    cache_hash: str
    step: int
    num_regions: list[int]
    metrics: dict[str, float] = field(default_factory=dict)


def _normalize_metadata_item(m: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(m, Mapping):
        return {}
    patient = str(m.get("patient") or m.get("patient_id") or "")
    frame = str(
        m.get("frame")
        if m.get("frame") is not None
        else (m.get("frame_index") if m.get("frame_index") is not None else "")
    )
    z_indices = list(m.get("z_indices", [])) if m.get("z_indices") is not None else []
    rev_tr = m.get("reversible_transform") or {}
    crop = m.get("crop") or rev_tr.get("crop") or {}
    source = str(m.get("source_image_sha256") or m.get("source") or "")
    spacing = list(m.get("spacing_xyz_mm", [])) if m.get("spacing_xyz_mm") is not None else []
    affine = m.get("current_affine_xyz_to_world_mm") or m.get("native_affine_xyz_to_world_mm") or []

    return {
        "patient": patient,
        "frame": frame,
        "z_indices": z_indices,
        "crop": crop,
        "source": source,
        "spacing": spacing,
        "affine": affine,
        "transform": rev_tr,
    }


def compute_sample_identity_hash(
    image: Tensor,
    geometric_valid: Tensor,
    metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> str:
    """Compute an immutable sample-identity cryptographic hash.

    For single volume items (B=1 or unbatched), hashes the individual sample.
    For batches (B>1), hashes individual samples and combines their identities.
    """
    if image.ndim == 5 and image.shape[0] > 1:
        b = image.shape[0]
        meta_list = metadata if isinstance(metadata, Sequence) else ([metadata] * b if metadata else [None] * b)
        hashes = [
            compute_sample_identity_hash(
                image[i : i + 1],
                geometric_valid[i : i + 1],
                meta_list[i] if i < len(meta_list) else None,
            )
            for i in range(b)
        ]
        return hashlib.sha256("::".join(hashes).encode()).hexdigest()

    h = hashlib.sha256()

    img_t = image.squeeze(0) if (image.ndim == 5 and image.shape[0] == 1) else image
    val_t = (
        geometric_valid.squeeze(0)
        if (geometric_valid.ndim == 4 and geometric_valid.shape[0] == 1)
        else geometric_valid
    )

    # 1. Full image bytes (no sparse skipping: detects any interior voxel change)
    img_arr = np.ascontiguousarray(img_t.detach().cpu().numpy())
    h.update(str(img_arr.shape).encode())
    h.update(str(img_arr.dtype).encode())
    h.update(img_arr.tobytes())

    # 2. Full geometric validity support bytes
    val_arr = np.ascontiguousarray(val_t.detach().cpu().numpy().astype(np.uint8))
    h.update(str(val_arr.shape).encode())
    h.update(val_arr.tobytes())

    # 3. Normalized loader metadata
    if metadata:
        meta_items = []
        raw_list = [metadata] if isinstance(metadata, Mapping) else metadata
        for m in raw_list:
            if isinstance(m, Mapping):
                meta_items.append(_normalize_metadata_item(m))
        h.update(json.dumps(meta_items, sort_keys=True).encode())
    else:
        h.update(b"no_metadata")

    return h.hexdigest()


def compute_cache_hash(
    image: Tensor,
    valid_mask: Tensor,
    metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    step: int,
    mode: str,
    *,
    config: RegionConfig | None = None,
    refresh_every: int = 1,
) -> str:
    """Compute a deterministic hash for region target caching for a sample."""
    sample_id = compute_sample_identity_hash(image, valid_mask, metadata)
    interval = (step // refresh_every) if refresh_every > 0 else step
    cfg_parts = [f"mode={mode}", f"interval={interval}"]
    if config is not None:
        cfg_parts.extend(
            [
                f"k={config.target_num_regions}",
                f"sw={config.spatial_weight}",
                f"min={config.min_region_size}",
                f"it={config.iterations}",
                f"bt={config.boundary_threshold}",
                f"bd={config.boundary_downweight}",
                f"br={int(config.boundary_rejection)}",
                f"lt={config.loss_type}",
            ]
        )
    cfg_str = ";".join(cfg_parts)
    return hashlib.sha256(f"{sample_id}::{cfg_str}".encode()).hexdigest()


def compute_physical_gradient(
    image_zyx: np.ndarray,
    spacing_zyx_mm: np.ndarray,
) -> np.ndarray:
    """Compute normalized 3D spatial gradient magnitude scaled by physical spacing.

    Gradient is computed in physical units: dI / (dz * s_z), etc.
    """
    z, y, x = image_zyx.shape
    gz = np.zeros_like(image_zyx, dtype=np.float32)
    gy = np.zeros_like(image_zyx, dtype=np.float32)
    gx = np.zeros_like(image_zyx, dtype=np.float32)

    sz, sy, sx = float(spacing_zyx_mm[0]), float(spacing_zyx_mm[1]), float(spacing_zyx_mm[2])

    if z > 1:
        gz[1:-1] = (image_zyx[2:] - image_zyx[:-2]) / (2.0 * sz)
        gz[0] = (image_zyx[1] - image_zyx[0]) / sz
        gz[-1] = (image_zyx[-1] - image_zyx[-2]) / sz
    if y > 1:
        gy[:, 1:-1] = (image_zyx[:, 2:] - image_zyx[:, :-2]) / (2.0 * sy)
        gy[:, 0] = (image_zyx[:, 1] - image_zyx[:, 0]) / sy
        gy[:, -1] = (image_zyx[:, -1] - image_zyx[:, -2]) / sy
    if x > 1:
        gx[:, :, 1:-1] = (image_zyx[:, :, 2:] - image_zyx[:, :, :-2]) / (2.0 * sx)
        gx[:, :, 0] = (image_zyx[:, :, 1] - image_zyx[:, :, 0]) / sx
        gx[:, :, -1] = (image_zyx[:, :, -1] - image_zyx[:, :, -2]) / sx

    grad_mag = np.sqrt(gz * gz + gy * gy + gx * gx)
    p95 = float(np.percentile(grad_mag, 95)) if grad_mag.size > 0 else 1.0
    scale = p95 if p95 > 1e-6 else 1.0
    return grad_mag / scale


def generate_deterministic_supervoxels_3d(
    image_zyx: np.ndarray,
    valid_zyx: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    target_num_regions: int = 32,
    spatial_weight: float = 1.0,
    min_region_size: int = 8,
    iterations: int = 4,
    features_czyx: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Generate deterministic, physically-aware 3D supervoxels on valid voxels.

    Returns:
        labels: int32 array (Z, Y, X) with region IDs 0..K-1, or -1 for invalid.
        num_regions: total number of distinct supervoxels.
    """
    z_dim, y_dim, x_dim = image_zyx.shape
    valid_mask = valid_zyx.astype(bool)
    n_valid = int(np.sum(valid_mask))
    if n_valid < min_region_size:
        labels = np.full((z_dim, y_dim, x_dim), -1, dtype=np.int32)
        if n_valid > 0:
            labels[valid_mask] = 0
            return labels, 1
        return labels, 0

    sz, sy, sx = float(spacing_zyx_mm[0]), float(spacing_zyx_mm[1]), float(spacing_zyx_mm[2])
    voxel_vol_mm3 = sz * sy * sx
    total_vol_mm3 = n_valid * voxel_vol_mm3

    k = max(1, min(target_num_regions, n_valid // min_region_size))
    target_voxel_vol = total_vol_mm3 / float(k)
    r_phys = float(target_voxel_vol ** (1.0 / 3.0))

    # Grid step in voxels per axis
    step_z = max(1, int(round(r_phys / sz)))
    step_y = max(1, int(round(r_phys / sy)))
    step_x = max(1, int(round(r_phys / sx)))

    # Deterministic regular lattice seed initialization
    z_coords = np.arange(step_z // 2, z_dim, step_z)
    y_coords = np.arange(step_y // 2, y_dim, step_y)
    x_coords = np.arange(step_x // 2, x_dim, step_x)
    if len(z_coords) == 0:
        z_coords = np.array([z_dim // 2])
    if len(y_coords) == 0:
        y_coords = np.array([y_dim // 2])
    if len(x_coords) == 0:
        x_coords = np.array([x_dim // 2])

    seed_coords = []
    for cz in z_coords:
        for cy in y_coords:
            for cx in x_coords:
                if valid_mask[cz, cy, cx]:
                    seed_coords.append((cz, cy, cx))

    if not seed_coords:
        # Fallback to deterministic valid points
        valid_indices = np.argwhere(valid_mask)
        stride = max(1, len(valid_indices) // k)
        seed_coords = [tuple(p) for p in valid_indices[::stride][:k]]

    if not seed_coords:
        labels = np.full((z_dim, y_dim, x_dim), -1, dtype=np.int32)
        return labels, 0

    # Perturb seeds deterministically to local minimum gradient to avoid placing on edges
    grad = compute_physical_gradient(image_zyx, spacing_zyx_mm)
    refined_seeds = []
    for cz, cy, cx in seed_coords:
        min_g = grad[cz, cy, cx]
        best_p = (cz, cy, cx)
        for dz in (-1, 0, 1):
            nz = cz + dz
            if not (0 <= nz < z_dim):
                continue
            for dy in (-1, 0, 1):
                ny = cy + dy
                if not (0 <= ny < y_dim):
                    continue
                for dx in (-1, 0, 1):
                    nx = cx + dx
                    if not (0 <= nx < x_dim):
                        continue
                    if valid_mask[nz, ny, nx] and grad[nz, ny, nx] < min_g:
                        min_g = grad[nz, ny, nx]
                        best_p = (nz, ny, nx)
        refined_seeds.append(best_p)

    # Initialize cluster centers: [z_mm, y_mm, x_mm, intensity, feat0, ...]
    seeds = np.array(refined_seeds, dtype=np.int32)
    n_seeds = len(seeds)
    has_features = features_czyx is not None
    n_feats = features_czyx.shape[0] if has_features else 0

    centers_pos_mm = np.zeros((n_seeds, 3), dtype=np.float32)
    centers_int = np.zeros(n_seeds, dtype=np.float32)
    centers_feat = np.zeros((n_seeds, n_feats), dtype=np.float32) if has_features else None

    for i, (cz, cy, cx) in enumerate(seeds):
        centers_pos_mm[i] = [cz * sz, cy * sy, cx * sx]
        centers_int[i] = image_zyx[cz, cy, cx]
        if has_features and centers_feat is not None:
            centers_feat[i] = features_czyx[:, cz, cy, cx]

    # Precompute physical coordinates of all valid voxels
    valid_idx = np.argwhere(valid_mask)
    v_z, v_y, v_x = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
    v_pos_mm = np.stack([v_z * sz, v_y * sy, v_x * sx], axis=-1).astype(np.float32)
    v_int = image_zyx[v_z, v_y, v_x].astype(np.float32)
    v_feat = features_czyx[:, v_z, v_y, v_x].T.astype(np.float32) if has_features else None

    assignments = np.zeros(len(valid_idx), dtype=np.int32)
    s_phys = max(r_phys, 1e-4)

    # Deterministic Lloyd / SLIC iterations
    for it in range(iterations):
        block_size = 4096
        for start_idx in range(0, len(valid_idx), block_size):
            end_idx = min(start_idx + block_size, len(valid_idx))
            pos_chunk = v_pos_mm[start_idx:end_idx]  # (B_sub, 3)
            int_chunk = v_int[start_idx:end_idx]  # (B_sub,)

            # Physical spatial distance: (B_sub, n_seeds)
            d_pos2 = np.sum((pos_chunk[:, None, :] - centers_pos_mm[None, :, :]) ** 2, axis=-1)
            # Intensity distance: (B_sub, n_seeds)
            d_int2 = (int_chunk[:, None] - centers_int[None, :]) ** 2

            dist2 = d_int2 + (spatial_weight / (s_phys * s_phys)) * d_pos2

            if has_features and v_feat is not None and centers_feat is not None:
                feat_chunk = v_feat[start_idx:end_idx]  # (B_sub, C)
                d_feat2 = np.mean((feat_chunk[:, None, :] - centers_feat[None, :, :]) ** 2, axis=-1)
                dist2 += d_feat2

            assignments[start_idx:end_idx] = np.argmin(dist2, axis=1)

        # Update cluster centers deterministically
        for c in range(n_seeds):
            c_mask = assignments == c
            if np.any(c_mask):
                centers_pos_mm[c] = np.mean(v_pos_mm[c_mask], axis=0)
                centers_int[c] = float(np.mean(v_int[c_mask]))
                if has_features and centers_feat is not None and v_feat is not None:
                    centers_feat[c] = np.mean(v_feat[c_mask], axis=0)

    # Re-index clusters to eliminate empty ones
    unique_clusters, inv = np.unique(assignments, return_inverse=True)
    assignments = inv.astype(np.int32)
    num_clusters = len(unique_clusters)

    # Construct full label volume (-1 for invalid)
    labels = np.full((z_dim, y_dim, x_dim), -1, dtype=np.int32)
    labels[v_z, v_y, v_x] = assignments

    return labels, num_clusters


def compute_boundary_weights(
    image_zyx: np.ndarray,
    valid_zyx: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    probabilities_czyx: np.ndarray | None,
    config: RegionConfig,
) -> np.ndarray:
    """Compute per-voxel confidence/boundary weight in [0, 1].

    Boundary voxels crossing physical image gradients or confident teacher class
    transitions receive downweighted weights (or zero if boundary_rejection is True).
    """
    weights = np.zeros_like(image_zyx, dtype=np.float32)
    valid_mask = valid_zyx.astype(bool)
    weights[valid_mask] = 1.0

    # 1. Image physical gradient boundary
    grad = compute_physical_gradient(image_zyx, spacing_zyx_mm)
    is_boundary = (grad > config.boundary_threshold) & valid_mask

    # 2. Confident teacher class boundary (prediction-only, NO GT masks read)
    if probabilities_czyx is not None:
        p_max = np.max(probabilities_czyx, axis=0)
        c_argmax = np.argmax(probabilities_czyx, axis=0)
        z_dim, y_dim, x_dim = image_zyx.shape

        class_boundary = np.zeros_like(valid_mask, dtype=bool)
        if z_dim > 1:
            diff_z = (c_argmax[1:] != c_argmax[:-1]) & (p_max[1:] > 0.6) & (p_max[:-1] > 0.6)
            class_boundary[1:] |= diff_z
            class_boundary[:-1] |= diff_z
        if y_dim > 1:
            diff_y = (c_argmax[:, 1:] != c_argmax[:, :-1]) & (p_max[:, 1:] > 0.6) & (p_max[:, :-1] > 0.6)
            class_boundary[:, 1:] |= diff_y
            class_boundary[:, :-1] |= diff_y
        if x_dim > 1:
            diff_x = (c_argmax[:, :, 1:] != c_argmax[:, :, :-1]) & (p_max[:, :, 1:] > 0.6) & (p_max[:, :, :-1] > 0.6)
            class_boundary[:, :, 1:] |= diff_x
            class_boundary[:, :, :-1] |= diff_x

        is_boundary |= (class_boundary & valid_mask)

    downweight_val = 0.0 if config.boundary_rejection else float(config.boundary_downweight)
    weights[is_boundary] = downweight_val
    return weights


def split_or_reject_cross_class_regions(
    labels: np.ndarray,
    weights: np.ndarray,
    probabilities_czyx: np.ndarray | None,
    valid_mask: np.ndarray,
    config: RegionConfig,
    confidence_threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split supervoxels by confident teacher class boundaries and reject cross-class regions.

    Enforces that no single region pools interiors of conflicting confident classes.
    For any initial supervoxel spanning multiple confident classes (p_max >= confidence_threshold):
    1. Splits the supervoxel into distinct subregions corresponding to each confident class.
    2. Subregions with fewer voxels than min_region_size are rejected (weight=0, label=-1).
    3. If a region cannot be split or still crosses confident classes:
       - If boundary_rejection is True, the entire region is rejected (weight=0, label=-1).
       - If boundary_rejection is False, the entire region is downweighted.
    4. Immediate boundary voxels remain downweighted or zeroed as computed by compute_boundary_weights.
    """
    if probabilities_czyx is None or config.mode != "ema":
        unique_lbls = [lbl for lbl in np.unique(labels) if lbl >= 0]
        return labels, weights, len(unique_lbls)

    p_max = np.max(probabilities_czyx, axis=0)  # (Z, Y, X)
    c_argmax = np.argmax(probabilities_czyx, axis=0)  # (Z, Y, X)

    new_labels = np.full_like(labels, -1)
    new_weights = weights.copy()
    next_label_id = 0

    unique_initial_labels = [lbl for lbl in np.unique(labels) if lbl >= 0]

    for r_id in unique_initial_labels:
        r_mask = (labels == r_id) & valid_mask
        if not np.any(r_mask):
            continue

        conf_mask = r_mask & (p_max >= confidence_threshold)
        conf_classes = np.unique(c_argmax[conf_mask])

        if len(conf_classes) <= 1:
            # Region does not cross multiple confident classes
            new_labels[r_mask] = next_label_id
            next_label_id += 1
        else:
            # Region spans multiple confident classes: split by class argmax
            created_subregions = []
            for c in conf_classes:
                sub_mask = r_mask & (c_argmax == c)
                n_c = int(np.sum(sub_mask))
                if n_c >= config.min_region_size:
                    created_subregions.append(sub_mask)

            if created_subregions:
                # Successfully partitioned into homogeneous class subregions
                for sub_mask in created_subregions:
                    new_labels[sub_mask] = next_label_id
                    next_label_id += 1
                # Any residual voxels in r_mask not assigned (e.g. tiny subregions < min_region_size)
                unassigned = r_mask & (new_labels == -1)
                if np.any(unassigned):
                    new_weights[unassigned] = 0.0
            else:
                # Cannot split: entire region crosses confident classes
                if config.boundary_rejection:
                    new_weights[r_mask] = 0.0
                    new_labels[r_mask] = -1
                else:
                    new_weights[r_mask] = np.minimum(new_weights[r_mask], float(config.boundary_downweight))
                    new_labels[r_mask] = next_label_id
                    next_label_id += 1

    return new_labels, new_weights, next_label_id


def region_consistency_loss(
    student_logits: Tensor,
    targets: RegionTargets,
    geometric_valid: Tensor | None = None,
    *,
    loss_type: str = "mse",
    weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compute differentiable region consistency loss on student logits.

    The loss penalizes intra-region prediction variance or discrepancy against
    detached region targets. Gradients flow strictly into student_logits.
    """
    device = student_logits.device
    zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
    metrics: dict[str, float] = {
        "region_loss": 0.0,
        "active_regions": 0.0,
        "valid_region_voxels": 0.0,
    }

    if targets.mode in {"none", "off"} or weight <= 0.0:
        return zero_loss, metrics

    b, c, z, h, w = student_logits.shape
    p_student = F.softmax(student_logits, dim=1)

    labels = targets.region_labels.to(device)  # (B, Z, H, W)
    weights = targets.region_weights.to(device)  # (B, Z, H, W)

    if geometric_valid is not None:
        valid_t = geometric_valid.to(device=device, dtype=torch.bool)
        weights = weights * valid_t.float()

    total_loss = torch.tensor(0.0, device=device)
    total_active_regions = 0
    total_valid_voxels = 0

    has_target_probs = targets.target_probabilities is not None
    target_probs = targets.target_probabilities.to(device).detach() if has_target_probs else None

    for bi in range(b):
        lbl = labels[bi]  # (Z, H, W)
        wgt = weights[bi]  # (Z, H, W)
        p_sub = p_student[bi]  # (C, Z, H, W)

        valid_mask = (lbl >= 0) & (wgt > 0.0)
        if not valid_mask.any():
            continue

        unique_lbls = torch.unique(lbl[valid_mask])
        batch_loss = torch.tensor(0.0, device=device)
        batch_weight_sum = torch.tensor(0.0, device=device)

        for r_id in unique_lbls:
            r_mask = (lbl == r_id) & valid_mask
            r_w = wgt[r_mask]  # (N_r,)
            r_w_sum = r_w.sum()
            if r_w_sum <= 1e-6:
                continue

            r_p = p_sub[:, r_mask]  # (C, N_r)

            if target_probs is not None:
                # Target is pooled teacher probabilities in the region
                t_sub = target_probs[bi][:, r_mask]  # (C, N_r)
                p_target = (t_sub * r_w[None, :]).sum(dim=-1, keepdim=True) / r_w_sum
            else:
                # Target is pooled student prediction in the region (detached counterfactual mean)
                p_target = ((r_p * r_w[None, :]).sum(dim=-1, keepdim=True) / r_w_sum).detach()

            if loss_type == "kl":
                # KL divergence: target * (log target - log pred)
                kl = p_target * (torch.log(p_target.clamp(min=1e-7)) - torch.log(r_p.clamp(min=1e-7)))
                err = kl.sum(dim=0)  # (N_r,)
            else:
                # Mean squared probability error
                diff = r_p - p_target
                err = (diff * diff).sum(dim=0)  # (N_r,)

            reg_loss = (err * r_w).sum()
            batch_loss = batch_loss + reg_loss
            batch_weight_sum = batch_weight_sum + r_w_sum
            total_active_regions += 1
            total_valid_voxels += int((r_w > 0.0).sum().item())

        if batch_weight_sum > 0:
            total_loss = total_loss + (batch_loss / batch_weight_sum)

    if b > 0:
        total_loss = (total_loss / float(b)) * float(weight)

    metrics["region_loss"] = float(total_loss.detach())
    metrics["active_regions"] = float(total_active_regions)
    metrics["valid_region_voxels"] = float(total_valid_voxels)

    return total_loss, metrics


class RegionManager:
    """Manages physical supervoxel target generation, caching, and consistency loss.

    Adheres strictly to the coordinator contract:
    - `.targets(image, geometric_valid, metadata, *, probabilities=None, features=None, step=0)`
    - `.loss(student_logits, targets, geometric_valid=None)`
    - `.state_dict()` and `.load_state_dict()`
    """

    def __init__(
        self,
        config: RegionConfig | Mapping[str, Any] | None = None,
        max_cache_size: int = 64,
    ) -> None:
        self.config = RegionConfig.from_mapping(config)
        cfg_max = getattr(self.config, "max_cache_size", None)
        self.max_cache_size = int(cfg_max) if cfg_max is not None and int(cfg_max) > 0 else max(1, int(max_cache_size))
        self.config.max_cache_size = self.max_cache_size
        # Bounded LRU cache: cache_hash -> RegionTargets
        self._cache: collections.OrderedDict[str, RegionTargets] = collections.OrderedDict()
        self._cached_targets: RegionTargets | None = None
        self._cache_hash: str = ""
        self._last_step: int = -1
        self._refresh_count: int = 0

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def weight(self) -> float:
        return self.config.weight

    @property
    def refresh_every(self) -> int:
        return self.config.refresh_every

    def reset_cache(self) -> None:
        self._cache.clear()
        self._cached_targets = None
        self._cache_hash = ""
        self._last_step = -1

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state dict for deterministic checkpoint resume."""
        serialized_entries = []
        for k, t in self._cache.items():
            serialized_entries.append(
                {
                    "key": k,
                    "mode": t.mode,
                    "region_labels": t.region_labels.cpu(),
                    "region_weights": t.region_weights.cpu(),
                    "target_probabilities": (
                        t.target_probabilities.cpu() if t.target_probabilities is not None else None
                    ),
                    "cache_hash": t.cache_hash,
                    "step": t.step,
                    "num_regions": t.num_regions,
                    "metrics": dict(t.metrics),
                }
            )

        targets_dict = None
        if self._cached_targets is not None:
            targets_dict = {
                "mode": self._cached_targets.mode,
                "region_labels": self._cached_targets.region_labels.cpu(),
                "region_weights": self._cached_targets.region_weights.cpu(),
                "target_probabilities": (
                    self._cached_targets.target_probabilities.cpu()
                    if self._cached_targets.target_probabilities is not None
                    else None
                ),
                "cache_hash": self._cached_targets.cache_hash,
                "step": self._cached_targets.step,
                "num_regions": self._cached_targets.num_regions,
                "metrics": dict(self._cached_targets.metrics),
            }
        return {
            "config": asdict(self.config),
            "max_cache_size": self.max_cache_size,
            "last_step": self._last_step,
            "refresh_count": self._refresh_count,
            "cache_hash": self._cache_hash,
            "cached_targets": targets_dict,
            "cache_entries": serialized_entries,
        }

    def _deserialize_target(self, cached: Mapping[str, Any]) -> RegionTargets:
        labels = torch.as_tensor(cached["region_labels"], dtype=torch.long)
        weights = torch.as_tensor(cached["region_weights"], dtype=torch.float32)
        target_probs = (
            torch.as_tensor(cached["target_probabilities"], dtype=torch.float32)
            if cached.get("target_probabilities") is not None
            else None
        )
        h = str(cached.get("cache_hash", ""))

        if labels.ndim != 4:
            raise ValueError(f"Corrupted cache target: region_labels must have 4 dimensions, got {labels.ndim}")
        if weights.shape != labels.shape:
            raise ValueError("Corrupted cache target: region_weights shape mismatch with labels")
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("Corrupted cache target: non-finite values in region_weights")
        if (weights < 0.0).any() or (weights > 1.0).any():
            raise ValueError("Corrupted cache target: weights outside [0, 1]")
        if h and len(h) != 64:
            raise ValueError(f"Corrupted cache target: invalid hash length {len(h)}")

        return RegionTargets(
            mode=str(cached.get("mode", self.config.mode)),
            region_labels=labels,
            region_weights=weights,
            target_probabilities=target_probs,
            cache_hash=h,
            step=int(cached.get("step", 0)),
            num_regions=list(cached.get("num_regions", [])),
            metrics=dict(cached.get("metrics", {})),
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore exact state from a checkpoint with integrity validation."""
        if not isinstance(state, Mapping):
            raise ValueError(f"state must be a mapping, got {type(state).__name__}")
        if "config" in state and isinstance(state["config"], Mapping):
            self.config = RegionConfig.from_mapping(state["config"])
        if "max_cache_size" in state:
            self.max_cache_size = max(1, int(state["max_cache_size"]))
        elif hasattr(self.config, "max_cache_size") and self.config.max_cache_size is not None:
            self.max_cache_size = max(1, int(self.config.max_cache_size))
        self.config.max_cache_size = self.max_cache_size
        self._last_step = int(state.get("last_step", -1))
        self._refresh_count = int(state.get("refresh_count", 0))
        self._cache_hash = str(state.get("cache_hash", ""))

        if self._cache_hash and len(self._cache_hash) != 64:
            raise ValueError(
                f"Corrupted cache state: cache_hash must be a 64-character hex string, got {self._cache_hash}"
            )

        self._cache.clear()

        # Restore cache entries
        entries = state.get("cache_entries", [])
        if entries and isinstance(entries, Sequence):
            for e in entries:
                if not isinstance(e, Mapping):
                    raise ValueError("Corrupted cache entry in state_dict")
                key = str(e.get("key", ""))
                h = str(e.get("cache_hash", ""))
                if key and h and key != h:
                    raise ValueError(f"Corrupted cache entry: key {key} does not match hash {h}")
                tgt = self._deserialize_target(e)
                self._cache[key] = tgt

        cached = state.get("cached_targets")
        if cached is not None and isinstance(cached, Mapping):
            tgt = self._deserialize_target(cached)
            if self._cache_hash and tgt.cache_hash != self._cache_hash:
                raise ValueError("Corrupted cache state: cached_targets hash does not match state cache_hash")
            self._cached_targets = tgt
        else:
            self._cached_targets = None

        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

    def targets(
        self,
        image: Tensor,
        geometric_valid: Tensor,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        *,
        probabilities: Tensor | None = None,
        features: Tensor | None = None,
        step: int = 0,
    ) -> RegionTargets:
        """Generate or retrieve cached physical region targets.

        Strictly enforces:
        1. Fails closed on missing or invalid physical metadata (no silent 1mm default).
        2. Immutable sample-identity cache keyed by full image/support/metadata.
        3. Separate frozen teacher version per refresh interval; cadence preserved.
        4. Bounded cache capacity.
        """
        if self.config.mode in {"none", "off"}:
            b, _, z, h, w = image.shape
            device = image.device
            return RegionTargets(
                mode="none",
                region_labels=torch.full((b, z, h, w), -1, dtype=torch.long, device=device),
                region_weights=torch.zeros((b, z, h, w), dtype=torch.float32, device=device),
                target_probabilities=None,
                cache_hash="",
                step=step,
                num_regions=[0] * b,
                metrics={"active_regions": 0.0},
            )

        b, c_img, z_dim, h_dim, w_dim = image.shape
        device = image.device

        # In EMA mode, require actual supported detached EMA source before cache/build
        if self.config.mode == "ema":
            if probabilities is None:
                if features is not None:
                    raise ValueError(
                        "mode='ema' does not support features-only inputs without teacher probabilities. "
                        "An actual supported detached EMA probability source is required."
                    )
                raise ValueError(
                    "mode='ema' requires an actual supported detached EMA source (probabilities), got None. "
                    "Use mode='image' for image-only region learning."
                )
            if probabilities.ndim != 5:
                raise ValueError(
                    f"probabilities must have 5 dimensions (B, C, Z, H, W), got {probabilities.ndim}"
                )
            if probabilities.shape[0] != b or probabilities.shape[2:] != (z_dim, h_dim, w_dim):
                raise ValueError(
                    f"probabilities spatial shape {probabilities.shape} does not match image shape {image.shape}"
                )

        # Reject missing physical metadata in active region learning
        if metadata is None or len(metadata) < b:
            raise ValueError(
                f"Missing physical metadata: expected {b} metadata entries, got "
                f"{0 if metadata is None else len(metadata)}. Missing physical metadata fails closed."
            )
        for bi in range(b):
            m = metadata[bi]
            if not isinstance(m, Mapping):
                raise ValueError(f"Metadata entry {bi} must be a mapping, got {type(m).__name__}")
            sp = m.get("spacing_xyz_mm")
            if sp is None or len(sp) != 3 or any(not np.isfinite(x) or x <= 0 for x in sp):
                raise ValueError(
                    f"Missing or invalid spacing_xyz_mm in metadata[{bi}]: {sp}. "
                    "Physical voxel spacing (mm) is required for physically-aware region learning."
                )

        center_ch = 1 if c_img == 3 else 0
        img_np = image[:, center_ch].detach().cpu().numpy().astype(np.float32)
        valid_np = geometric_valid.detach().cpu().numpy().astype(bool)

        prob_np = probabilities.detach().cpu().numpy().astype(np.float32) if probabilities is not None else None
        feat_np = features.detach().cpu().numpy().astype(np.float32) if features is not None else None

        sample_targets: list[RegionTargets] = []
        fresh_computed_count = 0
        all_hit = True

        for bi in range(b):
            sub_img = image[bi : bi + 1]
            sub_val = geometric_valid[bi : bi + 1]
            sub_meta = [metadata[bi]]
            sub_prob = probabilities[bi : bi + 1] if probabilities is not None else None
            sub_feat = features[bi : bi + 1] if features is not None else None

            sample_hash = compute_cache_hash(
                sub_img,
                sub_val,
                sub_meta,
                step,
                self.config.mode,
                config=self.config,
                refresh_every=self.config.refresh_every,
            )

            can_use_cache = (
                self.config.refresh_every > 0
                and sample_hash in self._cache
            )

            if can_use_cache:
                cached_s = self._cache[sample_hash]
                self._cache.move_to_end(sample_hash)
                sample_targets.append(cached_s)
            else:
                all_hit = False
                sp = metadata[bi]["spacing_xyz_mm"]
                spacing_zyx = np.array([float(sp[2]), float(sp[1]), float(sp[0])], dtype=np.float32)

                s_img_np = img_np[bi]
                s_val_np = valid_np[bi]
                s_prob_np = prob_np[bi] if prob_np is not None else None
                s_feat_np = feat_np[bi] if feat_np is not None else None

                guiding_feats = s_prob_np if (self.config.mode == "ema" and s_prob_np is not None) else s_feat_np

                lbls, n_r = generate_deterministic_supervoxels_3d(
                    s_img_np,
                    s_val_np,
                    spacing_zyx,
                    target_num_regions=self.config.target_num_regions,
                    spatial_weight=self.config.spatial_weight,
                    min_region_size=self.config.min_region_size,
                    iterations=self.config.iterations,
                    features_czyx=guiding_feats,
                )

                wgts = compute_boundary_weights(
                    s_img_np,
                    s_val_np,
                    spacing_zyx,
                    probabilities_czyx=s_prob_np if self.config.mode == "ema" else None,
                    config=self.config,
                )

                lbls, wgts, n_r = split_or_reject_cross_class_regions(
                    lbls,
                    wgts,
                    probabilities_czyx=s_prob_np if self.config.mode == "ema" else None,
                    valid_mask=s_val_np,
                    config=self.config,
                )

                s_region_labels = torch.as_tensor(lbls[None], dtype=torch.long, device=device)
                s_region_weights = torch.as_tensor(wgts[None], dtype=torch.float32, device=device)

                s_target_probs = None
                if self.config.mode == "ema" and sub_prob is not None:
                    s_target_probs = sub_prob.detach().to(device=device, dtype=torch.float32).clone()

                fresh_s = RegionTargets(
                    mode=self.config.mode,
                    region_labels=s_region_labels,
                    region_weights=s_region_weights,
                    target_probabilities=s_target_probs,
                    cache_hash=sample_hash,
                    step=step,
                    num_regions=[n_r],
                    metrics={"num_regions": float(n_r)},
                )

                if self.config.refresh_every > 0:
                    self._cache[sample_hash] = fresh_s
                    while len(self._cache) > self.max_cache_size:
                        self._cache.popitem(last=False)

                sample_targets.append(fresh_s)
                fresh_computed_count += 1

        if b == 1:
            combined_hash = sample_targets[0].cache_hash
        else:
            combined_hash = hashlib.sha256("::".join(s.cache_hash for s in sample_targets).encode()).hexdigest()

        if fresh_computed_count > 0:
            self._refresh_count += 1

        # Preserve single-sample cached target identity
        if b == 1:
            res = sample_targets[0]
            self._cached_targets = res
            self._cache_hash = combined_hash
            self._last_step = step
            return res

        if all_hit and self._cached_targets is not None and self._cache_hash == combined_hash:
            self._last_step = step
            return self._cached_targets

        combined_labels = torch.cat([s.region_labels for s in sample_targets], dim=0)
        combined_weights = torch.cat([s.region_weights for s in sample_targets], dim=0)
        combined_probs = (
            torch.cat([s.target_probabilities for s in sample_targets], dim=0)
            if self.config.mode == "ema" and sample_targets[0].target_probabilities is not None
            else None
        )
        combined_num_regions = [s.num_regions[0] for s in sample_targets]

        metrics = {
            "mean_num_regions": float(np.mean(combined_num_regions)),
            "total_regions": float(sum(combined_num_regions)),
            "refresh_count": float(self._refresh_count),
        }

        combined_targets = RegionTargets(
            mode=self.config.mode,
            region_labels=combined_labels,
            region_weights=combined_weights,
            target_probabilities=combined_probs,
            cache_hash=combined_hash,
            step=step,
            num_regions=combined_num_regions,
            metrics=metrics,
        )

        self._cached_targets = combined_targets
        self._cache_hash = combined_hash
        self._last_step = step

        return combined_targets

    def loss(
        self,
        student_logits: Tensor,
        targets: RegionTargets,
        geometric_valid: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute differentiable region consistency loss on student logits."""
        return region_consistency_loss(
            student_logits,
            targets,
            geometric_valid=geometric_valid,
            loss_type=self.config.loss_type,
            weight=self.config.weight,
        )


__all__ = [
    "RegionConfig",
    "RegionManager",
    "RegionTargets",
    "compute_boundary_weights",
    "compute_cache_hash",
    "compute_physical_gradient",
    "compute_sample_identity_hash",
    "generate_deterministic_supervoxels_3d",
    "region_consistency_loss",
    "split_or_reject_cross_class_regions",
]
