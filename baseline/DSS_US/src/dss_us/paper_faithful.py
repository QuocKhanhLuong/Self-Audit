"""DSS-US PAPER_FAITHFUL_REIMPLEMENTATION (Tmenova et al., arXiv 2408.02043v1).

Composes the independent equation kernels exactly as the paper describes:

Step I (section 2.1): DINO ViT-S/8 last-layer keys f -> W_feat = f f^T masked to positive
values (Eq. 1); optional ultrasound affinities W_ssd / W_mi = exp(-delta * D) over k x k
patches with k = 8 (Eqs. 2-3); optional positional KNN affinity over (x, y) in [0, 1]
(Eq. 4); W_comb = W_feat + C_ssd W_ssd + C_mi W_mi + C_pos W_pos (Eq. 5); normalised
Laplacian D^-1/2 (D - W) D^-1/2 (Eq. 6); eigenvectors -> K-means with 15 segments;
upscale and CRF ("We upscale and apply CRF", marked for all Step I rows in Table 1).

Step II (section 2.2): per-segment bounding-box crop features, optionally linearly
combined with mask and position embeddings (dual embedding, "Ours step2"), then K-means.

Every value the paper leaves symbolic or unstated is a required setting with no
default. CRF requires an explicit backend callable; none ships with the canonical
environment. No unlicensed upstream source is used.
"""
from __future__ import annotations

import numpy as np

from .affinity import combine_affinities, mi_affinity, positional_affinity, positive_feature_affinity, ssd_affinity
from .segment_features import combine_segment_features
from .step1 import oversegment
from .step2 import semantic_cluster

STEP1_SEGMENTS = 15          # paper section 2.1 and Table 1
PATCH_SIZE = 8               # paper: k matches the ViT-S/8 patch size
SYMMETRIZATIONS = {"max", "mean", "min"}
UPSCALE_METHODS = {"nearest"}


class ProtocolSettingError(ValueError):
    pass


def _require(settings, name, allowed=None):
    value = settings.get(name)
    if value is None:
        raise ProtocolSettingError(f"paper-faithful setting required: {name}")
    if allowed is not None and value not in allowed:
        raise ProtocolSettingError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _kmeans(settings):
    return {"seed": int(_require(settings, "kmeans_seed")), "n_init": int(_require(settings, "kmeans_n_init")),
            "max_iter": int(_require(settings, "kmeans_max_iter")), "tolerance": float(_require(settings, "kmeans_tolerance")),
            "algorithm": _require(settings, "kmeans_algorithm")}


def image_patches(gray, grid_hw):
    """Non-overlapping k x k patches aligned with the transformer patch grid (k = 8)."""
    gray = np.asarray(gray, dtype=np.float64)
    rows, cols = grid_hw
    if gray.shape != (rows * PATCH_SIZE, cols * PATCH_SIZE):
        raise ValueError("image must already match the patch grid (explicit crop/resize policy)")
    return gray.reshape(rows, PATCH_SIZE, cols, PATCH_SIZE).transpose(0, 2, 1, 3).reshape(rows * cols, PATCH_SIZE, PATCH_SIZE)


def _symmetrize(matrix, mode):
    if mode == "max":
        return np.maximum(matrix, matrix.T)
    if mode == "min":
        return np.minimum(matrix, matrix.T)
    return 0.5 * (matrix + matrix.T)


def step1_affinity(keys, gray, grid_hw, settings):
    """Eqs. 1-5. ``settings['ultrasound_affinities']`` is the row flag (Ours_Aff / Ours_comb)."""
    normalize = bool(_require(settings, "feature_l2_normalization", {True, False}))
    if not settings["ultrasound_affinities"]:
        return positive_feature_affinity(keys, normalize=normalize)
    patches = image_patches(gray, grid_hw) * float(_require(settings, "patch_intensity_scale"))
    c_ssd, c_mi, c_pos = (float(_require(settings, name)) for name in ("c_ssd", "c_mi", "c_pos"))
    ssd = ssd_affinity(patches, delta=float(_require(settings, "delta_ssd"))) if c_ssd else None
    mi = (mi_affinity(patches, delta=float(_require(settings, "delta_mi")), bins=int(_require(settings, "mi_bins")),
                      intensity_range=tuple(_require(settings, "mi_intensity_range"))) if c_mi else None)
    position = None
    if c_pos:
        position = _symmetrize(positional_affinity(grid_hw, neighbors=int(_require(settings, "positional_knn_k"))),
                               _require(settings, "positional_symmetrization", SYMMETRIZATIONS))
    return combine_affinities(keys, normalize=normalize, ssd=ssd, mi=mi, position=position,
                              c_ssd=c_ssd, c_mi=c_mi, c_pos=c_pos)


def upscale_labels(labels, image_hw, settings):
    _require(settings, "upscale_method", UPSCALE_METHODS)
    rows, cols = labels.shape
    height, width = image_hw
    row_index = np.minimum((np.arange(height) * rows) // height, rows - 1)
    col_index = np.minimum((np.arange(width) * cols) // width, cols - 1)
    return labels[row_index][:, col_index]


def step1_partition(keys, gray, grid_hw, image_hw, settings, *, crf_backend=None, image=None):
    """Paper Step I for one image; returns the CRF-postprocessed partition at image size."""
    weights = step1_affinity(keys, gray, grid_hw, settings)
    labels, fit = oversegment(weights, grid_hw, dimensions=int(_require(settings, "eigenvectors")),
                              discard_first=bool(_require(settings, "discard_trivial_eigenvector", {True, False})),
                              normalize_rows=bool(_require(settings, "normalize_embedding_rows", {True, False})),
                              clusters=STEP1_SEGMENTS, **_kmeans(settings))
    partition = upscale_labels(labels, image_hw, settings)
    if crf_backend is None:
        raise ProtocolSettingError("Step I applies CRF (Table 1); an explicit CRF backend is required")
    refined = np.asarray(crf_backend(image, partition, _require(settings, "crf_parameters")))
    if refined.shape != tuple(image_hw) or not np.issubdtype(refined.dtype, np.integer):
        raise ValueError("CRF backend must return an integer partition at image size")
    return refined.astype(np.int32), {"eigenvalues": fit["eigenvalues"]}


def segment_inputs(image, partition, segment_id):
    """Bounding-box crop of the image and of the segment mask (section 2.2)."""
    rows, cols = np.nonzero(partition == segment_id)
    if rows.size == 0:
        raise ValueError(f"segment {segment_id} is empty")
    top, bottom, left, right = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    crop = np.asarray(image)[top:bottom, left:right]
    mask = (partition[top:bottom, left:right] == segment_id)
    return crop, mask, (top, bottom, left, right)


def step2_semantic(images_and_partitions, settings, *, phi_image, phi_mask=None, phi_position=None):
    """Paper Step II over a fit set: f = f_image (+ C_mask f_mask + C_pos f_pos for Ours step2) -> K-means."""
    variant = _require(settings, "step2_variant", {"dss_step2", "ours_step2"})
    image_rows, mask_rows, position_rows, inventory = [], [], [], []
    for image, partition in images_and_partitions:
        segment_ids = [int(s) for s in np.unique(partition)]
        for segment_id in segment_ids:
            crop, mask, box = segment_inputs(image, partition, segment_id)
            image_rows.append(np.asarray(phi_image(crop), dtype=np.float64))
            if variant == "ours_step2":
                if phi_mask is None or phi_position is None:
                    raise ProtocolSettingError("Ours step2 requires explicit mask and position embeddings")
                mask_rows.append(np.asarray(phi_mask(mask), dtype=np.float64))
                position_rows.append(np.asarray(phi_position(box, partition.shape), dtype=np.float64))
        inventory.append((partition, segment_ids))
    features = np.stack(image_rows)
    if variant == "ours_step2":
        features = combine_segment_features(features, np.stack(mask_rows), np.stack(position_rows),
                                            c_mask=float(_require(settings, "c_mask")),
                                            c_position=float(_require(settings, "c_position_embedding")))
    return semantic_cluster(features, inventory, clusters=int(_require(settings, "semantic_clusters")), **_kmeans(settings))
