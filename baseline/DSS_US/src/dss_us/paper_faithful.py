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


# ---------------------------------------------------------------------------
# Official-pipeline fallbacks (OFFICIAL_CODE_FALLBACK), independently implemented
# from the documented behaviour of alexaatm/UnsupervisedSegmentor4Ultrasound@d4ac44c
# (deep-spectral-segmentation/extract). Used only when a profile declares them.
# ---------------------------------------------------------------------------
STEP2_OFFICIAL = {"erode": 2, "dilate": 5, "min_crop": 8, "batch_size": 4096, "max_iter": 5000, "n_init": 10}


def infer_background_relabel(segmap):
    """Segment covering the largest share of the image border becomes label 0 (labels swapped)."""
    segmap = np.array(segmap, copy=True)
    border = np.concatenate([segmap[:, 0], segmap[:, -1], segmap[0, :], segmap[-1, :]])
    ids, counts = np.unique(border, return_counts=True)
    present = np.unique(segmap)
    totals = np.array([counts[ids == i].sum() if i in ids else 0 for i in present])
    background = int(present[int(np.argmax(totals))])
    swapped = segmap.copy()
    swapped[segmap == background] = 0
    swapped[segmap == 0] = background
    return swapped


def presegment(keys, gray, grid_hw, settings):
    """Step I eigensegments at patch resolution (before upscaling and CRF), background relabelled."""
    weights = step1_affinity(keys, gray, grid_hw, settings)
    labels, fit = oversegment(weights, grid_hw, dimensions=int(_require(settings, "eigenvectors")),
                              discard_first=bool(_require(settings, "discard_trivial_eigenvector", {True, False})),
                              normalize_rows=bool(_require(settings, "normalize_embedding_rows", {True, False})),
                              clusters=STEP1_SEGMENTS, **_kmeans(settings))
    return infer_background_relabel(labels), fit


def _morph(mask, steps, erode):
    from skimage.morphology import binary_dilation, binary_erosion
    operation = binary_erosion if erode else binary_dilation
    for _ in range(steps):
        updated = operation(mask)
        if updated.sum() > 0:  # never erode a mask away completely
            mask = updated
    return mask


def official_segment_boxes(segmap):
    """(segment id, (ymin, ymax, xmin, xmax)) at patch resolution; background 0 skipped."""
    boxes = []
    for segment_id in sorted(int(s) for s in np.unique(segmap)):
        if segment_id == 0:
            continue
        mask = _morph(segmap == segment_id, STEP2_OFFICIAL["erode"], True)
        mask = _morph(mask, STEP2_OFFICIAL["dilate"], False)
        rows, cols = np.nonzero(mask)
        boxes.append((segment_id, (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)))
    return boxes


def _official_pad(crop):
    import torch.nn.functional as F
    _, _, height, width = crop.shape
    target = STEP2_OFFICIAL["min_crop"]
    pad_h = max(max(0, target - height), target)
    pad_w = max(max(0, target - width), target)
    return F.pad(crop, (0, pad_w, 0, pad_h), mode="constant", value=0)


def step2_dss_official(items, settings, *, embed):
    """DSS step2 over the fit cohort. items: [(sample_id, image_tensor [1,3,H,W], segmap at patch grid)].

    Returns {sample_id: semantic map at patch resolution} and the clustering receipt.
    """
    from sklearn.cluster import MiniBatchKMeans
    if _require(settings, "step2_variant", {"dss_step2", "ours_step2"}) != "dss_step2":
        raise ProtocolSettingError("only DSS step2 is wired in the paper-faithful runner")
    if _require(settings, "step2_crf_applied", {True, False}):
        raise ProtocolSettingError("Step II CRF requested but no dense-CRF backend exists in the canonical environment")
    rows, index = [], []
    for sample_id, tensor, segmap in items:
        for segment_id, (top, bottom, left, right) in official_segment_boxes(segmap):
            crop = tensor[:, :, top * PATCH_SIZE:bottom * PATCH_SIZE, left * PATCH_SIZE:right * PATCH_SIZE]
            if crop.shape[-2] < STEP2_OFFICIAL["min_crop"] or crop.shape[-1] < STEP2_OFFICIAL["min_crop"]:
                crop = _official_pad(crop)
            rows.append(np.asarray(embed(crop), dtype=np.float64))
            index.append((sample_id, segment_id))
    if not rows:
        raise ValueError("no foreground segments in the fit cohort")
    features = np.stack(rows)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    clusters = MiniBatchKMeans(n_clusters=int(_require(settings, "semantic_clusters")),
                               batch_size=STEP2_OFFICIAL["batch_size"], max_iter=STEP2_OFFICIAL["max_iter"],
                               random_state=int(_require(settings, "kmeans_seed")),
                               n_init=STEP2_OFFICIAL["n_init"]).fit_predict(features)
    assignment = {}
    for (sample_id, segment_id), cluster in zip(index, clusters):
        assignment.setdefault(sample_id, {})[segment_id] = int(cluster)
    outputs = {}
    for sample_id, _tensor, segmap in items:
        mapping = {0: 0, **assignment.get(sample_id, {})}
        outputs[sample_id] = np.vectorize(mapping.__getitem__)(segmap).astype(np.int32)
    return outputs, {"segments": len(index), "clusters": int(_require(settings, "semantic_clusters"))}


def official_resize_to_image(labels, image_hw):
    """Nearest-neighbour resize of a label map to the image grid (cv2.INTER_NEAREST)."""
    import cv2
    height, width = image_hw
    return cv2.resize(np.asarray(labels, dtype=np.uint8 if labels.max() < 256 else np.int32),
                      dsize=(width, height), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def dss_settings(config):
    settings = dict(config["scientific"])
    for group in ("paper_unspecified", "implementation_conventions"):
        settings.update({name: entry.get("value") for name, entry in config.get(group, {}).items()})
    return settings


OFFICIAL_PHI_IMAGE = "dino_vits8_output_embedding_of_imagenet_normalised_bbox_crop"
OFFICIAL_DSS_STEP2 = "official_bbox_pipeline_v1"
DINO_INPUT_POLICY = "rgb_totensor_imagenet_crop_to_patch_multiple"


def validate_dss_step2_settings(settings):
    """Pre-flight check for the wired Step II (DSS step2) path; fails before any data access."""
    _require(settings, "dino_input_policy", {DINO_INPUT_POLICY})
    _require(settings, "feature_l2_normalization", {True, False})
    for name in ("eigenvectors", "semantic_clusters"):
        if int(_require(settings, name)) < 1:
            raise ProtocolSettingError(f"{name} must be positive")
    _require(settings, "discard_trivial_eigenvector", {True, False})
    _require(settings, "normalize_embedding_rows", {True, False})
    _require(settings, "upscale_method", UPSCALE_METHODS)
    _kmeans(settings)
    if _require(settings, "step2_variant", {"dss_step2", "ours_step2"}) != "dss_step2":
        raise ProtocolSettingError("Ours step2 embeddings (phi_mask, phi_position, C_mask, C_position) are not wired")
    if _require(settings, "step2_crf_applied", {True, False}):
        raise ProtocolSettingError("Step II CRF requested but no dense-CRF backend exists in the canonical environment")
    _require(settings, "phi_image", {OFFICIAL_PHI_IMAGE})
    _require(settings, "dss_step2_definition", {OFFICIAL_DSS_STEP2})
    if settings.get("preprocessing"):
        method = _require(settings, "preprocessing_method",
                          {"histogram_equalization", "gaussian_blur", "histogram_equalization+gaussian_blur"})
        parameters = _require(settings, "preprocessing_parameters")
        if "gaussian_blur" in method and not {"gaussian_kernel", "gaussian_sigma"} <= set(parameters):
            raise ProtocolSettingError("gaussian_blur requires gaussian_kernel and gaussian_sigma")
    if settings.get("ultrasound_affinities"):
        for name in ("c_ssd", "c_mi", "c_pos", "patch_intensity_scale"):
            _require(settings, name)
        if settings["c_ssd"]:
            _require(settings, "delta_ssd")
        if settings["c_mi"]:
            for name in ("delta_mi", "mi_bins", "mi_intensity_range"):
                _require(settings, name)
        if settings["c_pos"]:
            _require(settings, "positional_knn_k")
            _require(settings, "positional_symmetrization", SYMMETRIZATIONS)
    return settings
