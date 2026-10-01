"""Independent implementations of paper equations 1–5, with explicit parameters."""
import numpy as np


def positive_feature_affinity(features, *, normalize):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("finite 2-D feature matrix required")
    if normalize:
        norms = np.linalg.norm(features, axis=1)
        if np.any(norms == 0):
            raise ValueError("zero feature normalization is undefined")
        features = features / norms[:, None]
    return np.maximum(features @ features.T, 0)


def ssd_affinity(patches, *, delta):
    patches = np.asarray(patches, dtype=np.float64)
    flat = patches.reshape(len(patches), -1)
    difference = flat[:, None] - flat[None, :]
    distance = (difference * difference).sum(axis=2)
    return np.exp(-delta * distance)


def _entropy(histogram):
    probability = histogram[histogram > 0] / histogram.sum()
    return float(-(probability * np.log(probability)).sum())


def mi_affinity(patches, *, delta, bins, intensity_range):
    """Paper MI=(Hx+Hy)/Hxy, D=1-MI. Do not silently replace its definition."""
    patches = np.asarray(patches).reshape(len(patches), -1)
    result = np.empty((len(patches), len(patches)), dtype=np.float64)
    for i, left in enumerate(patches):
        for j, right in enumerate(patches):
            joint = np.histogram2d(left, right, bins=bins, range=[intensity_range, intensity_range])[0]
            if joint.sum() == 0:
                raise ValueError("histogram range excludes all pixels")
            denominator = _entropy(joint)
            if denominator == 0:
                raise ValueError("zero joint entropy: protocol must specify handling")
            mi = (_entropy(joint.sum(axis=0)) + _entropy(joint.sum(axis=1))) / denominator
            result[i, j] = np.exp(-delta * (1 - mi))
    return result


def positional_affinity(grid_hw, *, neighbors):
    h, w = grid_hw
    y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    positions = np.stack([x.ravel(), y.ravel()], axis=1)
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    if not 0 < neighbors <= len(positions):
        raise ValueError("invalid explicit neighbor count")
    choices = np.argsort(distances, axis=1, kind="stable")[:, :neighbors]
    value = np.zeros_like(distances)
    rows = np.arange(len(positions))[:, None]
    value[rows, choices] = 1 - distances[rows, choices]
    return value  # symmetrization policy is separately locked, never silently inferred


def combine_affinities(features, *, normalize, ssd, mi, position, c_ssd, c_mi, c_pos):
    value = positive_feature_affinity(features, normalize=normalize)
    for coefficient, affinity in ((c_ssd, ssd), (c_mi, mi), (c_pos, position)):
        if coefficient != 0:
            if affinity is None or affinity.shape != value.shape:
                raise ValueError("active affinity requires a matching explicit matrix")
            value = value + coefficient * affinity
    if not np.isfinite(value).all():
        raise ValueError("nonfinite combined affinity")
    return value
