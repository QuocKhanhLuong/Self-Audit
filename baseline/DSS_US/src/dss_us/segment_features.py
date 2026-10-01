"""Paper's dual-embedding linear combination; embedding recipes are externally locked."""
import numpy as np


def combine_segment_features(image_features, mask_features, position_features, *, c_mask, c_position):
    value = np.asarray(image_features, dtype=np.float64).copy()
    for coefficient, features in ((c_mask, mask_features), (c_position, position_features)):
        if coefficient != 0:
            if features is None or features.shape != value.shape:
                raise ValueError("active segment embedding needs aligned explicit features")
            value += coefficient * features
    if not np.isfinite(value).all():
        raise ValueError("nonfinite segment representation")
    return value
