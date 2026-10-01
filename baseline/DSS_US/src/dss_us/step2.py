"""Dataset-level semantic clustering from explicitly supplied segment features."""
import numpy as np
from .spectral import kmeans_labels


def semantic_cluster(features, image_segments, *, clusters, seed, n_init, max_iter, tolerance, algorithm):
    """image_segments are (partition, segment_ids); rows follow this explicit order."""
    cluster_ids, fitted = kmeans_labels(features, clusters=clusters, seed=seed, n_init=n_init,
                                      max_iter=max_iter, tolerance=tolerance, algorithm=algorithm)
    total = sum(len(ids) for _, ids in image_segments)
    if total != len(features):
        raise ValueError("fit-set segment inventory does not match feature rows")
    outputs = []
    offset = 0
    for partition, segment_ids in image_segments:
        result = np.full(partition.shape, -1, dtype=np.int32)
        for segment_id in segment_ids:
            result[partition == segment_id] = cluster_ids[offset]
            offset += 1
        if np.any(result < 0):
            raise ValueError("unmapped segment: background policy must be explicitly resolved")
        outputs.append(result)
    return outputs, {"centers": fitted.cluster_centers_, "segment_assignment": cluster_ids}
