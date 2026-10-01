"""Independent Step I computation. Caller must supply an evidenced, locked recipe."""
from .spectral import spectral_embedding, kmeans_labels


def oversegment(weights, grid_hw, *, dimensions, discard_first, normalize_rows,
                clusters, seed, n_init, max_iter, tolerance, algorithm):
    embedding, eigenvalues = spectral_embedding(weights, dimensions=dimensions,
                                                discard_first=discard_first, normalize_rows=normalize_rows)
    ids, fitted = kmeans_labels(embedding, clusters=clusters, seed=seed, n_init=n_init,
                              max_iter=max_iter, tolerance=tolerance, algorithm=algorithm)
    if len(ids) != grid_hw[0] * grid_hw[1]:
        raise ValueError("patch grid and affinity do not align")
    return ids.reshape(grid_hw), {"eigenvalues": eigenvalues, "centers": fitted.cluster_centers_}
