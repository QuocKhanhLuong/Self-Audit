"""Normalized graph Laplacian and explicit clustering; no hidden scientific defaults."""
import numpy as np


def normalized_laplacian(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1] or not np.isfinite(weights).all():
        raise ValueError("finite square affinity required")
    if not np.allclose(weights, weights.T):
        raise ValueError("directed affinity needs a protocol-defined symmetrization")
    degree = weights.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("nonpositive degree: no unregistered epsilon/diagonal repair")
    scale = 1 / np.sqrt(degree)
    return (np.diag(degree) - weights) * scale[:, None] * scale[None, :]


def spectral_embedding(weights, *, dimensions, discard_first, normalize_rows):
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian(weights))
    start = int(discard_first)
    if not 0 < dimensions <= len(eigenvalues) - start:
        raise ValueError("invalid explicit eigenspace dimension")
    embedding = eigenvectors[:, start:start + dimensions]
    if normalize_rows:
        norms = np.linalg.norm(embedding, axis=1)
        if np.any(norms == 0):
            raise ValueError("zero embedding row")
        embedding = embedding / norms[:, None]
    return embedding, eigenvalues


def kmeans_labels(features, *, clusters, seed, n_init, max_iter, tolerance, algorithm):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=clusters, random_state=seed, n_init=n_init,
                   max_iter=max_iter, tol=tolerance, algorithm=algorithm)
    return model.fit_predict(features).astype(np.int32), model
