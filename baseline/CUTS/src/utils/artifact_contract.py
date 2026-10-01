"""Small provenance helpers; no model or diffusion execution."""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_scalar(value):
    return np.asarray(json.dumps(value, sort_keys=True))


def require_empty_output(path):
    root = Path(path)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {root}; choose a new run path')


def validate_hierarchy(labels, granularities, pixels):
    if labels is None or granularities is None:
        raise ValueError('Diffusion failed; refusing to save an invalid hierarchy')
    labels, granularities = np.asarray(labels), np.asarray(granularities)
    if (labels.ndim != 2 or labels.shape[0] < 2 or labels.shape[1] != pixels
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)
            or granularities.shape != (labels.shape[0],)
            or not np.issubdtype(granularities.dtype, np.integer)):
        raise ValueError('Invalid diffusion hierarchy or granularities')
