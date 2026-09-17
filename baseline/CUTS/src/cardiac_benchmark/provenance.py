"""Canonical P0 provenance records, deliberately separate RNG roles."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = "cuts.cardiac.p0.provenance.v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    header = canonical_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return sha256_bytes(header + np.ascontiguousarray(array).tobytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def current_cuts_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def rng_contract(*, loader_seed_policy: str) -> dict[str, Any]:
    return {
        "split_seed": 42,
        "benchmark_seed": 42,
        "model_init_seed": 42,
        "training_rng_seed": 42,
        "loader_seed_policy": loader_seed_policy,
        "patch_sampler": {"constructor_seed": 42, "behavior": "official_per_call_reseeding_preserved"},
        "clustering_seed": 1,
        "clustering_retry_seed": 2,
    }


def environment_identity() -> dict[str, Any]:
    import phate
    import scipy
    import sklearn
    import skimage
    import yaml
    import nibabel
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "scikit_image": skimage.__version__,
        "phate": phate.__version__,
        "nibabel": nibabel.__version__,
        "pyyaml": yaml.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_json(path: str | Path, value: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_bytes(value) + b"\n")
    return sha256_file(target)
