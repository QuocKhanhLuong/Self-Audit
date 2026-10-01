"""Canonical provenance records for PICIE cardiac benchmark."""

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

SCHEMA_VERSION = "picie.cardiac.provenance.v1"


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


def current_picie_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def checkpoint_provenance(checkpoint_path: str | Path) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_domain": "medical",
        "backbone": "ResNet-18 (from scratch)",
        "head": "PICIE clustering",
        "role": "S0_from_scratch",
        "compute_cost_note": "trained from scratch on medical data",
    }


def rng_contract() -> dict[str, Any]:
    return {
        "benchmark_seed": 42,
        "inference_seed": 42,
        "note": "PICIE inference uses deterministic settings",
    }


def environment_identity() -> dict[str, Any]:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import pytorch_lightning as pl
        env["pytorch_lightning"] = pl.__version__
    except ImportError:
        pass
    try:
        import torchvision
        env["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    try:
        import scipy
        env["scipy"] = scipy.__version__
    except ImportError:
        pass
    return env


def write_json(path: str | Path, value: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_bytes(value) + b"\n")
    return sha256_file(target)