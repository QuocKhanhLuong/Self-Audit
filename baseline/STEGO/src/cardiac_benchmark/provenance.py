"""Canonical provenance records for STEGO cardiac benchmark."""

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

SCHEMA_VERSION = "stego.cardiac.provenance.v1"


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


def current_stego_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def checkpoint_provenance(checkpoint_path: str | Path) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_domain": "cityscapes",
        "backbone": "DINO ViT-Base/8 (ImageNet self-supervised)",
        "head": "STEGO contrastive (Cityscapes unsupervised)",
        "role": "S1_SSL_out_of_domain",
        "compute_cost_note": "pretrained out-of-domain; compute cost not incurred by Self-Audit",
    }


def classify_prior_checkpoint(checkpoint_path: str | Path | None) -> dict[str, Any]:
    if checkpoint_path is None:
        return {
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "checkpoint_format": "implicit_dino_download",
            "backbone_prior": "official_dino_vit",
            "head_state_presence": "none",
            "fair_status": "matched",
            "allowed_for_fair_mode": True,
            "requires_explicit_override": False,
            "note": (
                "No local checkpoint supplied; fair STEGO will defer to the official "
                "DINO teacher-weight loader."
            ),
        }

    resolved = Path(checkpoint_path).resolve()
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if "teacher" in checkpoint:
        teacher_state = checkpoint["teacher"]
        return {
            "checkpoint_path": str(resolved),
            "checkpoint_sha256": sha256_file(resolved),
            "checkpoint_format": "dino_teacher",
            "backbone_prior": "official_dino_teacher",
            "head_state_presence": "none",
            "teacher_key_count": len(teacher_state),
            "fair_status": "matched",
            "allowed_for_fair_mode": True,
            "requires_explicit_override": False,
            "note": "Teacher-format DINO checkpoint is an acceptable B4 backbone prior.",
        }

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        backbone_key_count = sum(1 for key in state_dict if key.startswith("net.model."))
        head_key_count = sum(
            1
            for key in state_dict
            if key.startswith("net.cluster1.") or key.startswith("net.cluster2.")
        )
        return {
            "checkpoint_path": str(resolved),
            "checkpoint_sha256": sha256_file(resolved),
            "checkpoint_format": "stego_lightning",
            "backbone_prior": "stego_backbone_from_downstream_checkpoint",
            "head_state_presence": "downstream_projection_head_present" if head_key_count else "none_detected",
            "backbone_key_count": backbone_key_count,
            "head_key_count": head_key_count,
            "fair_status": "fidelity_gap",
            "allowed_for_fair_mode": False,
            "requires_explicit_override": True,
            "note": (
                "Lightning STEGO checkpoint contains downstream projection-head state; "
                "fair B4 should prefer a DINO teacher prior unless an explicit override "
                "accepts this adaptation."
            ),
        }

    raise ValueError(f"Unknown checkpoint format in {resolved}")


def rng_contract() -> dict[str, Any]:
    return {
        "benchmark_seed": 42,
        "inference_seed": 42,
        "note": "STEGO uses pretrained checkpoint; no training RNG within cardiac benchmark",
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
