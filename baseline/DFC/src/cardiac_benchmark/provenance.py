"""Canonical identity and deterministic RNG utilities for DFC P0."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

BENCHMARK_SEED = 42
SEED_VERSION = "dfc-sample-seed-v1"
UPSTREAM_DFC_SHA = "181318ad40dbfb5c0add8580a05c30e5e3a7ad58"
FREEMASK_POLICY_SHA = "96c32b10fc7b8e09b48822e10ae9eb6cc149e253"


def canonical_json(value: Any) -> bytes:
    """Stable UTF-8 JSON: ASCII escaped, sorted mappings, no spare whitespace."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_sample_seed(sample_id: str, benchmark_seed: int = BENCHMARK_SEED) -> dict[str, Any]:
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty canonical string")
    payload = [SEED_VERSION, benchmark_seed, sample_id]
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(serialized).digest()
    return {
        "seed_derivation_version": SEED_VERSION,
        "benchmark_seed": benchmark_seed,
        "seed_payload": serialized.decode("ascii"),
        "seed_digest": digest.hex(),
        "sample_seed": int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1),
    }


def seed_everything(sample_seed: int) -> None:
    """Set declared global generators immediately before CPU model construction."""
    random.seed(sample_seed)
    np.random.seed([sample_seed & 0xFFFFFFFF, sample_seed >> 32])
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)


def source_digest(paths: list[Path]) -> str:
    """Digest only source text identities, not paths or volatile runtime state."""
    return sha256_json({p.name: file_sha256(p) for p in sorted(paths)})
