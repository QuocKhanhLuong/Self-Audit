"""Strict loader for the immutable primary profile (JSON is valid YAML 1.2)."""
from __future__ import annotations
import json
from pathlib import Path
from .dfc_runner import DFCConfig

def load_primary_config(path: str | Path) -> DFCConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = set(DFCConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown: raise ValueError(f"unknown/prohibited config keys: {sorted(unknown)}")
    return DFCConfig(**raw)
