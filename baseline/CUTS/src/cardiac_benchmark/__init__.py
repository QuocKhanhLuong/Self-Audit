"""P0 image-only cardiac benchmark shell for CUTS.

This package deliberately ends at an anonymous K=10 partition.  It contains
no cardiac semantic adapter and no reference-label evaluation code.
"""
from pathlib import Path
import sys

_PROJECT_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

from .manifest import MANIFEST_SCHEMA_VERSION, load_manifest, require_scientific_manifest

__all__ = ["MANIFEST_SCHEMA_VERSION", "load_manifest", "require_scientific_manifest"]
