"""Strict, image-only DFC cardiac P0 wrapper.

This package deliberately ends at anonymous ``int32[H,W]`` partitions.
"""
from pathlib import Path
import sys

_PROJECT_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

from .dfc_runner import DFCConfig, DFCResult, MyNet, run_dfc
from .provenance import derive_sample_seed

__all__ = ["DFCConfig", "DFCResult", "MyNet", "derive_sample_seed", "run_dfc"]
