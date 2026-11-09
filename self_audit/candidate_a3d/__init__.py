"""Opt-in full-volume A-3D evaluation utilities.

This package is independent from :mod:`self_audit.candidate_a`: A-2D/2.5D
keeps its in-plane bank and packet protocol, while A-3D requires a native
``[T,D,H,W]`` cache and physical ``(dz,dy,dx)`` geometry.
"""

SCHEMA = "self_audit.candidate_a3d.v1"
CLASS_NAMES = ("BG", "RV", "MYO", "LV")
CLASS_MAPPING = {"0": "background", "1": "RV", "2": "MYO", "3": "LV", "255": "ignore"}
CHECKPOINT_SCHEMA = "self_audit.candidate_a3d.checkpoint.v1"
EXPORT_SCHEMA = "self_audit.candidate_a3d.student.v1"

from .geometry import VolumeGeometry3D, canonical_geometry, validate_geometry
from .bank import BankConfig3D, Query3D, QueryBank3D
from .codec import CodePacket3D, MeasurementCodec3D, QuantizedValues3D
from .data import CandidateA3DData, synthetic_fixture, synthetic_candidate_a3d_fixture

__all__ = [
    "SCHEMA", "CLASS_NAMES", "CLASS_MAPPING", "CHECKPOINT_SCHEMA", "EXPORT_SCHEMA",
    "VolumeGeometry3D", "canonical_geometry", "validate_geometry",
    "BankConfig3D", "Query3D", "QueryBank3D",
    "CodePacket3D", "MeasurementCodec3D", "QuantizedValues3D",
    "CandidateA3DData", "synthetic_fixture", "synthetic_candidate_a3d_fixture",
]
