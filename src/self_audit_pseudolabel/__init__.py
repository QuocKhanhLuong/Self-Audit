"""Experimental cine pseudo-label teacher + adaptive deployment annotator."""

from .system_v3 import (
    AdaptiveAnnotationStudent,
    CinePseudoTeacher,
    PROFILES,
    ResourceProfile,
    UNKNOWN,
    pseudo_supervision_loss,
)

__all__ = [
    "AdaptiveAnnotationStudent",
    "CinePseudoTeacher",
    "PROFILES",
    "ResourceProfile",
    "UNKNOWN",
    "pseudo_supervision_loss",
]
