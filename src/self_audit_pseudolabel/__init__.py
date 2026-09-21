"""Experimental cine pseudo-label teacher + adaptive deployment annotator."""
from .system_v3 import AdaptiveAnnotationStudent,CinePseudoTeacher,PROFILES,ResourceProfile,UNKNOWN,pseudo_supervision_loss
from .adaptive import AdaptiveRuntime,RuntimeBudget,choose_profile
from .evidence import EvidenceConfig,build_region_evidence
from .evolution import PrototypeBank,accepted_region_mask

__all__=[
    "AdaptiveAnnotationStudent","CinePseudoTeacher","PROFILES","ResourceProfile","UNKNOWN","pseudo_supervision_loss",
    "AdaptiveRuntime","RuntimeBudget","choose_profile","EvidenceConfig","build_region_evidence",
    "PrototypeBank","accepted_region_mask",
]
