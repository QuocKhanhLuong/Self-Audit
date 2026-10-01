"""Lazy exports: independent freeze/evaluation utilities do not load torch or models."""
from importlib import import_module
_EXPORTS={
 'AdaptiveAnnotationStudent':'system_v3','CinePseudoTeacher':'system_v3','PROFILES':'system_v3',
 'ResourceProfile':'system_v3','UNKNOWN':'system_v3','pseudo_supervision_loss':'system_v3',
 'AdaptiveRuntime':'adaptive','RuntimeBudget':'adaptive','choose_profile':'adaptive',
 'EvidenceConfig':'evidence','build_region_evidence':'evidence',
 'PrototypeBank':'evolution','accepted_region_mask':'evolution',
}
__all__=list(_EXPORTS)
def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(name)
    value=getattr(import_module('.'+_EXPORTS[name],__name__),name)
    globals()[name]=value
    return value
