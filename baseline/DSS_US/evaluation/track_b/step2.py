"""Step II semantic matching primitive, evaluator-only.

Matching mechanics (Hungarian for equal counts, exclusive majority vote otherwise)
are resolved in spec.json; complete Step II evaluation stays gated in run.py
because the evaluated stage (CRF or not) and the label-consistency metric are
unresolved.
"""
import numpy as np

try:
    from .matching import match_segments, remap
except ImportError:  # loaded by file path
    import importlib.util as _u, pathlib as _p
    _spec = _u.spec_from_file_location("dss_us_track_b__sibling", _p.Path(__file__).with_name("_sibling.py"))
    _sibling = _u.module_from_spec(_spec); _spec.loader.exec_module(_sibling)
    _matching = _sibling.load("matching")
    match_segments, remap = _matching.match_segments, _matching.remap


def semantic_match(partition, gt):
    mapping, _branch = match_segments(partition, gt)
    return remap(partition, mapping).astype(np.asarray(gt).dtype), mapping
