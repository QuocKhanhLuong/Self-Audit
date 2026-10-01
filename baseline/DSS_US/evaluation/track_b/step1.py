"""Step I evaluator: per-image Dice of the CRF-postprocessed oversegmentation.

Resolved from the pinned evaluation path (see spec.json step_I): every Step I
partition is matched to GT per image (Hungarian or exclusive majority vote) and
scored with foreground Dice; the dataset score is the mean and population std.
Never substitutes a Step II rule; it shares the per-image matching primitive.
"""
try:
    from .matching import per_image_remapped_dice
except ImportError:  # loaded by file path
    import importlib.util as _u, pathlib as _p
    _spec = _u.spec_from_file_location("dss_us_track_b__sibling", _p.Path(__file__).with_name("_sibling.py"))
    _sibling = _u.module_from_spec(_spec); _spec.loader.exec_module(_sibling)
    per_image_remapped_dice = _sibling.load("matching").per_image_remapped_dice


def per_image_segment_dice(pairs, *, n_classes):
    """pairs: iterable of (sample_id, sealed Step I raw partition, integer GT)."""
    return per_image_remapped_dice(pairs, n_classes)
