"""Step I segment Dice cannot be replaced by Step II matching."""
from shared_benchmark.native_protocol import ProtocolBlocked


def per_image_segment_dice(*args, **kwargs):
    raise ProtocolBlocked(["Step I original segment/GT pairing, unmatched handling and aggregation UNRESOLVED"])
