# SPDX-License-Identifier: GPL-3.0
"""Paper-supported max-intersection selection and DSC; unresolved metrics gated."""
import numpy as np
from shared_benchmark.native_protocol import ProtocolBlocked


def max_overlap_dice(partition, foreground):
    if partition.shape != foreground.shape or foreground.dtype != np.bool_:
        raise ValueError("matched native grid and binary foreground required")
    ids = np.unique(partition)
    counts = np.array([np.count_nonzero((partition == i) & foreground) for i in ids])
    best = np.flatnonzero(counts == counts.max())
    if len(best) != 1:
        raise ProtocolBlocked(["Original max-overlap tie policy unresolved"])
    chosen = int(ids[best[0]])
    selected = partition == chosen
    denominator = np.count_nonzero(selected) + np.count_nonzero(foreground)
    return {"selected_raw_id": chosen,
            "dice": float(2 * np.count_nonzero(selected & foreground) / denominator)}
