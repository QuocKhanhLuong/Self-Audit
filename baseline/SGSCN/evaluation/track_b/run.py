# SPDX-License-Identifier: GPL-3.0
"""Original-paper evaluator gate. Verify raw before any GT access."""
from pathlib import Path
from shared_benchmark.native_artifacts import verify_seal_receipt
from shared_benchmark.native_protocol import load_lock, value_hash, ProtocolBlocked


def evaluate_native(raw_root, *, seal_receipt, protocol_path, gt_manifest):
    # External finalized receipt + raw seal first; this evaluator never writes or reseals raw.
    raw = verify_seal_receipt(Path(raw_root), Path(seal_receipt))
    config = load_lock(protocol_path, purpose="native_track_b")
    # This point is unreachable for current unresolved full-paper metric profiles.
    if raw["run"]["config_sha256"] != value_hash(config):
        raise ValueError("evaluator/producer protocol mismatch")
    raise ProtocolBlocked(["Complete HM/XOR/tie specification must be evidenced before GT is opened"])
