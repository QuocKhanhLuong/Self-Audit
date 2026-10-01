# SPDX-License-Identifier: GPL-3.0
"""Original-paper evaluator gate. Verify raw before any GT access."""
from pathlib import Path
from shared_benchmark.native_artifacts import verify_raw_run
from shared_benchmark.native_protocol import load_lock, value_hash, ProtocolBlocked


def evaluate_native(raw_root, *, protocol_path, gt_manifest):
    raw = verify_raw_run(Path(raw_root))
    config = load_lock(protocol_path, purpose="native_track_b")
    # This point is unreachable for current unresolved full-paper metric profiles.
    if raw["run"]["config_sha256"] != value_hash(config):
        raise ValueError("evaluator/producer protocol mismatch")
    raise ProtocolBlocked(["Complete HM/XOR/tie specification must be evidenced before GT is opened"])
