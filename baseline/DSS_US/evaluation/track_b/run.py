"""Native Step I / II gates; no GT is read before verified raw and locked evaluator."""
from pathlib import Path
from shared_benchmark.native_artifacts import verify_seal_receipt
from shared_benchmark.native_protocol import load_lock, ProtocolBlocked


def evaluate_native(raw_root, *, seal_receipt, protocol_path, gt_manifest):
    # External finalized receipt + raw seal first; this evaluator never writes or reseals raw.
    raw = verify_seal_receipt(Path(raw_root), Path(seal_receipt))
    config = load_lock(protocol_path, purpose="native_track_b")
    if raw["run"]["stage"] != config["stage"]:
        raise ValueError("native evaluator stage mismatch")
    raise ProtocolBlocked(["Resolve complete original evaluator recipe before GT access"])
