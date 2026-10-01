"""Native Step I / II gates; no GT is read before verified raw and locked evaluator."""
from pathlib import Path
from shared_benchmark.native_artifacts import verify_raw_run
from shared_benchmark.native_protocol import load_lock, ProtocolBlocked


def evaluate_native(raw_root, *, protocol_path, gt_manifest):
    raw = verify_raw_run(Path(raw_root))
    config = load_lock(protocol_path, purpose="native_track_b")
    if raw["run"]["stage"] != config["stage"]:
        raise ValueError("native evaluator stage mismatch")
    raise ProtocolBlocked(["Resolve complete original evaluator recipe before GT access"])
