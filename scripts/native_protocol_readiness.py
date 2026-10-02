"""Report native gates without data, producer imports or GT access."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.native_protocol import load_lock, ProtocolBlocked, native_status


if __name__ == "__main__":
    result = {}
    for name in ("DSS_US", "SGSCN"):
        profiles = {}
        for path in sorted((ROOT / "baseline" / name / "config/native").glob("*.yaml")):
            entry = {}
            for purpose in ("producer", "native_track_b"):
                try:
                    load_lock(path, purpose=purpose)
                    entry[purpose] = {"status": "REFERENCE_READY" if purpose == "producer" else "EVALUATOR_READY"}
                except ProtocolBlocked as error:
                    entry[purpose] = {"status": error.status, "reasons": error.reasons}
            profiles[path.stem] = entry
        result[name] = {"profiles": profiles, "statuses": native_status(
            producer_complete=False, track_b_status="BLOCKED_PROTOCOL",
            track_a_status="BLOCKED_ADAPTER", all_required_data=False, protocol_resolved=False)}
    print(json.dumps(result, indent=2))
