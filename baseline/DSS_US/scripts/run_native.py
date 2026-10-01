"""Validate locked CAMUS profiles before any producer input or model access."""
import argparse
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from shared_benchmark.native_protocol import load_lock, ProtocolBlocked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-protocol", action="store_true")
    args = parser.parse_args()
    try:
        load_lock(args.config)
    except ProtocolBlocked as error:
        print(json.dumps({"status": error.status, "reasons": error.reasons}))
        return 2
    # Only a fully unblocked profile reaches this point: it must run in the canonical
    # environment before any producer input, model or output access.
    from environment_contract import require_official_environment
    require_official_environment()
    blocked = ProtocolBlocked(["Full native orchestration awaits evidenced row recipes and CRF correspondence"])
    print(json.dumps({"status": blocked.status, "reasons": blocked.reasons}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
